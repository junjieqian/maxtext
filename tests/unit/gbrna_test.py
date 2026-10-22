# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for GB.RNA (genbio-ai/GB.RNA-1.6B), an encoder-only BERT model with an MLM head.

The tests compare the full MaxText model (embedding, encoder layers, final LayerNorm, MLM head) with a
NumPy float64 copy of `RNABertForMaskedLM` from the GB.RNA `modeling_rnabert.py`. Both use the same weights.
"""

import math
import sys
import unittest

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

# Tiny GB.RNA config for the numerical tests.
_LAYERS = 2
_HIDDEN = 128
_HEADS = 4
_HEAD_DIM = 32
_FFN = 344
_VOCAB = 16
_SEQ = 64
_BATCH = 2
_EPS = 1e-5

# ------------------------------------------------------------------------------------------------
# NumPy float64 reference of RNABertForMaskedLM (modeling_rnabert.py) in eval mode.
# The weight names and layouts are the HF names and the torch layouts (Linear weight is [out, in]).
# ------------------------------------------------------------------------------------------------


def _round_to_bf16(x):
  """Rounds float32 values to the nearest bfloat16 value (round half to even)."""
  u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
  u = ((u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000).astype(np.uint32)
  return u.view(np.float32)


def make_hf_weights(seed):
  """Returns random GB.RNA weights with the HF names. All biases and LayerNorm params are nonzero."""
  rng = np.random.default_rng(seed)

  def normal(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float32)

  w = {}

  def linear(name, n_in, n_out):
    w[f"{name}.weight"] = normal(n_out, n_in, std=n_in**-0.5)
    w[f"{name}.bias"] = normal(n_out, std=0.1)

  def layer_norm(name):
    w[f"{name}.weight"] = 1.0 + normal(_HIDDEN, std=0.1)
    w[f"{name}.bias"] = normal(_HIDDEN, std=0.1)

  # MaxText computes the tied logits with a bfloat16 copy of the embedding table. Use bfloat16 values so
  # that the copy is exact.
  w["bert.embeddings.word_embeddings.weight"] = _round_to_bf16(normal(_VOCAB, _HIDDEN))
  for i in range(_LAYERS):
    p = f"bert.encoder.layer.{i}"
    layer_norm(f"{p}.attention.ln")
    linear(f"{p}.attention.self.query", _HIDDEN, _HIDDEN)
    linear(f"{p}.attention.self.key", _HIDDEN, _HIDDEN)
    linear(f"{p}.attention.self.value", _HIDDEN, _HIDDEN)
    linear(f"{p}.attention.output.dense", _HIDDEN, _HIDDEN)
    layer_norm(f"{p}.ln")
    linear(f"{p}.mlp.gate_proj", _HIDDEN, _FFN)
    linear(f"{p}.mlp.up_proj", _HIDDEN, _FFN)
    linear(f"{p}.mlp.down_proj", _FFN, _HIDDEN)
  layer_norm("bert.encoder.ln")
  linear("cls.predictions.transform.dense", _HIDDEN, _HIDDEN)
  layer_norm("cls.predictions.transform.LayerNorm")
  w["cls.predictions.bias"] = normal(_VOCAB, std=0.1)
  return w


def _layer_norm(x, weight, bias):
  mean = x.mean(-1, keepdims=True)
  var = np.square(x - mean).mean(-1, keepdims=True)
  return (x - mean) / np.sqrt(var + _EPS) * weight + bias


def _rope(x, positions, base=10000.0):
  """Megatron rotate-half RoPE on the full head_dim. x is [batch, seq, heads, head_dim]."""
  head_dim = x.shape[-1]
  inv_freq = 1.0 / base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
  freqs = positions[..., None] * inv_freq
  emb = np.concatenate([freqs, freqs], axis=-1)[:, :, None, :]
  x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
  return x * np.cos(emb) + np.concatenate([-x2, x1], axis=-1) * np.sin(emb)


def _softmax(x):
  e = np.exp(x - x.max(-1, keepdims=True))
  return e / e.sum(-1, keepdims=True)


def _gelu(x):
  """Exact (erf) GELU, as torch ACT2FN["gelu"]."""
  return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def reference_logits(w, input_ids, attention_mask=None):
  """Returns the MLM logits [batch, seq, vocab] of RNABertForMaskedLM in float64."""
  w = {k: np.asarray(v, np.float64) for k, v in w.items()}
  batch, seq = input_ids.shape
  if attention_mask is None:
    attention_mask = np.ones((batch, seq))
  # [batch, 1, query, key]. Keep a pair only if both tokens are real (bert_extended_attention_mask).
  keep = (attention_mask[:, None, :, None] * attention_mask[:, None, None, :]) > 0.5
  positions = np.broadcast_to(np.arange(seq, dtype=np.float64), (batch, seq))

  def linear(x, name):
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]

  def heads(x):
    return x.reshape(batch, seq, _HEADS, _HEAD_DIM)

  h = w["bert.embeddings.word_embeddings.weight"][input_ids]
  for i in range(_LAYERS):
    p = f"bert.encoder.layer.{i}"
    x = _layer_norm(h, w[f"{p}.attention.ln.weight"], w[f"{p}.attention.ln.bias"])
    q = _rope(heads(linear(x, f"{p}.attention.self.query")), positions)
    k = _rope(heads(linear(x, f"{p}.attention.self.key")), positions)
    v = heads(linear(x, f"{p}.attention.self.value"))
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(_HEAD_DIM)
    # No causal mask. Padding pairs get zero probability.
    probs = np.where(keep, _softmax(np.where(keep, scores, -1e30)), 0.0)
    context = np.einsum("bhqk,bkhd->bqhd", probs, v).reshape(batch, seq, _HIDDEN)
    h = h + linear(context, f"{p}.attention.output.dense")
    x = _layer_norm(h, w[f"{p}.ln.weight"], w[f"{p}.ln.bias"])
    gate = linear(x, f"{p}.mlp.gate_proj")
    h = h + linear(gate / (1.0 + np.exp(-gate)) * linear(x, f"{p}.mlp.up_proj"), f"{p}.mlp.down_proj")
  h = _layer_norm(h, w["bert.encoder.ln.weight"], w["bert.encoder.ln.bias"])
  t = _gelu(linear(h, "cls.predictions.transform.dense"))
  t = _layer_norm(t, w["cls.predictions.transform.LayerNorm.weight"], w["cls.predictions.transform.LayerNorm.bias"])
  return t @ w["bert.embeddings.word_embeddings.weight"].T + w["cls.predictions.bias"]


# ------------------------------------------------------------------------------------------------
# HF -> MaxText weight mapping.
# ------------------------------------------------------------------------------------------------


def _maxtext_layer_weights(w, i):
  """Returns the MaxText weights of encoder layer i as {attribute path: array}."""
  p = f"bert.encoder.layer.{i}"
  # MaxText does not scale the attention logits. It folds 1/sqrt(head_dim) into the query projection,
  # so scale the query kernel and the query bias.
  q_scale = 1.0 / math.sqrt(_HEAD_DIM)

  def kernel(name):
    return w[f"{name}.weight"].T  # [in, out]

  qkv_shape = (_HIDDEN, _HEADS, _HEAD_DIM)
  return {
      # Gpt3LayerNorm applies `scale + 1`.
      ("pre_self_attention_layer_norm", "scale"): w[f"{p}.attention.ln.weight"] - 1.0,
      ("pre_self_attention_layer_norm", "bias"): w[f"{p}.attention.ln.bias"],
      ("self_attention", "query", "kernel"): (kernel(f"{p}.attention.self.query") * q_scale).reshape(qkv_shape),
      ("self_attention", "query", "bias"): (w[f"{p}.attention.self.query.bias"] * q_scale).reshape(_HEADS, _HEAD_DIM),
      ("self_attention", "key", "kernel"): kernel(f"{p}.attention.self.key").reshape(qkv_shape),
      ("self_attention", "key", "bias"): w[f"{p}.attention.self.key.bias"].reshape(_HEADS, _HEAD_DIM),
      ("self_attention", "value", "kernel"): kernel(f"{p}.attention.self.value").reshape(qkv_shape),
      ("self_attention", "value", "bias"): w[f"{p}.attention.self.value.bias"].reshape(_HEADS, _HEAD_DIM),
      ("self_attention", "out", "kernel"): kernel(f"{p}.attention.output.dense").reshape(_HEADS, _HEAD_DIM, _HIDDEN),
      ("self_attention", "out", "bias"): w[f"{p}.attention.output.dense.bias"],
      ("post_self_attention_layer_norm", "scale"): w[f"{p}.ln.weight"] - 1.0,
      ("post_self_attention_layer_norm", "bias"): w[f"{p}.ln.bias"],
      ("mlp", "wi_0", "kernel"): kernel(f"{p}.mlp.gate_proj"),
      ("mlp", "wi_0", "bias"): w[f"{p}.mlp.gate_proj.bias"],
      ("mlp", "wi_1", "kernel"): kernel(f"{p}.mlp.up_proj"),
      ("mlp", "wi_1", "bias"): w[f"{p}.mlp.up_proj.bias"],
      ("mlp", "wo", "kernel"): kernel(f"{p}.mlp.down_proj"),
      ("mlp", "wo", "bias"): w[f"{p}.mlp.down_proj.bias"],
  }


def _get(module, path):
  for name in path:
    module = getattr(module, name)
  return module


def _assign(param, value):
  value = jnp.asarray(value, param[...].dtype)
  if param[...].shape != value.shape:
    raise ValueError(f"Shape mismatch: param {param[...].shape} vs value {value.shape}")
  param[...] = value


def load_hf_weights(model, cfg, w):
  """Loads HF GB.RNA weights into a MaxText `Transformer` (scanned or unscanned layers)."""
  decoder = model.decoder
  _assign(model.token_embedder.embedding, w["bert.embeddings.word_embeddings.weight"])
  per_layer = [_maxtext_layer_weights(w, i) for i in range(_LAYERS)]
  for path in per_layer[0]:
    if cfg.scan_layers:
      stacked = np.stack([layer[path] for layer in per_layer], axis=cfg.param_scan_axis)
      _assign(_get(decoder.layers, path), stacked)
    else:
      for i, layer in enumerate(per_layer):
        _assign(_get(getattr(decoder, f"layers_{i}"), path), layer[path])
  _assign(decoder.decoder_norm.scale, w["bert.encoder.ln.weight"] - 1.0)
  _assign(decoder.decoder_norm.bias, w["bert.encoder.ln.bias"])
  _assign(decoder.mlm_head_dense.kernel, w["cls.predictions.transform.dense.weight"].T)
  _assign(decoder.mlm_head_dense.bias, w["cls.predictions.transform.dense.bias"])
  _assign(decoder.mlm_head_norm.scale, w["cls.predictions.transform.LayerNorm.weight"] - 1.0)
  _assign(decoder.mlm_head_norm.bias, w["cls.predictions.transform.LayerNorm.bias"])
  _assign(decoder.mlm_head_bias, w["cls.predictions.bias"])


def _tiny_config(scan_layers):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="gbrna_test",
      model_name="gbrna-1.6b",
      override_model_config=True,
      enable_checkpointing=False,
      skip_jax_distributed_system=True,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      attention="dot_product",
      scan_layers=scan_layers,
      base_num_decoder_layers=_LAYERS,
      base_emb_dim=_HIDDEN,
      base_num_query_heads=_HEADS,
      base_num_kv_heads=_HEADS,
      head_dim=_HEAD_DIM,
      base_mlp_dim=_FFN,
      max_target_length=_SEQ,
      per_device_batch_size=_BATCH,
  )


class GbRnaTest(parameterized.TestCase):
  """Numerical and structural tests for GB.RNA."""

  def _build(self, scan_layers):
    cfg = _tiny_config(scan_layers)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg, devices=[jax.devices()[0]]), cfg.mesh_axes)
    model = models.Transformer(cfg, mesh, quant=None, rngs=nnx.Rngs(params=0, dropout=0))
    weights = make_hf_weights(seed=1234)
    load_hf_weights(model, cfg, weights)
    return model, mesh, weights

  def _logits(self, model, mesh, input_ids, segment_ids):
    positions = np.broadcast_to(np.arange(input_ids.shape[1], dtype=np.int32), input_ids.shape)
    with jax.set_mesh(mesh):
      logits = model(
          decoder_input_tokens=jnp.asarray(input_ids, jnp.int32),
          decoder_positions=jnp.asarray(positions),
          decoder_segment_ids=jnp.asarray(segment_ids, jnp.int32),
          enable_dropout=False,
      )
    return np.asarray(logits, np.float64)

  def _input_ids(self):
    return np.random.default_rng(7).integers(0, _VOCAB, size=(_BATCH, _SEQ))

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_logits_match_reference(self, scan_layers):
    """The full model gives the logits of the reference implementation."""
    model, mesh, weights = self._build(scan_layers)
    input_ids = self._input_ids()
    logits = self._logits(model, mesh, input_ids, np.ones_like(input_ids))
    expected = reference_logits(weights, input_ids)
    np.testing.assert_allclose(logits, expected, rtol=1e-4, atol=1e-4)

  def test_padding_mask_matches_reference(self):
    """Padding tokens (segment id 0) do not change the logits of the real tokens."""
    model, mesh, weights = self._build(scan_layers=False)
    input_ids = self._input_ids()
    attention_mask = np.ones_like(input_ids)
    attention_mask[1, _SEQ - 9 :] = 0
    logits = self._logits(model, mesh, input_ids, attention_mask)
    expected = reference_logits(weights, input_ids, attention_mask=attention_mask)
    valid = attention_mask.astype(bool)
    np.testing.assert_allclose(logits[valid], expected[valid], rtol=1e-4, atol=1e-4)

  def test_attention_is_bidirectional(self):
    """A change to the last token changes the logits at earlier positions."""
    model, mesh, _ = self._build(scan_layers=False)
    input_ids = self._input_ids()
    changed_ids = input_ids.copy()
    changed_ids[:, -1] = (changed_ids[:, -1] + 1) % _VOCAB
    segment_ids = np.ones_like(input_ids)
    logits = self._logits(model, mesh, input_ids, segment_ids)
    changed_logits = self._logits(model, mesh, changed_ids, segment_ids)
    self.assertGreater(np.abs(changed_logits[:, 0, :] - logits[:, 0, :]).max(), 1e-3)

  def test_fused_attention_matches_default_path(self):
    """`gbrna_fused_attention` (Pallas kernel, interpret mode) gives the logits and grads of the default path."""

    def build(fused):
      cfg = pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="gbrna_test",
          model_name="gbrna-1.6b",
          override_model_config=True,
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          dtype="float32",
          weight_dtype="float32",
          matmul_precision="highest",
          attention="dot_product",
          scan_layers=False,
          base_num_decoder_layers=1,
          base_emb_dim=128,
          base_num_query_heads=2,
          base_num_kv_heads=2,
          head_dim=64,
          base_mlp_dim=256,
          max_target_length=128,
          per_device_batch_size=1,
          gbrna_fused_attention=fused,
          gbrna_attention_heads_per_step=2,
          gbrna_attention_block_q=128,
          gbrna_attention_block_q_bwd=64,
      )
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg, devices=[jax.devices()[0]]), cfg.mesh_axes)
      return models.Transformer(cfg, mesh, quant=None, rngs=nnx.Rngs(params=0, dropout=0)), mesh

    ids = jnp.asarray(np.random.default_rng(3).integers(0, _VOCAB, size=(1, 128)), jnp.int32)
    pos = jnp.arange(128, dtype=jnp.int32)[None]
    seg = jnp.ones_like(ids)
    results = []
    for fused in (False, True):
      model, mesh = build(fused)

      def loss(m):
        logits = m(decoder_input_tokens=ids, decoder_positions=pos, decoder_segment_ids=seg, enable_dropout=False)
        return jnp.mean(jax.nn.log_softmax(logits)[..., 0]), logits

      with jax.set_mesh(mesh):
        (_, logits), grads = nnx.value_and_grad(loss, has_aux=True)(model)
      grads = nnx.state(grads, nnx.Param)
      results.append((logits, grads))
    (ref_logits, ref_grads), (logits, grads) = results
    np.testing.assert_allclose(logits, ref_logits, rtol=1e-4, atol=1e-4)
    for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(ref_grads)):
      np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-5)

  def test_full_model_param_count(self):
    """gbrna-1.6b has the parameter count of the HF checkpoint (1,611,593,744)."""
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="gbrna_test",
        model_name="gbrna-1.6b",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        max_target_length=1024,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg, devices=[jax.devices()[0]]), cfg.mesh_axes)
    abstract_model = nnx.eval_shape(lambda: models.Transformer(cfg, mesh, quant=None, rngs=nnx.Rngs(params=0)))
    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(abstract_model, nnx.Param)))
    self.assertEqual(num_params, 1_611_593_744)


if __name__ == "__main__":
  unittest.main()
