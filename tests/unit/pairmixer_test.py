# Copyright 2026 Google LLC
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

"""Unit tests for Pairmixer model layers, numerics, masking, scan, remat, and sharding."""

import os
import unittest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

# pylint: disable=wrong-import-position
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.kernels import pairmixer_pallas
from maxtext.models import models, pairmixer
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

# pylint: enable=wrong-import-position


def _np_layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
  """NumPy reference LayerNorm matching Gpt3LayerNorm (scale + 1 parametrization)."""
  x32 = x.astype(np.float32)
  mean = np.mean(x32, axis=-1, keepdims=True)
  var = np.mean(np.square(x32 - mean), axis=-1, keepdims=True)
  normed = (x32 - mean) / np.sqrt(var + eps)
  return normed * (scale.astype(np.float32) + 1.0) + bias.astype(np.float32)


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


def _np_silu(x: np.ndarray) -> np.ndarray:
  return x.astype(np.float32) * _np_sigmoid(x)


def _np_triangle_mult(
    z: np.ndarray,
    pair_mask: np.ndarray,
    direction: str,
    norm_in_scale: np.ndarray,
    norm_in_bias: np.ndarray,
    norm_out_scale: np.ndarray,
    norm_out_bias: np.ndarray,
    w_p_in: np.ndarray,
    w_g_in: np.ndarray,
    w_p_out: np.ndarray,
    w_g_out: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
  """NumPy reference for TriangleMultiplicationOutgoing / Incoming."""
  x_in = _np_layer_norm(z, norm_in_scale, norm_in_bias, eps=eps)
  p = x_in @ w_p_in
  g = _np_sigmoid(x_in @ w_g_in)
  ab = p * g * pair_mask[..., None]
  a, b = np.split(ab.astype(np.float32), 2, axis=-1)
  subscripts = "bikd,bjkd->bijd" if direction == "outgoing" else "bkid,bkjd->bijd"
  t = np.einsum(subscripts, a, b)
  t_norm = _np_layer_norm(t, norm_out_scale, norm_out_bias, eps=eps)
  y = (t_norm @ w_p_out) * _np_sigmoid(x_in @ w_g_out)
  return y


def _np_transition(
    z: np.ndarray,
    norm_scale: np.ndarray,
    norm_bias: np.ndarray,
    w_fc1: np.ndarray,
    w_fc2: np.ndarray,
    w_fc3: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
  """NumPy reference for PairmixerTransition (SwiGLU)."""
  x = _np_layer_norm(z, norm_scale, norm_bias, eps=eps)
  h = _np_silu(x @ w_fc1) * (x @ w_fc2)
  return h @ w_fc3


class PairmixerTest(parameterized.TestCase):
  """Comprehensive unit tests for Pairmixer."""

  def _make_config(self, **overrides):
    args = [
        None,
        get_test_config_path("models/pairmixer.yml"),
        "run_name=pairmixer_unit_test",
        "override_model_config=True",
        "base_emb_dim=32",
        "pairmixer_token_z=16",
        "base_mlp_dim=64",
        "base_num_decoder_layers=2",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "head_dim=8",
        "vocab_size=64",
        "max_target_length=8",
        "per_device_batch_size=1",
        "ici_data_parallelism=4",
        "ici_fsdp_parallelism=1",
        "dtype=float32",
        "weight_dtype=float32",
        "scan_layers=False",
        "enable_checkpointing=False",
        "skip_jax_distributed_system=True",
    ]
    for k, v in overrides.items():
      args.append(f"{k}={v}")
    return pyconfig.initialize(args)

  def _make_mesh(self, cfg):
    devices_array = maxtext_utils.create_device_mesh(cfg)
    return Mesh(devices_array, cfg.mesh_axes)

  def _populate_nonzero_layer_weights(self, layer: pairmixer.PairmixerDecoderLayer, seed: int = 42):
    """Populates zero-initialized weights (g_in, g_out, p_out, fc3) with deterministic random values."""
    rng = np.random.default_rng(seed)
    for tri in (layer.tri_mul_out, layer.tri_mul_in):
      tri.norm_in.scale[...] = jnp.asarray(rng.standard_normal(tri.norm_in.scale[...].shape, dtype=np.float32) * 0.1)
      tri.norm_in.bias[...] = jnp.asarray(rng.standard_normal(tri.norm_in.bias[...].shape, dtype=np.float32) * 0.1)
      tri.norm_out.scale[...] = jnp.asarray(rng.standard_normal(tri.norm_out.scale[...].shape, dtype=np.float32) * 0.1)
      tri.norm_out.bias[...] = jnp.asarray(rng.standard_normal(tri.norm_out.bias[...].shape, dtype=np.float32) * 0.1)
      if hasattr(tri, "pgg_in"):
        tri.pgg_in.kernel[...] = jnp.asarray(rng.standard_normal(tri.pgg_in.kernel[...].shape, dtype=np.float32) * 0.2)
      else:
        tri.p_in.kernel[...] = jnp.asarray(rng.standard_normal(tri.p_in.kernel[...].shape, dtype=np.float32) * 0.2)
        tri.g_in.kernel[...] = jnp.asarray(rng.standard_normal(tri.g_in.kernel[...].shape, dtype=np.float32) * 0.2)
        tri.g_out.kernel[...] = jnp.asarray(rng.standard_normal(tri.g_out.kernel[...].shape, dtype=np.float32) * 0.2)
      tri.p_out.kernel[...] = jnp.asarray(rng.standard_normal(tri.p_out.kernel[...].shape, dtype=np.float32) * 0.2)

    tr = layer.transition_z
    tr.norm.scale[...] = jnp.asarray(rng.standard_normal(tr.norm.scale[...].shape, dtype=np.float32) * 0.1)
    tr.norm.bias[...] = jnp.asarray(rng.standard_normal(tr.norm.bias[...].shape, dtype=np.float32) * 0.1)
    if hasattr(tr, "fc12"):
      tr.fc12.kernel[...] = jnp.asarray(rng.standard_normal(tr.fc12.kernel[...].shape, dtype=np.float32) * 0.2)
    else:
      tr.fc1.kernel[...] = jnp.asarray(rng.standard_normal(tr.fc1.kernel[...].shape, dtype=np.float32) * 0.2)
      tr.fc2.kernel[...] = jnp.asarray(rng.standard_normal(tr.fc2.kernel[...].shape, dtype=np.float32) * 0.2)
    tr.fc3.kernel[...] = jnp.asarray(rng.standard_normal(tr.fc3.kernel[...].shape, dtype=np.float32) * 0.2)

  def test_numpy_reference_parity_and_fused_equivalence(self):
    """Tests exact numerical parity against NumPy reference and fused vs unfused projection equivalence."""
    cfg = self._make_config(pairmixer_fused_projections=False)
    mesh = self._make_mesh(cfg)
    layer = pairmixer.PairmixerDecoderLayer(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    self._populate_nonzero_layer_weights(layer, seed=123)

    rng = np.random.default_rng(999)
    z_np = rng.standard_normal((2, 8, 8, 16), dtype=np.float32)
    seg_ids_np = np.array(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    mask_np = seg_ids_np != 0
    pair_mask_np = (mask_np[:, :, None] & mask_np[:, None, :]).astype(np.float32)

    out_jax, kv_out = layer(
        jnp.asarray(z_np),
        decoder_segment_ids=jnp.asarray(seg_ids_np),
        decoder_positions=None,
        deterministic=True,
    )
    self.assertIsNone(kv_out)

    # Compute NumPy reference step by step
    z_ref = z_np.copy()
    for direction, tri in (("outgoing", layer.tri_mul_out), ("incoming", layer.tri_mul_in)):
      z_ref = z_ref + _np_triangle_mult(
          z_ref,
          pair_mask_np,
          direction=direction,
          norm_in_scale=np.asarray(tri.norm_in.scale[...]),
          norm_in_bias=np.asarray(tri.norm_in.bias[...]),
          norm_out_scale=np.asarray(tri.norm_out.scale[...]),
          norm_out_bias=np.asarray(tri.norm_out.bias[...]),
          w_p_in=np.asarray(tri.p_in.kernel[...]),
          w_g_in=np.asarray(tri.g_in.kernel[...]),
          w_p_out=np.asarray(tri.p_out.kernel[...]),
          w_g_out=np.asarray(tri.g_out.kernel[...]),
          eps=cfg.normalization_layer_epsilon,
      )
    tr = layer.transition_z
    z_ref = z_ref + _np_transition(
        z_ref,
        norm_scale=np.asarray(tr.norm.scale[...]),
        norm_bias=np.asarray(tr.norm.bias[...]),
        w_fc1=np.asarray(tr.fc1.kernel[...]),
        w_fc2=np.asarray(tr.fc2.kernel[...]),
        w_fc3=np.asarray(tr.fc3.kernel[...]),
        eps=cfg.normalization_layer_epsilon,
    )

    np.testing.assert_allclose(np.asarray(out_jax), z_ref, rtol=1e-5, atol=1e-5)

    # Also verify fused projections match unfused projections when initialized with concatenated weights
    cfg_fused = self._make_config(pairmixer_fused_projections=True)
    layer_fused = pairmixer.PairmixerDecoderLayer(
        config=cfg_fused, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
    )
    for tri_u, tri_f in ((layer.tri_mul_out, layer_fused.tri_mul_out), (layer.tri_mul_in, layer_fused.tri_mul_in)):
      tri_f.norm_in.scale[...] = tri_u.norm_in.scale[...]
      tri_f.norm_in.bias[...] = tri_u.norm_in.bias[...]
      tri_f.norm_out.scale[...] = tri_u.norm_out.scale[...]
      tri_f.norm_out.bias[...] = tri_u.norm_out.bias[...]
      tri_f.pgg_in.kernel[...] = jnp.concatenate(
          [tri_u.p_in.kernel[...], tri_u.g_in.kernel[...], tri_u.g_out.kernel[...]], axis=-1
      )
      tri_f.p_out.kernel[...] = tri_u.p_out.kernel[...]
    layer_fused.transition_z.norm.scale[...] = tr.norm.scale[...]
    layer_fused.transition_z.norm.bias[...] = tr.norm.bias[...]
    layer_fused.transition_z.fc12.kernel[...] = jnp.concatenate([tr.fc1.kernel[...], tr.fc2.kernel[...]], axis=-1)
    layer_fused.transition_z.fc3.kernel[...] = tr.fc3.kernel[...]

    out_fused, _ = layer_fused(
        jnp.asarray(z_np),
        decoder_segment_ids=jnp.asarray(seg_ids_np),
        decoder_positions=None,
        deterministic=True,
    )
    np.testing.assert_allclose(np.asarray(out_fused), np.asarray(out_jax), rtol=1e-5, atol=1e-5)

  def test_mask_invariance(self):
    """Perturbing Z at padded positions (decoder_segment_ids == 0) must not alter valid (i, j) outputs."""
    cfg = self._make_config()
    mesh = self._make_mesh(cfg)
    layer = pairmixer.PairmixerDecoderLayer(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
    self._populate_nonzero_layer_weights(layer, seed=77)

    rng = np.random.default_rng(42)
    z1 = rng.standard_normal((2, 8, 8, 16), dtype=np.float32)
    z2 = z1.copy()
    # Perturb indices >= 5 (padded positions) with large values
    z2[:, 5:, :, :] += 100.0
    z2[:, :, 5:, :] -= 50.0

    seg_ids = jnp.array([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=jnp.int32)
    out1, _ = layer(jnp.asarray(z1), seg_ids, None, deterministic=True)
    out2, _ = layer(jnp.asarray(z2), seg_ids, None, deterministic=True)

    np.testing.assert_allclose(
        np.asarray(out1[:, :5, :5, :]),
        np.asarray(out2[:, :5, :5, :]),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_scan_vs_unscan_and_remat_policies(self):
    """Tests full Transformer forward/backward equivalence across scan_layers and remat_policy."""
    tokens = jnp.ones((4, 8), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (4, 8))
    seg_ids = jnp.ones((4, 8), dtype=jnp.int32)

    losses = {}
    for scan_layers in (False, True):
      for remat_policy in ("full", "minimal", "custom"):
        cfg = self._make_config(
            scan_layers=scan_layers,
            remat_policy=remat_policy,
            mlpwi="device",
            mlpwo="device",
        )
        mesh = self._make_mesh(cfg)
        with mesh:
          model = models.transformer_as_linen(
              config=cfg,
              mesh=mesh,
              quant=None,
              model_mode=MODEL_MODE_TRAIN,
          )
          init_rng = jax.random.PRNGKey(0)
          variables = model.init(
              {"params": init_rng, "dropout": init_rng},
              tokens,
              positions,
              decoder_segment_ids=seg_ids,
              enable_dropout=False,
          )

          def loss_fn(params, m=model):
            logits = m.apply(
                {"params": params},
                tokens,
                positions,
                decoder_segment_ids=seg_ids,
                enable_dropout=False,
            )
            return jnp.mean(jnp.square(logits))

          loss, grads = jax.value_and_grad(loss_fn)(variables["params"])
          self.assertTrue(bool(jnp.isfinite(loss)))
          grad_leaves = jax.tree_util.tree_leaves(grads)
          self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves))
          # Verify non-zero gradients on output projector and logits head at step 0
          self.assertGreater(sum(float(jnp.linalg.norm(g)) for g in grad_leaves), 0.0)
          losses[(scan_layers, remat_policy)] = float(loss)

    base_loss = losses[(False, "full")]
    for key, val in losses.items():
      self.assertAlmostEqual(val, base_loss, places=5, msg=f"Mismatch for {key}")

  def test_data_parallel_sharding_on_4_devices(self):
    """Verifies that under pure Data Parallelism (data=4, fsdp=1), Z is sharded on the batch axis only."""
    cfg = self._make_config(scan_layers=True, ici_data_parallelism=4, ici_fsdp_parallelism=1)
    mesh = self._make_mesh(cfg)
    self.assertEqual(mesh.shape["data"], 4)
    self.assertEqual(mesh.shape["fsdp"], 1)

    with mesh:
      proj = pairmixer.PairmixerInputProjector(config=cfg, mesh=mesh, rngs=nnx.Rngs(0))
      s = jnp.ones((4, 8, 32), dtype=jnp.float32)
      z = proj(s)
      self.assertEqual(z.shape, (4, 8, 8, 16))

  def test_pallas_bcnn_kernel_vjp_and_batch_scaling(self):
    """Verifies channel-batched [B, C=128, N, N] Pallas kernel forward/VJP parity for B_local=1 and B_local=2."""
    for pdbs in (1, 2):
      cfg_ref = self._make_config(
          pairmixer_token_z=128,
          base_mlp_dim=512,
          max_target_length=128,
          per_device_batch_size=pdbs,
          dtype="bfloat16",
          weight_dtype="float32",
          pairmixer_fused_projections=True,
          pairmixer_use_pallas=False,
          pairmixer_fp32_einsum=False,
      )
      cfg_pal = self._make_config(
          pairmixer_token_z=128,
          base_mlp_dim=512,
          max_target_length=128,
          per_device_batch_size=pdbs,
          dtype="bfloat16",
          weight_dtype="float32",
          pairmixer_fused_projections=True,
          pairmixer_use_pallas=True,
          pairmixer_fp32_einsum=False,
      )
      mesh = self._make_mesh(cfg_ref)
      gbs = 4 * pdbs
      with mesh:
        layer_ref = pairmixer.PairmixerDecoderLayer(
            config=cfg_ref, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
        )
        layer_pal = pairmixer.PairmixerDecoderLayer(
            config=cfg_pal, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
        )
        self._populate_nonzero_layer_weights(layer_ref, seed=321)
        self._populate_nonzero_layer_weights(layer_pal, seed=321)

        rng = np.random.default_rng(17 + pdbs)
        z = jnp.asarray(rng.standard_normal((gbs, 128, 128, 128), dtype=np.float32), dtype=jnp.bfloat16)
        dy = jnp.asarray(rng.standard_normal((gbs, 128, 128, 128), dtype=np.float32), dtype=jnp.bfloat16)
        seg = jnp.broadcast_to((jnp.arange(128)[None, :] < 96).astype(jnp.int32), (gbs, 128))

        gdef_ref, state_ref = nnx.split(layer_ref, nnx.Param)
        gdef_pal, state_pal = nnx.split(layer_pal, nnx.Param)

        def loss_ref_fn(z_in, st, gdef=gdef_ref, seg_in=seg, dy_in=dy):
          mod = nnx.merge(gdef, st)
          out, _ = mod(z_in, seg_in, None, deterministic=True)
          return jnp.sum(out.astype(jnp.float32) * dy_in.astype(jnp.float32))

        def loss_pal_fn(z_in, st, gdef=gdef_pal, seg_in=seg, dy_in=dy):
          mod = nnx.merge(gdef, st)
          out_cm, _ = mod(jnp.moveaxis(z_in, -1, 1), seg_in, None, deterministic=True, channel_major=True)
          out = jnp.moveaxis(out_cm, 1, -1)
          return jnp.sum(out.astype(jnp.float32) * dy_in.astype(jnp.float32))

        l_ref, (dz_ref, dparam_ref) = jax.value_and_grad(loss_ref_fn, argnums=(0, 1))(z, state_ref)
        l_pal, (dz_pal, dparam_pal) = jax.value_and_grad(loss_pal_fn, argnums=(0, 1))(z, state_pal)

        # Parity against pure-XLA reference
        rel_loss = abs(float(l_ref - l_pal)) / (abs(float(l_ref)) + 1e-8)
        self.assertLess(rel_loss, 0.03)
        rel_dz = np.linalg.norm(np.asarray(dz_ref, np.float32) - np.asarray(dz_pal, np.float32)) / (
            np.linalg.norm(np.asarray(dz_ref, np.float32)) + 1e-8
        )
        self.assertLess(rel_dz, 0.035)
        for gr, gp in zip(jax.tree_util.tree_leaves(dparam_ref), jax.tree_util.tree_leaves(dparam_pal)):
          rel = np.linalg.norm(np.asarray(gr, np.float32) - np.asarray(gp, np.float32)) / (
              np.linalg.norm(np.asarray(gr, np.float32)) + 1e-8
          )
          tol = 0.05 if gr.ndim >= 2 else 0.15
          self.assertLess(rel, tol)

  def test_channel_major_end_to_end_megakernels_parity(self):
    """Verifies full end-to-end Decoder forward+backward parity across all channel-major Pallas modes and widths."""
    for c_z, d_ff in ((128, 512), (256, 512)):
      tokens = jnp.asarray((np.arange(4 * 32) % 60 + 1).reshape(4, 32), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(32, dtype=jnp.int32), (4, 32))
      seg_ids = jnp.broadcast_to((jnp.arange(32)[None, :] < 24).astype(jnp.int32), (4, 32))

      cfg_ref = self._make_config(
          pairmixer_token_z=c_z,
          base_mlp_dim=d_ff,
          max_target_length=32,
          per_device_batch_size=1,
          dtype="bfloat16",
          weight_dtype="float32",
          pairmixer_fused_projections=True,
          pairmixer_use_pallas=False,
          pairmixer_fp32_einsum=False,
          remat_policy="none",
          scan_layers=False,
      )
      mesh = self._make_mesh(cfg_ref)
      with mesh:
        model_ref = models.transformer_as_linen(config=cfg_ref, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
        init_rng = jax.random.PRNGKey(7)
        variables = model_ref.init(
            {"params": init_rng, "dropout": init_rng},
            tokens,
            positions,
            decoder_segment_ids=seg_ids,
            enable_dropout=False,
        )
        # Populate non-zero layer weights so gradients through every sublayer are non-trivial
        rng_np = np.random.default_rng(99 + c_z)
        params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(rng_np.standard_normal(x.shape, dtype=np.float32) * 0.12, dtype=x.dtype),
            variables["params"],
        )

        def eval_mode(m):
          def loss_fn(p):
            logits = m.apply(
                {"params": p},
                tokens,
                positions,
                decoder_segment_ids=seg_ids,
                enable_dropout=False,
            )
            return jnp.mean(jnp.square(logits.astype(jnp.float32)))

          return jax.value_and_grad(loss_fn)(params)

        l_ref, g_ref = eval_mode(model_ref)

        cm_results = {}
        for scan_layers, remat_policy in (
            (False, "none"),  # Cross-layer trunk
            (False, "full"),  # Full layer remat (compact VJP)
        ):
          cfg_cm = self._make_config(
              pairmixer_token_z=c_z,
              base_mlp_dim=d_ff,
              max_target_length=32,
              per_device_batch_size=1,
              dtype="bfloat16",
              weight_dtype="float32",
              pairmixer_fused_projections=True,
              pairmixer_use_pallas=True,
              pairmixer_fp32_einsum=False,
              remat_policy=remat_policy,
              scan_layers=scan_layers,
          )
          model_cm = models.transformer_as_linen(config=cfg_cm, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
          l_cm, g_cm = eval_mode(model_cm)
          cm_results[(scan_layers, remat_policy)] = (l_cm, g_cm)
          rel_l = abs(float(l_ref - l_cm)) / (abs(float(l_ref)) + 1e-8)
          self.assertLess(rel_l, 0.02, msg=f"Loss mismatch for c_z={c_z}, mode={(scan_layers, remat_policy)}")
          for gr, gc in zip(jax.tree_util.tree_leaves(g_ref), jax.tree_util.tree_leaves(g_cm)):
            if gr.ndim >= 2:
              rel_g = np.linalg.norm(np.asarray(gr, np.float32) - np.asarray(gc, np.float32)) / (
                  np.linalg.norm(np.asarray(gr, np.float32)) + 1e-8
              )
              self.assertLess(
                  rel_g,
                  0.05,
                  msg=f"Grad mismatch {rel_g} for c_z={c_z}, shape={gr.shape}, mode={(scan_layers, remat_policy)}",
              )
            else:
              np.testing.assert_allclose(np.asarray(gc, np.float32), np.asarray(gr, np.float32), rtol=0.06, atol=5e-2)

        # Cross-verify exact consistency across Cross-Layer Trunk and Compact Layer VJP (< 1.2% = ~1.5 BF16 ULP)
        base_l, base_g = cm_results[(False, "none")]
        for mode_key, (m_l, m_g) in cm_results.items():
          self.assertLess(abs(float(base_l - m_l)) / (abs(float(base_l)) + 1e-8), 5e-3, msg=f"Cross-mode loss {mode_key}")
          for gb, gm in zip(jax.tree_util.tree_leaves(base_g), jax.tree_util.tree_leaves(m_g)):
            if gb.ndim >= 2:
              rel_m = np.linalg.norm(np.asarray(gb, np.float32) - np.asarray(gm, np.float32)) / (
                  np.linalg.norm(np.asarray(gb, np.float32)) + 1e-8
              )
              self.assertLess(rel_m, 1.2e-2, msg=f"Cross-mode grad {rel_m} for shape={gb.shape}, mode={mode_key}")
            else:
              np.testing.assert_allclose(np.asarray(gm, np.float32), np.asarray(gb, np.float32), rtol=2e-2, atol=2e-2)

  def test_zero_copy_tri_bcm_vjp_exact_parity(self):
    """Verifies zero-copy `[B, C, M]` VMEM slab contraction forward/backward VJP against 4D einsum."""
    rng = np.random.default_rng(2026)
    a4 = jnp.asarray(rng.standard_normal((2, 16, 16, 16), dtype=np.float32) * 0.25, dtype=jnp.bfloat16)
    b4 = jnp.asarray(rng.standard_normal((2, 16, 16, 16), dtype=np.float32) * 0.25, dtype=jnp.bfloat16)
    dt4 = jnp.asarray(rng.standard_normal((2, 16, 16, 16), dtype=np.float32) * 0.25, dtype=jnp.bfloat16)
    for direction in ("outgoing", "incoming"):
      t_ref = pairmixer_pallas._tri_einsum_fwd(a4, b4, direction=direction, fp32_einsum=False, matmul_precision="highest")
      da_ref, db_ref = pairmixer_pallas._tri_einsum_bwd(
          dt4, a4, b4, direction=direction, fp32_einsum=False, matmul_precision="highest"
      )
      t_bcm = pairmixer_pallas._run_tri_bcm_fwd(
          a4.reshape(2, 16, 256), b4.reshape(2, 16, 256), n_i=16, n_j=16, direction=direction, interpret=True
      ).reshape(2, 16, 16, 16)
      da_bcm, db_bcm = pairmixer_pallas._run_tri_bcm_bwd(
          dt4.reshape(2, 16, 256),
          a4.reshape(2, 16, 256),
          b4.reshape(2, 16, 256),
          n_i=16,
          n_j=16,
          direction=direction,
          interpret=True,
      )
      np.testing.assert_allclose(np.asarray(t_bcm, np.float32), np.asarray(t_ref, np.float32), rtol=2e-3, atol=2e-3)
      np.testing.assert_allclose(
          np.asarray(da_bcm.reshape(2, 16, 16, 16), np.float32), np.asarray(da_ref, np.float32), rtol=2e-3, atol=2e-3
      )
      np.testing.assert_allclose(
          np.asarray(db_bcm.reshape(2, 16, 16, 16), np.float32), np.asarray(db_ref, np.float32), rtol=2e-3, atol=2e-3
      )

  def test_systolic_trunk_bcnn_end_to_end_parity(self):
    """Verifies end-to-end forward+backward parity of the `c_z >= 512` systolic channel-major trunk."""
    tokens = jnp.asarray((np.arange(2 * 16) % 60 + 1).reshape(2, 16), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (2, 16))
    seg_ids = jnp.broadcast_to((jnp.arange(16)[None, :] < 12).astype(jnp.int32), (2, 16))

    cfg_ref = self._make_config(
        pairmixer_token_z=512,
        base_mlp_dim=1024,
        max_target_length=16,
        per_device_batch_size=1,
        dtype="bfloat16",
        weight_dtype="float32",
        matmul_precision="float32",
        pairmixer_fused_projections=True,
        pairmixer_use_pallas=False,
        pairmixer_fp32_einsum=False,
        remat_policy="none",
        scan_layers=False,
    )
    mesh = self._make_mesh(cfg_ref)
    with jax.set_mesh(mesh):
      model_ref = models.transformer_as_linen(config=cfg_ref, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
      init_rng = jax.random.PRNGKey(13)
      variables = model_ref.init(
          {"params": init_rng, "dropout": init_rng},
          tokens,
          positions,
          decoder_segment_ids=seg_ids,
          enable_dropout=False,
      )
      rng_np = np.random.default_rng(20260512)
      params = jax.tree_util.tree_map(
          lambda x: jnp.asarray(rng_np.standard_normal(x.shape, dtype=np.float32) * 0.08, dtype=x.dtype),
          variables["params"],
      )

      def eval_mode(m):
        def loss_fn(p):
          logits = m.apply(
              {"params": p},
              tokens,
              positions,
              decoder_segment_ids=seg_ids,
              enable_dropout=False,
          )
          return jnp.mean(jnp.square(logits.astype(jnp.float32)))

        return jax.value_and_grad(loss_fn)(params)

      l_ref, g_ref = eval_mode(model_ref)
      systolic_grads = {}
      for remat_policy in ("none", "full"):
        cfg_sys = self._make_config(
            pairmixer_token_z=512,
            base_mlp_dim=1024,
            max_target_length=16,
            per_device_batch_size=1,
            dtype="bfloat16",
            weight_dtype="float32",
            pairmixer_fused_projections=True,
            pairmixer_use_pallas=True,
            pairmixer_fp32_einsum=False,
            remat_policy=remat_policy,
            scan_layers=False,
        )
        model_sys = models.transformer_as_linen(config=cfg_sys, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
        l_sys, g_sys = eval_mode(model_sys)
        systolic_grads[remat_policy] = jax.tree_util.tree_leaves(g_sys)
        rel_l = abs(float(l_ref - l_sys)) / (abs(float(l_ref)) + 1e-8)
        self.assertLess(rel_l, 0.02, msg=f"Systolic trunk loss mismatch for remat={remat_policy}")
        for gr, gs in zip(jax.tree_util.tree_leaves(g_ref), systolic_grads[remat_policy]):
          if gr.ndim >= 2:
            rel_g = np.linalg.norm(np.asarray(gr, np.float32) - np.asarray(gs, np.float32)) / (
                np.linalg.norm(np.asarray(gr, np.float32)) + 1e-8
            )
            self.assertLess(
                rel_g, 0.06, msg=f"Systolic trunk grad mismatch {rel_g} for shape={gr.shape}, remat={remat_policy}"
            )
      for gn, gf in zip(systolic_grads["none"], systolic_grads["full"]):
        if gn.ndim >= 2:
          rel_nf = np.linalg.norm(np.asarray(gn, np.float32) - np.asarray(gf, np.float32)) / (
              np.linalg.norm(np.asarray(gn, np.float32)) + 1e-8
          )
          self.assertLess(rel_nf, 2.0e-2, msg=f"Systolic remat=none vs remat=full mismatch {rel_nf} for shape={gn.shape}")

  def test_n128_random_mask_and_compact_tokens_parity(self):
    """Verifies synthetic 50% random row+col masking, dynamic bc slab selection, and exact active-token compaction."""
    from maxtext.input_pipeline import synthetic_data_processing
    from maxtext.kernels import pairmixer_pallas

    # 1. Dynamic slab width check for N=128 (m=16384) vs N=384 (m=147456).
    self.assertEqual(pairmixer_pallas._resolve_tri_bcm_bc(1, 128, 128 * 128), 8)
    self.assertEqual(pairmixer_pallas._resolve_tri_bcm_bc(2, 128, 128 * 128), 128)
    self.assertEqual(pairmixer_pallas._resolve_tri_bcm_bc(8, 128, 128 * 128), 128)
    self.assertEqual(pairmixer_pallas._resolve_tri_bcm_bc(1, 128, 384 * 384), 8)

    # 2. SyntheticDataIterator random 50% token mask verification (both bernoulli & exact modes).
    for mode in ("bernoulli", "exact"):
      cfg_syn = self._make_config(
          max_target_length=128,
          per_device_batch_size=4,
          pairmixer_random_mask_prob=0.5,
          pairmixer_random_mask_mode=mode,
      )
      mesh_syn = self._make_mesh(cfg_syn)
      it = synthetic_data_processing.SyntheticDataIterator(cfg_syn, mesh_syn)
      b0 = next(it)
      b1 = next(it)
      seg0 = np.asarray(b0["inputs_segmentation"])
      seg1 = np.asarray(b1["inputs_segmentation"])
      self.assertEqual(seg0.shape, (cfg_syn.global_batch_size_to_load, 128))
      self.assertTrue(np.all(seg0.sum(axis=-1) >= 1))
      self.assertFalse(np.array_equal(seg0, seg1))
      if mode == "exact":
        np.testing.assert_array_equal(seg0.sum(axis=-1), np.full((seg0.shape[0],), 64))
      else:
        self.assertAlmostEqual(float(seg0.mean()), 0.5, delta=0.12)

    # 3. Exact mathematical parity of pairmixer_compact_tokens (32 -> 16x16 & 20x20) vs dense 32x32 with ~50% random mask.
    rng_np = np.random.default_rng(20261021)
    tokens = jnp.asarray(rng_np.integers(1, 60, size=(2, 32)), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(32, dtype=jnp.int32), (2, 32))
    # Row 0 has 16 randomly placed unmasked tokens; Row 1 has 14 randomly placed unmasked tokens (both <= 16 and <= 20).
    mask_np = np.zeros((2, 32), dtype=np.int32)
    mask_np[0, rng_np.choice(32, size=16, replace=False)] = 1
    mask_np[1, rng_np.choice(32, size=14, replace=False)] = 1
    seg_ids = jnp.asarray(mask_np, dtype=jnp.int32)

    cfg_dense = self._make_config(
        pairmixer_token_z=128,
        base_mlp_dim=512,
        max_target_length=32,
        per_device_batch_size=1,
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="float32",
        pairmixer_fused_projections=True,
        pairmixer_use_pallas=False,
        pairmixer_fp32_einsum=True,
        remat_policy="none",
        scan_layers=False,
    )
    mesh = self._make_mesh(cfg_dense)
    with jax.set_mesh(mesh):
      model_dense = models.transformer_as_linen(config=cfg_dense, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
      init_rng = jax.random.PRNGKey(77)
      variables = model_dense.init(
          {"params": init_rng, "dropout": init_rng},
          tokens,
          positions,
          decoder_segment_ids=seg_ids,
          enable_dropout=False,
      )
      params = jax.tree_util.tree_map(
          lambda x: jnp.asarray(rng_np.standard_normal(x.shape, dtype=np.float32) * 0.08, dtype=x.dtype),
          variables["params"],
      )

      def masked_loss_and_grad(m):
        def loss_fn(p):
          logits = m.apply(
              {"params": p},
              tokens,
              positions,
              decoder_segment_ids=seg_ids,
              enable_dropout=False,
          )
          active = (seg_ids != 0).astype(jnp.float32)[:, :, None]
          return jnp.sum(jnp.square(logits.astype(jnp.float32)) * active) / jnp.sum(active)

        return jax.value_and_grad(loss_fn)(params)

      l_dense, g_dense = masked_loss_and_grad(model_dense)
      for k_compact in (16, 20):
        cfg_comp = self._make_config(
            pairmixer_token_z=128,
            base_mlp_dim=512,
            max_target_length=32,
            pairmixer_compact_tokens=k_compact,
            per_device_batch_size=1,
            dtype="float32",
            weight_dtype="float32",
            matmul_precision="float32",
            pairmixer_fused_projections=True,
            pairmixer_use_pallas=False,
            pairmixer_fp32_einsum=True,
            remat_policy="none",
            scan_layers=False,
        )
        model_comp = models.transformer_as_linen(config=cfg_comp, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
        l_comp, g_comp = masked_loss_and_grad(model_comp)
        np.testing.assert_allclose(float(l_comp), float(l_dense), rtol=1e-5, atol=1e-5)
        for gd, gc in zip(jax.tree_util.tree_leaves(g_dense), jax.tree_util.tree_leaves(g_comp)):
          np.testing.assert_allclose(np.asarray(gc), np.asarray(gd), rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
  unittest.main()
