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

"""Pairmixer model definitions (genesisml/pairmixer)."""

from typing import Any

from flax import nnx
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Array, Config, DType, MODEL_MODE_TRAIN, PRNGKey, Shape, ShardMode
from maxtext.kernels import pairmixer_pallas
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers.initializers import Initializer, InitializerAxis, NdInitializer, nd_dense_init
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import maybe_shard_with_logical


def nd_zero_init() -> NdInitializer:
  """Zero initializer matching Pairmixer reference final_init_ and gating_init_."""

  def init_fn(
      key: PRNGKey,
      shape: Shape,
      dtype: DType,
      in_axis: InitializerAxis,
      out_axis: InitializerAxis,
  ) -> Array:
    del key, in_axis, out_axis
    return jnp.zeros(shape, dtype=dtype)

  return init_fn


def nd_pgg_init() -> NdInitializer:
  """Initializes fused (c_z, 5 * c_z) pgg_in: LeCun normal on p_in (0:2*c_z) and zeros on g_in/g_out (2*c_z:5*c_z)."""
  lecun_fn = nd_dense_init(1.0, "fan_in", "truncated_normal")

  def init_fn(
      key: PRNGKey,
      shape: Shape,
      dtype: DType,
      in_axis: InitializerAxis,
      out_axis: InitializerAxis,
  ) -> Array:
    c_z = shape[1] // 5
    p_shape = (shape[0], 2 * c_z)
    g_shape = (shape[0], 3 * c_z)
    p_val = lecun_fn(key, p_shape, dtype, in_axis, out_axis)
    g_val = jnp.zeros(g_shape, dtype=dtype)
    return jnp.concatenate([p_val, g_val], axis=1)

  return init_fn


class PairmixerLayerNorm(nnx.Module):
  """Layer normalization on the last axis using elementwise affine scale and bias.

  Avoids Gpt3LayerNorm's `jnp.einsum("i...k,...k->i...k")`, which lowers on 4D
  `[B, N, N, D]` tensors to a batched `dot_general` with `batch_dims=(D,)` and a
  4D transpose `{1,2,3,0}` into `{2,1,0,3}` physical layout.
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = jax.nn.initializers.zeros,
      use_bias: bool = True,
      reductions_in_fp32: bool = True,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.use_bias = use_bias
    self.reductions_in_fp32 = reductions_in_fp32
    self.parameter_memory_host_offload = parameter_memory_host_offload

    self.scale = nnx.Param(
        self.scale_init(rngs.params(), (num_features,), self.weight_dtype),
        sharding=self.kernel_axes,
    )
    if self.use_bias:
      self.bias = nnx.Param(
          initializers.default_bias_init(rngs.params(), (num_features,), self.weight_dtype),
          sharding=self.kernel_axes,
      )
    else:
      self.bias = None

  def __call__(self, x: jnp.ndarray, out_sharding: Any = None) -> jnp.ndarray:
    """Applies layer normalization on the last axis of `x`."""
    del out_sharding
    if self.reductions_in_fp32:
      x = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normed_inputs = (x - mean) * lax.rsqrt(var + self.epsilon)
    if self.reductions_in_fp32:
      normed_inputs = normed_inputs.astype(self.dtype)

    scale = self.scale[...]
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    scale = jnp.asarray(scale, self.dtype) + jnp.asarray(1.0, self.dtype)
    output = normed_inputs * scale
    if self.bias is not None:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output = output + bias
    return output


class TriangleMultiplication(nnx.Module):
  """Pairmixer TriangleMultiplication layer (Outgoing or Incoming).

  Reference:
    src/pairmixer/model/layers/triangular_mult.py in genesistherapeutics/pairmixer.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      direction: str = "outgoing",
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    if direction not in ("outgoing", "incoming"):
      raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction!r}")
    self.config = config
    self.mesh = mesh
    self.direction = direction
    self.quant = quant
    c_z = config.pairmixer_token_z

    self.norm_in = PairmixerLayerNorm(
        num_features=c_z,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        use_bias=True,
        reductions_in_fp32=True,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=rngs,
    )
    self.norm_out = PairmixerLayerNorm(
        num_features=c_z,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        use_bias=True,
        reductions_in_fp32=True,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=rngs,
    )

    lecun_init = nd_dense_init(1.0, "fan_in", "truncated_normal")
    zero_init = nd_zero_init()

    if config.pairmixer_fused_projections:
      self.pgg_in = DenseGeneral(
          in_features_shape=c_z,
          out_features_shape=5 * c_z,
          weight_dtype=config.weight_dtype,
          dtype=config.dtype,
          kernel_init=nd_pgg_init(),
          kernel_axes=("embed", "mlp"),
          quant=quant,
          use_bias=False,
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          mesh=mesh,
          rngs=rngs,
      )
    else:
      self.p_in = DenseGeneral(
          in_features_shape=c_z,
          out_features_shape=2 * c_z,
          weight_dtype=config.weight_dtype,
          dtype=config.dtype,
          kernel_init=lecun_init,
          kernel_axes=("embed", "mlp"),
          quant=quant,
          use_bias=False,
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          mesh=mesh,
          rngs=rngs,
      )
      self.g_in = DenseGeneral(
          in_features_shape=c_z,
          out_features_shape=2 * c_z,
          weight_dtype=config.weight_dtype,
          dtype=config.dtype,
          kernel_init=zero_init,
          kernel_axes=("embed", "mlp"),
          quant=quant,
          use_bias=False,
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          mesh=mesh,
          rngs=rngs,
      )
      self.g_out = DenseGeneral(
          in_features_shape=c_z,
          out_features_shape=c_z,
          weight_dtype=config.weight_dtype,
          dtype=config.dtype,
          kernel_init=zero_init,
          kernel_axes=("embed", "mlp"),
          quant=quant,
          use_bias=False,
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          mesh=mesh,
          rngs=rngs,
      )

    self.p_out = DenseGeneral(
        in_features_shape=c_z,
        out_features_shape=c_z,
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        kernel_init=zero_init,
        kernel_axes=("mlp", "embed"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(self, z: Array, pair_mask: Array | None = None) -> Array:
    cfg = self.config

    x_in = self.norm_in(z)
    x_in = maybe_shard_with_logical(
        x_in,
        ("activation_batch", "activation_length", None, "activation_embed"),
        self.mesh,
        cfg.shard_mode,
    )

    if cfg.pairmixer_fused_projections:
      c_z = cfg.pairmixer_token_z
      pgg = self.pgg_in(x_in)
      pgg = maybe_shard_with_logical(
          pgg,
          ("activation_batch", "activation_length", None, "activation_mlp"),
          self.mesh,
          cfg.shard_mode,
      )
      p, g, g_out_val = jnp.split(pgg, [2 * c_z, 4 * c_z], axis=-1)
    else:
      p = self.p_in(x_in)
      g = self.g_in(x_in)
      g_out_val = self.g_out(x_in)

    ab = p * jax.nn.sigmoid(g)
    if pair_mask is not None:
      ab = ab * pair_mask[..., None].astype(ab.dtype)
    ab = maybe_shard_with_logical(
        ab,
        ("activation_batch", "activation_length", None, "activation_mlp"),
        self.mesh,
        cfg.shard_mode,
    )
    a, b = jnp.split(ab, 2, axis=-1)

    einsum_subscripts = "bikd,bjkd->bijd" if self.direction == "outgoing" else "bkid,bkjd->bijd"
    if cfg.pairmixer_fp32_einsum:
      t = jnp.einsum(
          einsum_subscripts,
          a.astype(jnp.float32),
          b.astype(jnp.float32),
          precision=lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
    else:
      t = jnp.einsum(
          einsum_subscripts,
          a.astype(cfg.dtype),
          b.astype(cfg.dtype),
          precision=lax.Precision(cfg.matmul_precision),
      )

    t = maybe_shard_with_logical(
        t,
        ("activation_batch", "activation_length", None, "activation_embed"),
        self.mesh,
        cfg.shard_mode,
    )

    t_norm = self.norm_out(t)
    y = self.p_out(t_norm) * jax.nn.sigmoid(g_out_val)
    y = maybe_shard_with_logical(
        y,
        ("activation_batch", "activation_length", None, "activation_embed"),
        self.mesh,
        cfg.shard_mode,
    )
    return y


class PairmixerTransition(nnx.Module):
  """Pairmixer SwiGLU Transition layer operating on Z in R^{B x N x N x c_z}.

  Reference:
    src/pairmixer/model/layers/transition.py in genesistherapeutics/pairmixer.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    c_z = config.pairmixer_token_z
    mlp_dim = config.mlp_dim

    self.norm = PairmixerLayerNorm(
        num_features=c_z,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        use_bias=True,
        reductions_in_fp32=True,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=rngs,
    )

    lecun_init = nd_dense_init(1.0, "fan_in", "truncated_normal")
    zero_init = nd_zero_init()

    if config.pairmixer_fused_projections:
      self.fc12 = DenseGeneral(
          in_features_shape=c_z,
          out_features_shape=2 * mlp_dim,
          weight_dtype=config.weight_dtype,
          dtype=config.dtype,
          kernel_init=lecun_init,
          kernel_axes=("embed", "mlp"),
          quant=quant,
          use_bias=False,
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          mesh=mesh,
          rngs=rngs,
      )
    else:
      self.fc1 = DenseGeneral(
          in_features_shape=c_z,
          out_features_shape=mlp_dim,
          weight_dtype=config.weight_dtype,
          dtype=config.dtype,
          kernel_init=lecun_init,
          kernel_axes=("embed", "mlp"),
          quant=quant,
          use_bias=False,
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          mesh=mesh,
          rngs=rngs,
      )
      self.fc2 = DenseGeneral(
          in_features_shape=c_z,
          out_features_shape=mlp_dim,
          weight_dtype=config.weight_dtype,
          dtype=config.dtype,
          kernel_init=lecun_init,
          kernel_axes=("embed", "mlp"),
          quant=quant,
          use_bias=False,
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          mesh=mesh,
          rngs=rngs,
      )

    self.fc3 = DenseGeneral(
        in_features_shape=mlp_dim,
        out_features_shape=c_z,
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        kernel_init=zero_init,
        kernel_axes=("mlp", "embed"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(self, z: Array) -> Array:
    cfg = self.config
    x = self.norm(z)
    x = maybe_shard_with_logical(
        x,
        ("activation_batch", "activation_length", None, "activation_embed"),
        self.mesh,
        cfg.shard_mode,
    )
    if cfg.pairmixer_fused_projections:
      h12 = self.fc12(x)
      h12 = maybe_shard_with_logical(
          h12,
          ("activation_batch", "activation_length", None, "activation_mlp"),
          self.mesh,
          cfg.shard_mode,
      )
      h12 = checkpoint_name(h12, "mlpwi")
      h1, h2 = jnp.split(h12, 2, axis=-1)
      h1 = checkpoint_name(h1, "mlpwi_0")
      h2 = checkpoint_name(h2, "mlpwi_1")
    else:
      h1 = checkpoint_name(self.fc1(x), "mlpwi_0")
      h2 = checkpoint_name(self.fc2(x), "mlpwi_1")
      h1 = maybe_shard_with_logical(
          h1,
          ("activation_batch", "activation_length", None, "activation_mlp"),
          self.mesh,
          cfg.shard_mode,
      )
      h2 = maybe_shard_with_logical(
          h2,
          ("activation_batch", "activation_length", None, "activation_mlp"),
          self.mesh,
          cfg.shard_mode,
      )

    h = jax.nn.silu(h1) * h2
    y = checkpoint_name(self.fc3(h), "mlpwo")
    y = maybe_shard_with_logical(
        y,
        ("activation_batch", "activation_length", None, "activation_embed"),
        self.mesh,
        cfg.shard_mode,
    )
    return y


class PairmixerDecoderLayer(nnx.Module):
  """Single Pairmixer block operating on pairwise tensor Z in R^{B x N x N x c_z}.

  Reference:
    src/pairmixer/model/modules/pairmixer.py in genesistherapeutics/pairmixer.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant

    self.tri_mul_out = TriangleMultiplication(
        config=config,
        mesh=mesh,
        direction="outgoing",
        quant=quant,
        rngs=rngs,
    )
    self.tri_mul_in = TriangleMultiplication(
        config=config,
        mesh=mesh,
        direction="incoming",
        quant=quant,
        rngs=rngs,
    )
    self.transition_z = PairmixerTransition(
        config=config,
        mesh=mesh,
        quant=quant,
        rngs=rngs,
    )
    self.dropout = linears.Dropout(
        rate=config.dropout_rate,
        broadcast_dims=(-2, -1),
        rngs=rngs,
    )

  def _extract_pallas_layer_params(self) -> tuple[Array, ...]:
    params = (
        self.tri_mul_out.norm_in.scale[...],
        self.tri_mul_out.norm_in.bias[...],
        self.tri_mul_out.pgg_in.kernel[...],
        self.tri_mul_out.norm_out.scale[...],
        self.tri_mul_out.norm_out.bias[...],
        self.tri_mul_out.p_out.kernel[...],
        self.tri_mul_in.norm_in.scale[...],
        self.tri_mul_in.norm_in.bias[...],
        self.tri_mul_in.pgg_in.kernel[...],
        self.tri_mul_in.norm_out.scale[...],
        self.tri_mul_in.norm_out.bias[...],
        self.tri_mul_in.p_out.kernel[...],
        self.transition_z.norm.scale[...],
        self.transition_z.norm.bias[...],
        self.transition_z.fc12.kernel[...],
        self.transition_z.fc3.kernel[...],
    )
    if self.config.parameter_memory_host_offload:
      return tuple(jax.device_put(p, max_utils.device_space()) for p in params)
    return params

  def _can_use_pallas(self, z: Array, deterministic: bool, *, channel_major: bool = False) -> bool:
    cfg = self.config
    if not getattr(cfg, "pairmixer_use_pallas", False):
      return False
    if not cfg.pairmixer_fused_projections or self.quant is not None:
      return False
    if channel_major:
      return pairmixer_pallas.can_use_channel_major_megakernels(
          cfg.pairmixer_token_z, cfg.mlp_dim, z.shape[2], mesh=self.mesh
      )
    max_logging.log(
        "Pairmixer Gen 1 non-channel-major Pallas kernels are removed. Falling back to XLA path. Set pairmixer_use_pallas=false or use channel_major layouts."
    )
    return False

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: Array | None,
      decoder_positions: Array | None,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      slot: int | None = None,
      kv_cache: Any = None,
      attention_metadata: Any = None,
      channel_major: bool = False,
      mask_8m: Array | None = None,
      **kwargs,
  ) -> tuple[Array, None]:
    del decoder_positions, model_mode, previous_chunk, slot, kv_cache, attention_metadata, kwargs

    if channel_major:
      z = maybe_shard_with_logical(
          inputs,
          ("activation_batch", "activation_embed", "activation_length", None),
          self.mesh,
          self.config.shard_mode,
      )
      z = checkpoint_name(z, "decoder_layer_input")
      if mask_8m is None:
        mask_8m = pairmixer_pallas.build_channel_major_mask_8m(decoder_segment_ids, z.shape[0], z.shape[2])
      cfg = self.config
      z_out = pairmixer_pallas.pallas_pairmixer_layer_bcnn(
          z,
          mask_8m,
          self._extract_pallas_layer_params(),
          mesh=self.mesh,
          fp32_einsum=cfg.pairmixer_fp32_einsum,
          matmul_precision=cfg.matmul_precision,
          eps=cfg.normalization_layer_epsilon,
      )
      return z_out, None

    z = maybe_shard_with_logical(
        inputs,
        ("activation_batch", "activation_length", None, "activation_embed"),
        self.mesh,
        self.config.shard_mode,
    )
    z = checkpoint_name(z, "decoder_layer_input")

    if decoder_segment_ids is not None:
      token_mask = decoder_segment_ids != 0
      pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    else:
      pair_mask = None

    self._can_use_pallas(z, deterministic)

    tri_out = self.tri_mul_out(z, pair_mask=pair_mask)
    tri_out = self.dropout(tri_out, deterministic=deterministic)
    z = z + tri_out

    tri_in = self.tri_mul_in(z, pair_mask=pair_mask)
    tri_in = self.dropout(tri_in, deterministic=deterministic)
    z = z + tri_in

    z = z + self.transition_z(z)
    z = maybe_shard_with_logical(
        z,
        ("activation_batch", "activation_length", None, "activation_embed"),
        self.mesh,
        self.config.shard_mode,
    )
    return z, None


class PairmixerInputProjector(nnx.Module):
  """Lifts sequence embeddings S [B, N, c_s] to initial pairwise representation Z [B, N, N, c_z]."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    c_s = config.emb_dim
    c_z = config.pairmixer_token_z
    lecun_init = nd_dense_init(1.0, "fan_in", "truncated_normal")

    self.s_to_z_1 = DenseGeneral(
        in_features_shape=c_s,
        out_features_shape=c_z,
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        kernel_init=lecun_init,
        kernel_axes=("embed", "mlp"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        mesh=mesh,
        rngs=rngs,
    )
    self.s_to_z_2 = DenseGeneral(
        in_features_shape=c_s,
        out_features_shape=c_z,
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        kernel_init=lecun_init,
        kernel_axes=("embed", "mlp"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(self, s: Array, *, channel_major: bool = False) -> Array:
    if channel_major:
      z1 = jnp.swapaxes(self.s_to_z_1(s), 1, 2)
      z2 = jnp.swapaxes(self.s_to_z_2(s), 1, 2)
      z = z1[:, :, :, None] + z2[:, :, None, :]
      return maybe_shard_with_logical(
          z,
          ("activation_batch", "activation_embed", "activation_length", None),
          self.mesh,
          self.config.shard_mode,
      )
    z1 = self.s_to_z_1(s)
    z2 = self.s_to_z_2(s)
    z = z1[:, :, None, :] + z2[:, None, :, :]
    return maybe_shard_with_logical(
        z,
        ("activation_batch", "activation_length", None, "activation_embed"),
        self.mesh,
        self.config.shard_mode,
    )


class PairmixerOutputProjector(nnx.Module):
  """Projects pairwise representation Z [B, N, N, c_z] to sequence hidden state [B, N, c_s]."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    c_s = config.emb_dim
    c_z = config.pairmixer_token_z
    lecun_init = nd_dense_init(1.0, "fan_in", "truncated_normal")

    self.z_to_s = DenseGeneral(
        in_features_shape=c_z,
        out_features_shape=c_s,
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        kernel_init=lecun_init,
        kernel_axes=("mlp", "embed"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(
      self,
      z: Array,
      decoder_segment_ids: Array | None = None,
      *,
      channel_major: bool = False,
  ) -> Array:
    if channel_major:
      if decoder_segment_ids is not None:
        tok = (decoder_segment_ids != 0).astype(jnp.float32)
        denom = jnp.maximum(tok[:, None, :] * jnp.sum(tok, axis=1, keepdims=True)[:, None, :], 1.0)
        z_sum = jnp.einsum("bcij,bj->bci", z, tok.astype(z.dtype), preferred_element_type=jnp.float32)
        z_bcn = (tok[:, None, :] * z_sum) / denom
        z_pooled = jnp.swapaxes(z_bcn.astype(z.dtype), 1, 2)
      else:
        z_pooled = jnp.swapaxes(jnp.mean(z, axis=3), 1, 2)
    elif decoder_segment_ids is not None:
      token_mask = (decoder_segment_ids != 0).astype(z.dtype)
      pair_mask = token_mask[:, :, None] * token_mask[:, None, :]
      denom = jnp.maximum(jnp.sum(pair_mask, axis=2, keepdims=True), 1.0)
      z_pooled = jnp.sum(z * pair_mask[..., None], axis=2) / denom
    else:
      z_pooled = jnp.mean(z, axis=2)
    s_out = self.z_to_s(z_pooled)
    return maybe_shard_with_logical(
        s_out,
        ("activation_batch", "activation_length", "activation_embed"),
        self.mesh,
        self.config.shard_mode,
    )


def apply_synthetic_random_mask(config, segmentation, step=0):
  mask_prob = float(getattr(config, "pairmixer_random_mask_prob", 0.0))
  if mask_prob > 0.0:
    batch_sz, seq_len = segmentation.shape
    rng = jax.random.fold_in(jax.random.PRNGKey(1337), step)
    mask_mode = getattr(config, "pairmixer_random_mask_mode", "bernoulli")
    if mask_mode == "exact":
      n_keep = max(1, min(seq_len, int(round((1.0 - mask_prob) * seq_len))))
      scores = jax.random.uniform(rng, shape=(batch_sz, seq_len))
      ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
      rand_mask = (ranks < n_keep).astype(jnp.int32)
    else:
      rand_mask = jax.random.bernoulli(rng, p=1.0 - mask_prob, shape=(batch_sz, seq_len)).astype(jnp.int32)
      has_any = jnp.max(rand_mask, axis=-1, keepdims=True)
      rand_mask = jnp.where((jnp.arange(seq_len)[None, :] == 0) & (has_any == 0), 1, rand_mask)
    segmentation = segmentation * rand_mask
  return segmentation


def prepare_pairmixer_input(decoder, cfg, y, decoder_segment_ids, decoder_positions, deterministic):
  pairmixer_cm = False
  pairmixer_compact_idx = None
  pairmixer_orig_seq_len = y.shape[1]
  pairmixer_orig_segment_ids = decoder_segment_ids

  k_compact = int(getattr(cfg, "pairmixer_compact_tokens", 0))
  if k_compact > 0 and k_compact < y.shape[1] and decoder_segment_ids is not None:
    pairmixer_compact_idx = jax.vmap(lambda seg: jnp.argsort(-(seg != 0).astype(jnp.int32))[:k_compact])(
        decoder_segment_ids
    )
    y = jax.vmap(lambda s_row, idx_row: s_row[idx_row])(y, pairmixer_compact_idx)
    decoder_segment_ids = jax.vmap(lambda seg_row, idx_row: seg_row[idx_row])(decoder_segment_ids, pairmixer_compact_idx)
    if decoder_positions is not None:
      decoder_positions = jax.vmap(lambda pos_row, idx_row: pos_row[idx_row])(decoder_positions, pairmixer_compact_idx)
  pairmixer_cm = bool(
      getattr(cfg, "pairmixer_use_pallas", False)
      and getattr(cfg, "pairmixer_fused_projections", False)
      and getattr(decoder, "quant", None) is None
      and (deterministic or getattr(cfg, "dropout_rate", 0.0) == 0.0)
      and not getattr(cfg, "using_pipeline_parallelism", False)
      and not cfg.parameter_memory_host_offload
      and pairmixer_pallas.can_use_channel_major_megakernels(
          cfg.pairmixer_token_z,
          cfg.mlp_dim,
          y.shape[1],
          mesh=decoder.mesh,
      )
  )
  y = decoder.pairmixer_input_projector(y, channel_major=pairmixer_cm)

  pairmixer_ctx = (pairmixer_compact_idx, pairmixer_orig_seq_len, pairmixer_orig_segment_ids)
  return y, decoder_segment_ids, decoder_positions, pairmixer_cm, pairmixer_ctx


def run_pairmixer_trunk(decoder, cfg, y, decoder_segment_ids, layer_args):
  if cfg.pairmixer_token_z >= 512:
    pairmixer_mask_8m = pairmixer_pallas.build_channel_major_mask_8m(decoder_segment_ids, y.shape[0], y.shape[2])
    layers_list = [
        getattr(decoder, f"layers_{lyr}", None) or decoder.layers[lyr] for lyr in range(cfg.num_decoder_layers)
    ]
    all_layer_params = tuple(lyr_mod._extract_pallas_layer_params() for lyr_mod in layers_list)
    y = pairmixer_pallas.pallas_pairmixer_systolic_trunk_bcnn(
        y,
        pairmixer_mask_8m,
        all_layer_params,
        mesh=decoder.mesh,
        fp32_einsum=cfg.pairmixer_fp32_einsum,
        matmul_precision=cfg.matmul_precision,
        eps=cfg.normalization_layer_epsilon,
        remat_all=(cfg.remat_policy == "full"),
    )
    return y
  elif cfg.remat_policy in (None, "none"):
    batch_spec = pairmixer_pallas._get_batch_partition_spec(decoder.mesh, y.shape[0])
    if batch_spec is None:
      batch_shards = 1
    elif isinstance(batch_spec, tuple):
      batch_shards = 1
      for ax in batch_spec:
        batch_shards *= decoder.mesh.shape[ax]
    else:
      batch_shards = decoder.mesh.shape[batch_spec]
    b_local = y.shape[0] // max(1, batch_shards)
    seq_len_cm = y.shape[2]
    trunk_hbm_est = 10 * cfg.num_decoder_layers * b_local * cfg.pairmixer_token_z * (seq_len_cm**2) * 2
    pairmixer_mask_8m = pairmixer_pallas.build_channel_major_mask_8m(decoder_segment_ids, y.shape[0], seq_len_cm)
    layers_list = [
        getattr(decoder, f"layers_{lyr}", None) or decoder.layers[lyr] for lyr in range(cfg.num_decoder_layers)
    ]
    if trunk_hbm_est <= 27 * (1024**3):
      all_layer_params = tuple(lyr_mod._extract_pallas_layer_params() for lyr_mod in layers_list)
      y = pairmixer_pallas.pallas_pairmixer_cross_layer_trunk_bcnn(
          y,
          pairmixer_mask_8m,
          all_layer_params,
          mesh=decoder.mesh,
          fp32_einsum=cfg.pairmixer_fp32_einsum,
          matmul_precision=cfg.matmul_precision,
          eps=cfg.normalization_layer_epsilon,
      )
    else:
      for lyr_mod in layers_list:
        y, _ = lyr_mod(
            y,
            *layer_args,
            channel_major=True,
            mask_8m=pairmixer_mask_8m,
        )
    return y
  return y


def run_pairmixer_output(decoder, y, decoder_segment_ids, pairmixer_cm, pairmixer_ctx):
  pairmixer_compact_idx, pairmixer_orig_seq_len, pairmixer_orig_segment_ids = pairmixer_ctx
  y = decoder.pairmixer_output_projector(y, decoder_segment_ids, channel_major=pairmixer_cm)
  if pairmixer_compact_idx is not None:
    y = jax.vmap(
        lambda y_sub_row, idx_row: jnp.zeros((pairmixer_orig_seq_len, y_sub_row.shape[-1]), dtype=y_sub_row.dtype)
        .at[idx_row]
        .set(y_sub_row)
    )(y, pairmixer_compact_idx)
  decoder_segment_ids = pairmixer_orig_segment_ids
  return y, decoder_segment_ids
