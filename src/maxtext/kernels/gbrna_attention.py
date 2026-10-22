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

"""Fused RoPE + bidirectional attention Pallas kernel for GB.RNA (head_dim 64).

The kernel targets short, unmasked encoder attention (GB.RNA: S = 1024,
head_dim = 64). It does these things in one kernel:
  1. Applies rotate-half RoPE to Q and K in VMEM (no extra HBM copies).
  2. Computes a single-pass softmax over the full key row (no online rescale).
  3. Processes `heads_per_step` heads per grid step to amortize step overhead.

Layout: q, k, v and the output are `[B, S, H * D]` (the projection output with
the head axes merged). This avoids the `[B, H, S, D]` transposes.

Two heads share one 128-lane tile. The kernel never slices lanes at offset 64.
It zeroes the lanes of the other head instead. On the 256-wide v6e MXU, a
contraction or output width of 128 costs the same as 64.

The backward pass recomputes the softmax from the full row, so it needs no
saved logsumexp and no saved output: `delta = rowsum(p * dp)`.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

_LANES = 128
_HEAD_DIM = 64
_NT = (((1,), (1,)), ((), ()))  # a @ b.T
_TN = (((0,), (0,)), ((), ()))  # a.T @ b


def rope_tables(seq_len: int, head_dim: int = _HEAD_DIM, max_timescale: float = 10000.0):
  """Returns fp32 `cos`, `sin` tables of shape `[seq_len, 128]` (two heads per tile)."""
  half = head_dim // 2
  inv_freq = 1.0 / (max_timescale ** (np.arange(0, half, dtype=np.float64) * 2.0 / head_dim))
  pos = np.arange(seq_len, dtype=np.float64)[:, None]
  ang = pos * inv_freq[None, :]  # [S, half]
  ang = np.concatenate([ang, ang], axis=-1)  # [S, head_dim]
  reps = _LANES // head_dim
  cos = np.tile(np.cos(ang), (1, reps)).astype(np.float32)
  sin = np.tile(np.sin(ang), (1, reps)).astype(np.float32)
  return jnp.asarray(cos), jnp.asarray(sin)


def rotate_half_matrix(head_dim: int = _HEAD_DIM) -> np.ndarray:
  """Returns P `[128, 128]` with `x @ P == rotate_half(x)` per head_dim block."""
  half = head_dim // 2
  p = np.zeros((_LANES, _LANES), dtype=np.float32)
  for base in range(0, _LANES, head_dim):
    for j in range(half):
      p[base + j + half, base + j] = -1.0  # out[j] = -x[j + half]
      p[base + j, base + j + half] = 1.0  # out[j + half] = x[j]
  return p


def _rope_fwd(x, cos, sin, perm):
  """`x * cos + rotate_half(x) * sin` on a `[n, 128]` tile, fp32 result."""
  rot = jnp.dot(x, perm, preferred_element_type=jnp.float32)
  return x.astype(jnp.float32) * cos + rot * sin


def _rope_bwd(dy, cos, sin, perm):
  """Transpose of `_rope_fwd` with respect to x: `dy * cos + (dy * sin) @ P^T`."""
  back = lax.dot_general((dy * sin).astype(perm.dtype), perm, _NT, preferred_element_type=jnp.float32)
  return dy * cos + back


def _head_mask(h):
  lane = lax.broadcasted_iota(jnp.int32, (1, _LANES), 1)
  return (lane // _HEAD_DIM) == h


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _fwd_kernel(q_ref, k_ref, v_ref, cq_ref, sq_ref, ck_ref, sk_ref, p_ref, o_ref, *, n_tiles, scale):
  perm = p_ref[...]
  cq, sq = cq_ref[...], sq_ref[...]
  ck, sk = ck_ref[...], sk_ref[...]
  dt = q_ref.dtype
  for t in range(n_tiles):
    sl = pl.ds(t * _LANES, _LANES)
    qr = (_rope_fwd(q_ref[0, :, sl], cq, sq, perm) * scale).astype(dt)
    kr = _rope_fwd(k_ref[0, :, sl], ck, sk, perm).astype(dt)
    v = v_ref[0, :, sl]
    out = None
    for h in range(_LANES // _HEAD_DIM):
      mask = _head_mask(h)
      qh = jnp.where(mask, qr, jnp.zeros_like(qr))
      s = lax.dot_general(qh, kr, _NT, preferred_element_type=jnp.float32)  # [bq, S]
      m = jnp.max(s, axis=-1, keepdims=True)
      p = jnp.exp(s - m)
      l = jnp.sum(p, axis=-1, keepdims=True)
      o = jnp.dot(p.astype(dt), v, preferred_element_type=jnp.float32) * (1.0 / l)
      out = o if out is None else jnp.where(mask, o, out)
    o_ref[0, :, sl] = out.astype(o_ref.dtype)


def _fwd(q, k, v, *, sm_scale, heads_per_step, block_q, vmem_limit_bytes, interpret):
  b, s, hd = q.shape
  width = heads_per_step * _HEAD_DIM
  assert width % _LANES == 0 and hd % width == 0, (hd, width)
  assert s % block_q == 0, (s, block_q)
  cos, sin = rope_tables(s)
  perm = jnp.asarray(rotate_half_matrix(), dtype=q.dtype)
  grid = (b, hd // width, s // block_q)
  q_spec = pl.BlockSpec((1, block_q, width), lambda bi, gi, qi: (bi, qi, gi))
  kv_spec = pl.BlockSpec((1, s, width), lambda bi, gi, qi: (bi, 0, gi))
  tq_spec = pl.BlockSpec((block_q, _LANES), lambda bi, gi, qi: (qi, 0))
  tk_spec = pl.BlockSpec((s, _LANES), lambda bi, gi, qi: (0, 0))
  p_spec = pl.BlockSpec((_LANES, _LANES), lambda bi, gi, qi: (0, 0))
  kernel = functools.partial(_fwd_kernel, n_tiles=width // _LANES, scale=sm_scale)
  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[q_spec, kv_spec, kv_spec, tq_spec, tq_spec, tk_spec, tk_spec, p_spec],
      out_specs=q_spec,
      out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      interpret=interpret,
      name="gbrna_attention_fwd",
  )(q, k, v, cos, sin, cos, sin, perm)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _bwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    do_ref,
    cq_ref,
    sq_ref,
    ck_ref,
    sk_ref,
    p_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dk_acc,
    dv_acc,
    *,
    n_tiles,
    scale,
):
  qi = pl.program_id(2)

  @pl.when(qi == 0)
  def _init():
    dk_acc[...] = jnp.zeros_like(dk_acc)
    dv_acc[...] = jnp.zeros_like(dv_acc)

  perm = p_ref[...]
  cq, sq = cq_ref[...], sq_ref[...]
  ck, sk = ck_ref[...], sk_ref[...]
  dt = q_ref.dtype
  for t in range(n_tiles):
    sl = pl.ds(t * _LANES, _LANES)
    qr = (_rope_fwd(q_ref[0, :, sl], cq, sq, perm) * scale).astype(dt)
    kr = _rope_fwd(k_ref[0, :, sl], ck, sk, perm).astype(dt)
    v = v_ref[0, :, sl]
    do = do_ref[0, :, sl]
    dqr = dkr = dv = None
    for h in range(_LANES // _HEAD_DIM):
      mask = _head_mask(h)
      qh = jnp.where(mask, qr, jnp.zeros_like(qr))
      doh = jnp.where(mask, do, jnp.zeros_like(do))
      s = lax.dot_general(qh, kr, _NT, preferred_element_type=jnp.float32)  # [bq, S]
      m = jnp.max(s, axis=-1, keepdims=True)
      p = jnp.exp(s - m)
      p = p * (1.0 / jnp.sum(p, axis=-1, keepdims=True))
      pb = p.astype(dt)
      dv_h = lax.dot_general(pb, do, _TN, preferred_element_type=jnp.float32)  # [S, 128]
      dp = lax.dot_general(doh, v, _NT, preferred_element_type=jnp.float32)  # [bq, S]
      delta = jnp.sum(p * dp, axis=-1, keepdims=True)
      ds = (p * (dp - delta)).astype(dt)
      dq_h = jnp.dot(ds, kr, preferred_element_type=jnp.float32)  # [bq, 128], d(q_rope * scale)
      dk_h = lax.dot_general(ds, qr, _TN, preferred_element_type=jnp.float32)  # [S, 128]
      if dqr is None:
        dqr, dkr, dv = dq_h, dk_h, dv_h
      else:
        dqr = jnp.where(mask, dq_h, dqr)
        dkr = jnp.where(mask, dk_h, dkr)
        dv = jnp.where(mask, dv_h, dv)
    dq_ref[0, :, sl] = _rope_bwd(dqr * scale, cq, sq, perm).astype(dq_ref.dtype)
    dk_acc[:, sl] += _rope_bwd(dkr, ck, sk, perm)
    dv_acc[:, sl] += dv

  @pl.when(qi == pl.num_programs(2) - 1)
  def _store():
    dk_ref[0] = dk_acc[...].astype(dk_ref.dtype)
    dv_ref[0] = dv_acc[...].astype(dv_ref.dtype)


def _bwd(q, k, v, do, *, sm_scale, heads_per_step, block_q, vmem_limit_bytes, interpret):
  b, s, hd = q.shape
  width = heads_per_step * _HEAD_DIM
  assert width % _LANES == 0 and hd % width == 0, (hd, width)
  assert s % block_q == 0, (s, block_q)
  cos, sin = rope_tables(s)
  perm = jnp.asarray(rotate_half_matrix(), dtype=q.dtype)
  grid = (b, hd // width, s // block_q)
  q_spec = pl.BlockSpec((1, block_q, width), lambda bi, gi, qi: (bi, qi, gi))
  kv_spec = pl.BlockSpec((1, s, width), lambda bi, gi, qi: (bi, 0, gi))
  tq_spec = pl.BlockSpec((block_q, _LANES), lambda bi, gi, qi: (qi, 0))
  tk_spec = pl.BlockSpec((s, _LANES), lambda bi, gi, qi: (0, 0))
  p_spec = pl.BlockSpec((_LANES, _LANES), lambda bi, gi, qi: (0, 0))
  kernel = functools.partial(_bwd_kernel, n_tiles=width // _LANES, scale=sm_scale)
  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[q_spec, kv_spec, kv_spec, q_spec, tq_spec, tq_spec, tk_spec, tk_spec, p_spec],
      out_specs=[q_spec, kv_spec, kv_spec],
      out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, k, v)],
      scratch_shapes=[pltpu.VMEM((s, width), jnp.float32), pltpu.VMEM((s, width), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      interpret=interpret,
      name="gbrna_attention_bwd",
  )(q, k, v, do, cos, sin, cos, sin, perm)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def gbrna_attention(
    q,
    k,
    v,
    sm_scale: float = _HEAD_DIM**-0.5,
    heads_per_step: int = 4,
    block_q: int = 1024,
    block_q_bwd: int = 1024,
    vmem_limit_bytes: int = 96 * 1024 * 1024,
    interpret: bool = False,
):
  """RoPE + bidirectional softmax attention. q, k, v, out: `[B, S, H * 64]`.

  RoPE uses positions `0..S-1`, base 10000, rotate-half over the full head_dim.
  The logits are `sm_scale * rope(q) . rope(k)`. There is no mask.
  MaxText folds `1 / sqrt(head_dim)` into the query projection, so it passes `sm_scale=1.0`.
  """
  return _fwd(
      q,
      k,
      v,
      sm_scale=sm_scale,
      heads_per_step=heads_per_step,
      block_q=block_q,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )


def _vjp_fwd(q, k, v, sm_scale, heads_per_step, block_q, block_q_bwd, vmem_limit_bytes, interpret):
  out = _fwd(
      q,
      k,
      v,
      sm_scale=sm_scale,
      heads_per_step=heads_per_step,
      block_q=block_q,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  del block_q_bwd
  return out, (q, k, v)


def _vjp_bwd(sm_scale, heads_per_step, block_q, block_q_bwd, vmem_limit_bytes, interpret, res, do):
  del block_q
  q, k, v = res
  dq, dk, dv = _bwd(
      q,
      k,
      v,
      do.astype(q.dtype),
      sm_scale=sm_scale,
      heads_per_step=heads_per_step,
      block_q=block_q_bwd,
      vmem_limit_bytes=vmem_limit_bytes,
      interpret=interpret,
  )
  return dq, dk, dv


gbrna_attention.defvjp(_vjp_fwd, _vjp_bwd)


def reference_attention(q, k, v, sm_scale: float = _HEAD_DIM**-0.5, head_dim: int = _HEAD_DIM):
  """Plain JAX reference: rotate-half RoPE + full softmax. `[B, S, H * D]` in and out."""
  b, s, hd = q.shape
  h = hd // head_dim
  cos, sin = rope_tables(s, head_dim)
  cos, sin = cos[:, :head_dim], sin[:, :head_dim]

  def rope(x):
    x = x.reshape(b, s, h, head_dim).astype(jnp.float32)
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    rot = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos[None, :, None, :] + rot * sin[None, :, None, :]

  qr, kr = rope(q), rope(k)
  vv = v.reshape(b, s, h, head_dim).astype(jnp.float32)
  logits = jnp.einsum("bqhd,bkhd->bhqk", qr, kr, precision=lax.Precision.HIGHEST) * sm_scale
  p = jax.nn.softmax(logits, axis=-1)
  o = jnp.einsum("bhqk,bkhd->bqhd", p, vv, precision=lax.Precision.HIGHEST)
  return o.reshape(b, s, hd).astype(q.dtype)
