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

"""Fused Pallas TPU kernels for Pairmixer TriangleMultiplication and Transition.

Why these kernels beat standard XLA lowering on TPU v6e:
1. Fused SwiGLU Transition (`pallas_pairmixer_transition`):
   - In standard XLA, `fc12` (`[B*N^2, 128] @ [128, 1024] -> [B*N^2, 1024]`, 288 MiB per layer at
     N=384) exceeds TPU v6e's 128 MiB VMEM capacity and cannot be fused across two consecutive
     `dot_general` ops (`fc12` and `fc3`). XLA spills `x_norm` (36 MiB), `h12` (288 MiB), and `h`
     (144 MiB) to HBM in forward, recomputation, and backward (> 1.5 GiB HBM traffic per layer).
   - This kernel tiles along the flattened pair-token dimension `M = B * N * N` in blocks of
     `block_m` (default 2048) while holding `w12` (256 KiB), `w3` (128 KiB), `scale`, and `bias`
     in VMEM. Forward computes `LayerNorm -> fc12 -> SwiGLU -> fc3 -> residual_add` in a single
     HBM streaming pass (36 MiB read + 36 MiB write). Backward recomputes the tile-local hidden
     states in VMEM on the fly, accumulates `dw12, dw3, dscale, dbias` in fp32 in VMEM across all
     `M // block_m` tiles, and writes `dz_total = dy + dz` in one pass.
   - Under `remat_policy="full"`, because `pallas_pairmixer_transition` is the final sub-block of
     `PairmixerDecoderLayer` and saves only its input `z` (not its output) in the custom VJP
     residual, JAX dead-code-eliminates the entire Transition forward pass from backward remat.

2. Fused Channel-Batched TriangleMultiplication Prologue & Epilogue (`pallas_triangle_multiplication`):
   - Prologue fuses `norm_in(z) -> pgg = x_norm @ w_pgg ([bm, 640]) -> split(pa, pb, ga, gb, go) ->
     a = pa * sigmoid(ga), b = pb * sigmoid(gb), go_sig = sigmoid(go)` (plus optional `pair_mask`)
     in VMEM, never materializing `x_norm` (36 MiB) or `pgg` (180 MiB) to HBM, and transposes
     `(ti, 128, 128) -> (128, ti, 128)` directly in VMEM to emit `a, b` in contiguous channel-batched
     matrix layout `[B, C=128, N, N]`.
   - Emitting `a, b` and consuming `t` in `[B, C=128, N, N]` turns both outgoing (`bdik,bdjk->bdij`)
     and incoming (`bdki,bdkj->bdij`) contractions (and their backward passes) into zero-copy
     batched matrix multiplications across all per-device batch sizes `B`, eliminating all 4D HBM
     layout transposes (`copy`).
   - Epilogue transposes `t` from `[B, C=128, N, N]` back to `(ti*128, 128)` in VMEM and fuses
     `norm_out(t) -> p_out = t_norm @ w_out -> z_out = z_res + p_out * go_sig`. For `B >= 2`, only
     `(z, t)` are saved across layers while cheap `(a, b, go_sig)` projections are recomputed per
     layer from `z`, enabling zero-contraction-remat scaling up to `B = 3` (`~87%` HBM on TPU v6e).
"""

import functools
import os
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

_LANES = 128
_BATCH_MESH_AXES = ("diloco", "data", "stage", "fsdp", "fsdp_transpose", "expert")


def _resolve_interpret(interpret: bool | None, mesh: Mesh | Any = None) -> bool:
  if interpret is not None:
    return interpret
  if mesh is not None and hasattr(mesh, "devices") and mesh.devices.size > 0:
    first_dev = mesh.devices.flat[0]
    if getattr(first_dev, "platform", None) == "tpu":
      return False
  return jax.default_backend() != "tpu"


def _get_batch_partition_spec(mesh: Mesh | Any, batch_size: int) -> Any:
  """Returns the mesh axis (or tuple of axes) sharding the batch dimension."""
  if mesh is None:
    return None
  active = []
  prod = 1
  for ax in _BATCH_MESH_AXES:
    if ax in mesh.axis_names and mesh.shape[ax] > 1:
      active.append(ax)
      prod *= mesh.shape[ax]
  if not active or batch_size % prod != 0:
    return None
  return active[0] if len(active) == 1 else tuple(active)


# ==============================================================================
# 3. Native Channel-Major [B, C, N, N] Cross-Sublayer & Cross-Layer Mega-Kernels
# ==============================================================================
#
# Physical layout identity on TPU v6e (Trillium):
#   In row-major layout `{3, 2, 1, 0:T(8,128)(2,1)}`, `bf16[B, C, N, N]` has
#   contiguous `(N * N)` spatial slices of size `M = N * N` along axes `(2, 3)`.
#   Reshaping `[B, C, N, N] <-> [B, C, M]` is a zero-copy metadata bitcast that:
#   1. Keeps both outgoing (`bdik,bdjk->bdij`) and incoming (`bdki,bdkj->bdij`)
#      triangle contractions in their native zero-copy batched-matmul layout
#      (`96.5%–97.2%` MXU utilization, `0` XLU transposes).
#   2. Streams contiguous `(1, C, bm)` slabs (`1.0 MiB` at `C=128, bm=4096`) to
#      VMEM with zero `4 KiB` strided scatter/gather DMAs.
#   3. Eliminates all VMEM `[bm, C] <-> [C, bm]` `XLU` transposes and performs
#      `axis=0` LayerNorm reductions down vector lanes (`15×` fewer `XLU` bundles).
#   4. Fuses across sublayers AND across consecutive layers:
#      - `InitialOutPrologue`: `norm_in -> pgg -> gated/masked (a_out, b_out, go_out)`
#      - `MegaKernel_A`: `OutEpilogue + InPrologue`
#      - `MegaKernel_B`: `InEpilogue + SwiGLU Transition [+ NextLayer OutPrologue]`
#        keeping intermediate `z_2` purely in VMEM (saving `1.69 GiB/sample` HBM).

_MAX_MEGA_TILE_ELEMS = 4 * 1024 * 1024  # 4M elements (~68.7 MiB peak VMEM on Trillium)


def resolve_channel_major_block_m(m: int, c_z: int, d_ff: int, requested_block_m: int) -> int:
  """Selects largest multiple of 128 dividing `m` within on-chip VMEM budget."""
  max_width = max(c_z, 2 * d_ff)
  vmem_safe_cap = max(_LANES, ((_MAX_MEGA_TILE_ELEMS // max_width) // _LANES) * _LANES)
  target = min(m, requested_block_m, vmem_safe_cap)
  bm = max(_LANES, (target // _LANES) * _LANES)
  while bm >= _LANES:
    if m % bm == 0:
      return bm
    bm -= _LANES
  raise ValueError(f"Flattened pair dimension M={m} must be a multiple of {_LANES}.")


def can_use_channel_major_megakernels(
    c_z: int,
    d_ff: int,
    seq_len: int,
    mesh: Mesh | Any = None,
) -> bool:
  """Returns True when `[B, C, N, N]` channel-major mega-kernels are supported."""
  if c_z < _LANES or c_z % _LANES != 0:
    return False
  if d_ff < _LANES or d_ff % _LANES != 0:
    return False
  m = seq_len * seq_len
  if m < _LANES or m % _LANES != 0:
    return False
  if mesh is not None:
    for ax in mesh.axis_names:
      if ax not in _BATCH_MESH_AXES and mesh.shape[ax] > 1:
        return False
  return True


def _can_use_tri_bcm(
    c_z: int,
    n_i: int,
    n_j: int,
    fp32_einsum: bool,
    interpret: bool,
) -> bool:
  """Returns True when zero-copy 3D `[B, C, M]` VMEM slab contractions apply (`C <= 256, N <= 512`)."""
  if fp32_einsum or interpret:
    return False
  if n_i != n_j or n_i < _LANES or n_i % _LANES != 0:
    return False
  if c_z < 8 or c_z > 256 or c_z % 8 != 0:
    return False
  return n_i * n_j <= 512 * 512


def _tri_bcm_fwd_kernel(a_r, b_r, t_r, *, bc: int, n_i: int, n_j: int, outgoing: bool):
  """Zero-copy VMEM channel-slab triangle contraction on `[1, bc=8, M=N*N]`."""
  a3 = a_r[0].reshape(bc, n_i, n_j)
  b3 = b_r[0].reshape(bc, n_i, n_j)
  outs = [
      (
          jnp.dot(a3[c], b3[c].T, preferred_element_type=jnp.float32)
          if outgoing
          else jnp.dot(a3[c].T, b3[c], preferred_element_type=jnp.float32)
      ).astype(jnp.bfloat16)
      for c in range(bc)
  ]
  t_r[...] = jnp.stack(outs, axis=0).reshape(1, bc, n_i * n_j)


def _tri_bcm_bwd_kernel(dt_r, a_r, b_r, da_r, db_r, *, bc: int, n_i: int, n_j: int, outgoing: bool):
  """Fused dual-output VMEM backward contraction reading `dt` once from `[1, bc=8, M]`."""
  dt3 = dt_r[0].reshape(bc, n_i, n_j)
  a3 = a_r[0].reshape(bc, n_i, n_j)
  b3 = b_r[0].reshape(bc, n_i, n_j)
  das, dbs = [], []
  for c in range(bc):
    dtc, ac, bc_m = dt3[c], a3[c], b3[c]
    if outgoing:
      das.append(jnp.dot(dtc, bc_m, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
      dbs.append(jnp.dot(dtc.T, ac, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
    else:
      das.append(jnp.dot(bc_m, dtc.T, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
      dbs.append(jnp.dot(ac, dtc, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
  da_r[...] = jnp.stack(das, axis=0).reshape(1, bc, n_i * n_j)
  db_r[...] = jnp.stack(dbs, axis=0).reshape(1, bc, n_i * n_j)


def _resolve_tri_bcm_bc(b_dim: int, c_z: int, m: int) -> int:
  """Selects ~2-4 MiB contiguous channel-slab width `bc` at `b_dim >= 2`, keeping `bc = 8` at `b_dim == 1`."""
  forced = int(os.environ.get("PAIRMIXER_FORCE_TRI_BC", "0"))
  if forced > 0 and c_z % forced == 0:
    return forced
  if b_dim <= 1:
    return 8
  target_bc = max(8, min(c_z, ((2 * 1024 * 1024) // max(1, m) // 8) * 8))
  while target_bc > 8 and c_z % target_bc != 0:
    target_bc -= 8
  return target_bc


def _run_tri_bcm_fwd(
    a_m: jax.Array,
    b_m: jax.Array,
    *,
    n_i: int,
    n_j: int,
    direction: str,
    interpret: bool = False,
) -> jax.Array:
  b_dim, c_z, m = a_m.shape
  bc = _resolve_tri_bcm_bc(b_dim, c_z, m)
  spec = pl.BlockSpec((1, bc, m), lambda bi, ci: (bi, ci, 0))
  cp = (
      None
      if interpret
      else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"), vmem_limit_bytes=120 * 1024 * 1024)
  )
  return pl.pallas_call(
      functools.partial(_tri_bcm_fwd_kernel, bc=bc, n_i=n_i, n_j=n_j, outgoing=(direction == "outgoing")),
      out_shape=jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(b_dim, c_z // bc),
          in_specs=[spec, spec],
          out_specs=spec,
      ),
      compiler_params=cp,
      interpret=interpret,
      name=f"pairmixer_tri_bcm_{direction}_fwd",
  )(a_m, b_m)


def _run_tri_bcm_bwd(
    dt_m: jax.Array,
    a_m: jax.Array,
    b_m: jax.Array,
    *,
    n_i: int,
    n_j: int,
    direction: str,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
  b_dim, c_z, m = a_m.shape
  bc = _resolve_tri_bcm_bc(b_dim, c_z, m)
  spec = pl.BlockSpec((1, bc, m), lambda bi, ci: (bi, ci, 0))
  out_s = jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16)
  cp = (
      None
      if interpret
      else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"), vmem_limit_bytes=120 * 1024 * 1024)
  )
  return pl.pallas_call(
      functools.partial(_tri_bcm_bwd_kernel, bc=bc, n_i=n_i, n_j=n_j, outgoing=(direction == "outgoing")),
      out_shape=[out_s, out_s],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(b_dim, c_z // bc),
          in_specs=[spec, spec, spec],
          out_specs=[spec, spec],
      ),
      compiler_params=cp,
      interpret=interpret,
      name=f"pairmixer_tri_bcm_{direction}_bwd",
  )(dt_m, a_m, b_m)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _tri_bcm_vjp(a_m: jax.Array, b_m: jax.Array, meta: tuple[Any, ...]) -> jax.Array:
  out, _ = _tri_bcm_vjp_fwd(a_m, b_m, meta)
  return out


def _tri_bcm_vjp_fwd(a_m: jax.Array, b_m: jax.Array, meta: tuple[Any, ...]):
  n_i, n_j, direction, interpret = meta
  t_m = _run_tri_bcm_fwd(a_m, b_m, n_i=n_i, n_j=n_j, direction=direction, interpret=interpret)
  return t_m, (a_m, b_m)


def _tri_bcm_vjp_bwd(meta: tuple[Any, ...], res, dt_m: jax.Array):
  n_i, n_j, direction, interpret = meta
  a_m, b_m = res
  return _run_tri_bcm_bwd(dt_m, a_m, b_m, n_i=n_i, n_j=n_j, direction=direction, interpret=interpret)


_tri_bcm_vjp.defvjp(_tri_bcm_vjp_fwd, _tri_bcm_vjp_bwd)


def _tri_einsum_fwd(
    a: jax.Array,
    b: jax.Array,
    *,
    direction: str,
    fp32_einsum: bool,
    matmul_precision: str,
) -> jax.Array:
  sub = "bdik,bdjk->bdij" if direction == "outgoing" else "bdki,bdkj->bdij"
  if fp32_einsum:
    return jnp.einsum(
        sub,
        a.astype(jnp.float32),
        b.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
  return jnp.einsum(
      sub,
      a.astype(jnp.bfloat16),
      b.astype(jnp.bfloat16),
      precision=lax.Precision(matmul_precision),
  ).astype(jnp.bfloat16)


def _tri_einsum_bwd(
    dt: jax.Array,
    a: jax.Array,
    b: jax.Array,
    *,
    direction: str,
    fp32_einsum: bool,
    matmul_precision: str,
) -> tuple[jax.Array, jax.Array]:
  if direction == "outgoing":
    sub_da, sub_db = "bdij,bdjk->bdik", "bdij,bdik->bdjk"
  else:
    sub_da, sub_db = "bdij,bdkj->bdki", "bdij,bdki->bdkj"
  if fp32_einsum:
    dt_f, a_f, b_f = dt.astype(jnp.float32), a.astype(jnp.float32), b.astype(jnp.float32)
    da = jnp.einsum(sub_da, dt_f, b_f, precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    db = jnp.einsum(sub_db, dt_f, a_f, precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return da.astype(jnp.bfloat16), db.astype(jnp.bfloat16)
  prec = lax.Precision(matmul_precision)
  da = jnp.einsum(sub_da, dt.astype(jnp.bfloat16), b.astype(jnp.bfloat16), precision=prec)
  db = jnp.einsum(sub_db, dt.astype(jnp.bfloat16), a.astype(jnp.bfloat16), precision=prec)
  return da.astype(jnp.bfloat16), db.astype(jnp.bfloat16)


# ------------------------------------------------------------------------------
# 3A. Standalone Channel-Major Prologue (`norm_in -> pgg -> a, b, go_sig`)
# ------------------------------------------------------------------------------


def _cm_prologue_fwd_kernel(z_r, m_r, si_r, bi_r, wp_r, a_r, b_r, go_r, *, eps: float, c_z: int, bm: int):
  zf = z_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu = jnp.mean(zf, axis=0, keepdims=True)
  rs = lax.rsqrt(jnp.mean(jnp.square(zf - mu), axis=0, keepdims=True) + eps)
  xn = ((zf - mu) * rs * (si_r[...] + 1.0) + bi_r[...]).astype(jnp.bfloat16)
  pgg = jnp.dot(wp_r[...], xn, preferred_element_type=jnp.float32)
  mf = m_r[0, 0:1, :].astype(jnp.float32)
  a_r[...] = (pgg[0:c_z, :] * jax.nn.sigmoid(pgg[2 * c_z : 3 * c_z, :]) * mf).astype(jnp.bfloat16).reshape(1, c_z, bm)
  b_r[...] = (
      (pgg[c_z : 2 * c_z, :] * jax.nn.sigmoid(pgg[3 * c_z : 4 * c_z, :]) * mf).astype(jnp.bfloat16).reshape(1, c_z, bm)
  )
  go_r[...] = jax.nn.sigmoid(pgg[4 * c_z : 5 * c_z, :]).astype(jnp.bfloat16).reshape(1, c_z, bm)


def _cm_prologue_bwd_kernel(
    z_r,
    m_r,
    da_r,
    db_r,
    dgo_r,
    dz_res_r,
    si_r,
    bi_r,
    wp_r,
    dz_out_r,
    dsi_r,
    dbi_r,
    dwp_r,
    *,
    eps: float,
    c_z: int,
    bm: int,
):
  is_first = (pl.program_id(0) == 0) & (pl.program_id(1) == 0)

  @pl.when(is_first)
  def _init():
    dsi_r[...] = jnp.zeros_like(dsi_r)
    dbi_r[...] = jnp.zeros_like(dbi_r)
    dwp_r[...] = jnp.zeros_like(dwp_r)

  zf = z_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu = jnp.mean(zf, axis=0, keepdims=True)
  rs = lax.rsqrt(jnp.mean(jnp.square(zf - mu), axis=0, keepdims=True) + eps)
  zh = (zf - mu) * rs
  si = si_r[...] + 1.0
  xn = (zh * si + bi_r[...]).astype(jnp.bfloat16)
  wp = wp_r[...]
  pgg = jnp.dot(wp, xn, preferred_element_type=jnp.float32)
  pa = pgg[0:c_z, :]
  pb = pgg[c_z : 2 * c_z, :]
  ga = pgg[2 * c_z : 3 * c_z, :]
  gb = pgg[3 * c_z : 4 * c_z, :]
  go = pgg[4 * c_z : 5 * c_z, :]

  mf = m_r[0, 0:1, :].astype(jnp.float32)
  da = da_r[...].reshape(c_z, bm).astype(jnp.float32) * mf
  db = db_r[...].reshape(c_z, bm).astype(jnp.float32) * mf
  dgo = dgo_r[...].reshape(c_z, bm).astype(jnp.float32)
  sga, sgb, sgo = jax.nn.sigmoid(ga), jax.nn.sigmoid(gb), jax.nn.sigmoid(go)
  dpgg = jnp.concatenate(
      [
          (da * sga).astype(jnp.bfloat16),
          (db * sgb).astype(jnp.bfloat16),
          (da * pa * sga * (1.0 - sga)).astype(jnp.bfloat16),
          (db * pb * sgb * (1.0 - sgb)).astype(jnp.bfloat16),
          (dgo * sgo * (1.0 - sgo)).astype(jnp.bfloat16),
      ],
      axis=0,
  )
  dwp_r[...] = dwp_r[...] + jnp.dot(dpgg, xn.T, preferred_element_type=jnp.float32)
  dxn = jnp.dot(wp.T, dpgg, preferred_element_type=jnp.float32)
  dsi_r[...] = dsi_r[...] + jnp.sum(dxn * zh, axis=1, keepdims=True)
  dbi_r[...] = dbi_r[...] + jnp.sum(dxn, axis=1, keepdims=True)
  dzh = dxn * si
  dz = dz_res_r[...].reshape(c_z, bm).astype(jnp.float32) + rs * (
      dzh - jnp.mean(dzh, axis=0, keepdims=True) - zh * jnp.mean(dzh * zh, axis=0, keepdims=True)
  )
  dz_out_r[...] = dz.astype(jnp.bfloat16).reshape(1, c_z, bm)


def _run_cm_prologue_fwd(z_bcnn, mask_8m, s_in, b_in, w_pgg, *, eps: float, bm: int, interpret: bool):
  b_dim, c_z, n_i, n_j = z_bcnn.shape
  m = n_i * n_j
  num_b = m // bm
  grid = (b_dim, num_b)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  out_shape = jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16)
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  a_m, b_m, go_m = pl.pallas_call(
      functools.partial(_cm_prologue_fwd_kernel, eps=eps, c_z=c_z, bm=bm),
      out_shape=[out_shape, out_shape, out_shape],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[act_spec, act_spec, act_spec],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_cm_prologue_fwd",
  )(
      z_bcnn.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      mask_8m,
      s_in.astype(jnp.float32).reshape(c_z, 1),
      b_in.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w_pgg.astype(jnp.bfloat16), 0, 1),
  )
  return (
      a_m.reshape(b_dim, c_z, n_i, n_j),
      b_m.reshape(b_dim, c_z, n_i, n_j),
      go_m.reshape(b_dim, c_z, n_i, n_j),
  )


def _run_cm_prologue_bwd(
    z_bcnn,
    mask_8m,
    da_bcnn,
    db_bcnn,
    dgo_bcnn,
    dz_res_bcnn,
    s_in,
    b_in,
    w_pgg,
    *,
    eps: float,
    bm: int,
    interpret: bool,
):
  b_dim, c_z, n_i, n_j = z_bcnn.shape
  m = n_i * n_j
  num_b = m // bm
  grid = (b_dim, num_b)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"))
  dz_m, dsi, dbi, dwp_t = pl.pallas_call(
      functools.partial(_cm_prologue_bwd_kernel, eps=eps, c_z=c_z, bm=bm),
      out_shape=[
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((5 * c_z, c_z), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_cm_prologue_bwd",
  )(
      z_bcnn.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      mask_8m,
      da_bcnn.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      db_bcnn.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      dgo_bcnn.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      dz_res_bcnn.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      s_in.astype(jnp.float32).reshape(c_z, 1),
      b_in.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w_pgg.astype(jnp.bfloat16), 0, 1),
  )
  return (
      dz_m.reshape(b_dim, c_z, n_i, n_j),
      dsi.reshape(c_z).astype(s_in.dtype),
      dbi.reshape(c_z).astype(b_in.dtype),
      jnp.swapaxes(dwp_t, 0, 1).astype(w_pgg.dtype),
  )


# ------------------------------------------------------------------------------
# 3B. MegaKernel A: Outgoing Epilogue + Incoming Prologue (Cross-Sublayer Fused)
# ------------------------------------------------------------------------------


def _mega_a_fwd_kernel(
    t_r,
    z0_r,
    go1_r,
    m_r,
    so_r,
    bo_r,
    wo_r,
    si_r,
    bi_r,
    wp_r,
    z1_r,
    a_r,
    b_r,
    go2_r,
    *,
    eps: float,
    c_z: int,
    bm: int,
):
  tf = t_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu = jnp.mean(tf, axis=0, keepdims=True)
  rs = lax.rsqrt(jnp.mean(jnp.square(tf - mu), axis=0, keepdims=True) + eps)
  tn = ((tf - mu) * rs * (so_r[...] + 1.0) + bo_r[...]).astype(jnp.bfloat16)
  po = jnp.dot(wo_r[...], tn, preferred_element_type=jnp.float32)
  z1_bf16 = (
      z0_r[...].reshape(c_z, bm).astype(jnp.float32) + po * go1_r[...].reshape(c_z, bm).astype(jnp.float32)
  ).astype(jnp.bfloat16)
  z1_r[...] = z1_bf16.reshape(1, c_z, bm)

  z1f = z1_bf16.astype(jnp.float32)
  mu2 = jnp.mean(z1f, axis=0, keepdims=True)
  rs2 = lax.rsqrt(jnp.mean(jnp.square(z1f - mu2), axis=0, keepdims=True) + eps)
  xn2 = ((z1f - mu2) * rs2 * (si_r[...] + 1.0) + bi_r[...]).astype(jnp.bfloat16)
  pgg = jnp.dot(wp_r[...], xn2, preferred_element_type=jnp.float32)
  mf = m_r[0, 0:1, :].astype(jnp.float32)
  a_r[...] = (pgg[0:c_z, :] * jax.nn.sigmoid(pgg[2 * c_z : 3 * c_z, :]) * mf).astype(jnp.bfloat16).reshape(1, c_z, bm)
  b_r[...] = (
      (pgg[c_z : 2 * c_z, :] * jax.nn.sigmoid(pgg[3 * c_z : 4 * c_z, :]) * mf).astype(jnp.bfloat16).reshape(1, c_z, bm)
  )
  go2_r[...] = jax.nn.sigmoid(pgg[4 * c_z : 5 * c_z, :]).astype(jnp.bfloat16).reshape(1, c_z, bm)


def _mega_a_bwd_kernel(
    t_r,
    go1_r,
    z1_r,
    m_r,
    da_r,
    db_r,
    dgo2_r,
    dz1_res_r,
    so_r,
    bo_r,
    wo_r,
    si_r,
    bi_r,
    wp_r,
    dt_r,
    dz0_r,
    dgo1_r,
    dso_r,
    dbo_r,
    dwo_r,
    dsi_r,
    dbi_r,
    dwp_r,
    *,
    eps: float,
    c_z: int,
    bm: int,
):
  is_first = (pl.program_id(0) == 0) & (pl.program_id(1) == 0)

  @pl.when(is_first)
  def _init():
    dso_r[...] = jnp.zeros_like(dso_r)
    dbo_r[...] = jnp.zeros_like(dbo_r)
    dwo_r[...] = jnp.zeros_like(dwo_r)
    dsi_r[...] = jnp.zeros_like(dsi_r)
    dbi_r[...] = jnp.zeros_like(dbi_r)
    dwp_r[...] = jnp.zeros_like(dwp_r)

  z1f = z1_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu2 = jnp.mean(z1f, axis=0, keepdims=True)
  rs2 = lax.rsqrt(jnp.mean(jnp.square(z1f - mu2), axis=0, keepdims=True) + eps)
  zh2 = (z1f - mu2) * rs2
  si = si_r[...] + 1.0
  xn2 = (zh2 * si + bi_r[...]).astype(jnp.bfloat16)
  wp = wp_r[...]
  pgg = jnp.dot(wp, xn2, preferred_element_type=jnp.float32)
  pa = pgg[0:c_z, :]
  pb = pgg[c_z : 2 * c_z, :]
  ga = pgg[2 * c_z : 3 * c_z, :]
  gb = pgg[3 * c_z : 4 * c_z, :]
  go = pgg[4 * c_z : 5 * c_z, :]

  mf = m_r[0, 0:1, :].astype(jnp.float32)
  da = da_r[...].reshape(c_z, bm).astype(jnp.float32) * mf
  db = db_r[...].reshape(c_z, bm).astype(jnp.float32) * mf
  dgo2 = dgo2_r[...].reshape(c_z, bm).astype(jnp.float32)
  sga, sgb, sgo = jax.nn.sigmoid(ga), jax.nn.sigmoid(gb), jax.nn.sigmoid(go)
  dpgg = jnp.concatenate(
      [
          (da * sga).astype(jnp.bfloat16),
          (db * sgb).astype(jnp.bfloat16),
          (da * pa * sga * (1.0 - sga)).astype(jnp.bfloat16),
          (db * pb * sgb * (1.0 - sgb)).astype(jnp.bfloat16),
          (dgo2 * sgo * (1.0 - sgo)).astype(jnp.bfloat16),
      ],
      axis=0,
  )
  dwp_r[...] = dwp_r[...] + jnp.dot(dpgg, xn2.T, preferred_element_type=jnp.float32)
  dxn2 = jnp.dot(wp.T, dpgg, preferred_element_type=jnp.float32)
  dsi_r[...] = dsi_r[...] + jnp.sum(dxn2 * zh2, axis=1, keepdims=True)
  dbi_r[...] = dbi_r[...] + jnp.sum(dxn2, axis=1, keepdims=True)
  dzh2 = dxn2 * si
  dz1 = dz1_res_r[...].reshape(c_z, bm).astype(jnp.float32) + rs2 * (
      dzh2 - jnp.mean(dzh2, axis=0, keepdims=True) - zh2 * jnp.mean(dzh2 * zh2, axis=0, keepdims=True)
  )
  dz0_r[...] = dz1.astype(jnp.bfloat16).reshape(1, c_z, bm)

  tf = t_r[...].reshape(c_z, bm).astype(jnp.float32)
  go1 = go1_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu = jnp.mean(tf, axis=0, keepdims=True)
  rs = lax.rsqrt(jnp.mean(jnp.square(tf - mu), axis=0, keepdims=True) + eps)
  th = (tf - mu) * rs
  so = so_r[...] + 1.0
  tn = (th * so + bo_r[...]).astype(jnp.bfloat16)
  wo = wo_r[...]
  po = jnp.dot(wo, tn, preferred_element_type=jnp.float32)
  dgo1_r[...] = (dz1 * po).astype(jnp.bfloat16).reshape(1, c_z, bm)
  dpo = (dz1 * go1).astype(jnp.bfloat16)
  dwo_r[...] = dwo_r[...] + jnp.dot(dpo, tn.T, preferred_element_type=jnp.float32)
  dtn = jnp.dot(wo.T, dpo, preferred_element_type=jnp.float32)
  dso_r[...] = dso_r[...] + jnp.sum(dtn * th, axis=1, keepdims=True)
  dbo_r[...] = dbo_r[...] + jnp.sum(dtn, axis=1, keepdims=True)
  dth = dtn * so
  dt = rs * (dth - jnp.mean(dth, axis=0, keepdims=True) - th * jnp.mean(dth * th, axis=0, keepdims=True))
  dt_r[...] = dt.astype(jnp.bfloat16).reshape(1, c_z, bm)


def _run_mega_a_fwd(
    t_out,
    z0,
    go_out,
    mask_8m,
    so1,
    bo1,
    wo1,
    si2,
    bi2,
    wp2,
    *,
    eps: float,
    bm: int,
    interpret: bool,
):
  b_dim, c_z, n_i, n_j = z0.shape
  m = n_i * n_j
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  out_shape = jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16)
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  z1_m, a_in_m, b_in_m, go_in_m = pl.pallas_call(
      functools.partial(_mega_a_fwd_kernel, eps=eps, c_z=c_z, bm=bm),
      out_shape=[out_shape, out_shape, out_shape, out_shape],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[act_spec, act_spec, act_spec, act_spec],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_mega_a_fwd",
  )(
      t_out.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      z0.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      go_out.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      mask_8m,
      so1.astype(jnp.float32).reshape(c_z, 1),
      bo1.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo1.astype(jnp.bfloat16), 0, 1),
      si2.astype(jnp.float32).reshape(c_z, 1),
      bi2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wp2.astype(jnp.bfloat16), 0, 1),
  )
  return (
      z1_m.reshape(b_dim, c_z, n_i, n_j),
      a_in_m.reshape(b_dim, c_z, n_i, n_j),
      b_in_m.reshape(b_dim, c_z, n_i, n_j),
      go_in_m.reshape(b_dim, c_z, n_i, n_j),
  )


def _run_mega_a_bwd(
    t_out,
    go_out,
    z1,
    mask_8m,
    da_in,
    db_in,
    dgo_in,
    dz1_res,
    so1,
    bo1,
    wo1,
    si2,
    bi2,
    wp2,
    *,
    eps: float,
    bm: int,
    interpret: bool,
):
  b_dim, c_z, n_i, n_j = z1.shape
  m = n_i * n_j
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"))
  dt_m, dz0_m, dgo1_m, dso1, dbo1, dwo1_t, dsi2, dbi2, dwp2_t = pl.pallas_call(
      functools.partial(_mega_a_bwd_kernel, eps=eps, c_z=c_z, bm=bm),
      out_shape=[
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((5 * c_z, c_z), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_mega_a_bwd",
  )(
      t_out.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      go_out.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      z1.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      mask_8m,
      da_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      db_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      dgo_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      dz1_res.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      so1.astype(jnp.float32).reshape(c_z, 1),
      bo1.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo1.astype(jnp.bfloat16), 0, 1),
      si2.astype(jnp.float32).reshape(c_z, 1),
      bi2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wp2.astype(jnp.bfloat16), 0, 1),
  )
  return (
      dt_m.reshape(b_dim, c_z, n_i, n_j),
      dz0_m.reshape(b_dim, c_z, n_i, n_j),
      dgo1_m.reshape(b_dim, c_z, n_i, n_j),
      dso1.reshape(c_z).astype(so1.dtype),
      dbo1.reshape(c_z).astype(bo1.dtype),
      jnp.swapaxes(dwo1_t, 0, 1).astype(wo1.dtype),
      dsi2.reshape(c_z).astype(si2.dtype),
      dbi2.reshape(c_z).astype(bi2.dtype),
      jnp.swapaxes(dwp2_t, 0, 1).astype(wp2.dtype),
  )


# ------------------------------------------------------------------------------
# 3C. MegaKernel B: Incoming Epilogue + Transition [+ Next Layer OutPrologue]
#     (`z_2` stays 100% inside on-chip VMEM — NEVER materialized to HBM!)
# ------------------------------------------------------------------------------


def _final_mega_b_fwd_kernel(
    t_r,
    z1_r,
    go2_r,
    so_r,
    bo_r,
    wo_r,
    st_r,
    bt_r,
    w12_r,
    w3_r,
    z3_r,
    *,
    eps: float,
    c_z: int,
    d_ff: int,
    bm: int,
):
  tf = t_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu = jnp.mean(tf, axis=0, keepdims=True)
  rs = lax.rsqrt(jnp.mean(jnp.square(tf - mu), axis=0, keepdims=True) + eps)
  tn = ((tf - mu) * rs * (so_r[...] + 1.0) + bo_r[...]).astype(jnp.bfloat16)
  po = jnp.dot(wo_r[...], tn, preferred_element_type=jnp.float32)
  z2f = (
      (z1_r[...].reshape(c_z, bm).astype(jnp.float32) + po * go2_r[...].reshape(c_z, bm).astype(jnp.float32))
      .astype(jnp.bfloat16)
      .astype(jnp.float32)
  )

  mut = jnp.mean(z2f, axis=0, keepdims=True)
  rst = lax.rsqrt(jnp.mean(jnp.square(z2f - mut), axis=0, keepdims=True) + eps)
  xnt = ((z2f - mut) * rst * (st_r[...] + 1.0) + bt_r[...]).astype(jnp.bfloat16)
  h12 = jnp.dot(w12_r[...], xnt, preferred_element_type=jnp.float32)
  h = (jax.nn.silu(h12[:d_ff, :]) * h12[d_ff:, :]).astype(jnp.bfloat16)
  z3_r[...] = (z2f + jnp.dot(w3_r[...], h, preferred_element_type=jnp.float32)).astype(jnp.bfloat16).reshape(1, c_z, bm)


def _final_mega_b_bwd_kernel(
    t_r,
    z1_r,
    go2_r,
    dz3_r,
    so_r,
    bo_r,
    wo_r,
    st_r,
    bt_r,
    w12_r,
    w3_r,
    dt_r,
    dz1_out_r,
    dgo2_r,
    dso_r,
    dbo_r,
    dwo_r,
    dst_r,
    dbt_r,
    dw12_r,
    dw3_r,
    *,
    eps: float,
    c_z: int,
    d_ff: int,
    bm: int,
):
  is_first = (pl.program_id(0) == 0) & (pl.program_id(1) == 0)

  @pl.when(is_first)
  def _init():
    dso_r[...] = jnp.zeros_like(dso_r)
    dbo_r[...] = jnp.zeros_like(dbo_r)
    dwo_r[...] = jnp.zeros_like(dwo_r)
    dst_r[...] = jnp.zeros_like(dst_r)
    dbt_r[...] = jnp.zeros_like(dbt_r)
    dw12_r[...] = jnp.zeros_like(dw12_r)
    dw3_r[...] = jnp.zeros_like(dw3_r)

  tf = t_r[...].reshape(c_z, bm).astype(jnp.float32)
  go2 = go2_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu = jnp.mean(tf, axis=0, keepdims=True)
  rs = lax.rsqrt(jnp.mean(jnp.square(tf - mu), axis=0, keepdims=True) + eps)
  th = (tf - mu) * rs
  so = so_r[...] + 1.0
  tn = (th * so + bo_r[...]).astype(jnp.bfloat16)
  wo = wo_r[...]
  po = jnp.dot(wo, tn, preferred_element_type=jnp.float32)
  z2f = (z1_r[...].reshape(c_z, bm).astype(jnp.float32) + po * go2).astype(jnp.bfloat16).astype(jnp.float32)

  mut = jnp.mean(z2f, axis=0, keepdims=True)
  rst = lax.rsqrt(jnp.mean(jnp.square(z2f - mut), axis=0, keepdims=True) + eps)
  zht = (z2f - mut) * rst
  st = st_r[...] + 1.0
  xnt = (zht * st + bt_r[...]).astype(jnp.bfloat16)
  w12 = w12_r[...]
  w3 = w3_r[...]
  h12 = jnp.dot(w12, xnt, preferred_element_type=jnp.float32)
  h1, h2 = h12[:d_ff, :], h12[d_ff:, :]
  sig1 = jax.nn.sigmoid(h1)
  silu1 = h1 * sig1
  h = (silu1 * h2).astype(jnp.bfloat16)

  dz3_bf16 = dz3_r[...].reshape(c_z, bm)
  dz3 = dz3_bf16.astype(jnp.float32)
  dw3_r[...] = dw3_r[...] + jnp.dot(dz3_bf16, h.T, preferred_element_type=jnp.float32)
  dh = jnp.dot(w3.T, dz3_bf16, preferred_element_type=jnp.float32)
  dh12 = jnp.concatenate(
      [
          (dh * h2 * sig1 * (1.0 + h1 * (1.0 - sig1))).astype(jnp.bfloat16),
          (dh * silu1).astype(jnp.bfloat16),
      ],
      axis=0,
  )
  dw12_r[...] = dw12_r[...] + jnp.dot(dh12, xnt.T, preferred_element_type=jnp.float32)
  dxnt = jnp.dot(w12.T, dh12, preferred_element_type=jnp.float32)
  dst_r[...] = dst_r[...] + jnp.sum(dxnt * zht, axis=1, keepdims=True)
  dbt_r[...] = dbt_r[...] + jnp.sum(dxnt, axis=1, keepdims=True)
  dzht = dxnt * st
  dz2 = dz3 + rst * (dzht - jnp.mean(dzht, axis=0, keepdims=True) - zht * jnp.mean(dzht * zht, axis=0, keepdims=True))
  dz1_out_r[...] = dz2.astype(jnp.bfloat16).reshape(1, c_z, bm)

  dgo2_r[...] = (dz2 * po).astype(jnp.bfloat16).reshape(1, c_z, bm)
  dpo = (dz2 * go2).astype(jnp.bfloat16)
  dwo_r[...] = dwo_r[...] + jnp.dot(dpo, tn.T, preferred_element_type=jnp.float32)
  dtn = jnp.dot(wo.T, dpo, preferred_element_type=jnp.float32)
  dso_r[...] = dso_r[...] + jnp.sum(dtn * th, axis=1, keepdims=True)
  dbo_r[...] = dbo_r[...] + jnp.sum(dtn, axis=1, keepdims=True)
  dth = dtn * so
  dt = rs * (dth - jnp.mean(dth, axis=0, keepdims=True) - th * jnp.mean(dth * th, axis=0, keepdims=True))
  dt_r[...] = dt.astype(jnp.bfloat16).reshape(1, c_z, bm)


def _cross_mega_b_fwd_kernel(
    t_r,
    z1_r,
    go2_r,
    m_r,
    so_r,
    bo_r,
    wo_r,
    st_r,
    bt_r,
    w12_r,
    w3_r,
    si_r,
    bi_r,
    wp_r,
    z3_r,
    a_r,
    b_r,
    go3_r,
    *,
    eps: float,
    c_z: int,
    d_ff: int,
    bm: int,
):
  tf = t_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu = jnp.mean(tf, axis=0, keepdims=True)
  rs = lax.rsqrt(jnp.mean(jnp.square(tf - mu), axis=0, keepdims=True) + eps)
  tn = ((tf - mu) * rs * (so_r[...] + 1.0) + bo_r[...]).astype(jnp.bfloat16)
  po = jnp.dot(wo_r[...], tn, preferred_element_type=jnp.float32)
  z2f = (
      (z1_r[...].reshape(c_z, bm).astype(jnp.float32) + po * go2_r[...].reshape(c_z, bm).astype(jnp.float32))
      .astype(jnp.bfloat16)
      .astype(jnp.float32)
  )

  mut = jnp.mean(z2f, axis=0, keepdims=True)
  rst = lax.rsqrt(jnp.mean(jnp.square(z2f - mut), axis=0, keepdims=True) + eps)
  xnt = ((z2f - mut) * rst * (st_r[...] + 1.0) + bt_r[...]).astype(jnp.bfloat16)
  h12 = jnp.dot(w12_r[...], xnt, preferred_element_type=jnp.float32)
  h = (jax.nn.silu(h12[:d_ff, :]) * h12[d_ff:, :]).astype(jnp.bfloat16)
  z3_bf16 = (z2f + jnp.dot(w3_r[...], h, preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
  z3_r[...] = z3_bf16.reshape(1, c_z, bm)

  z3f = z3_bf16.astype(jnp.float32)
  mu3 = jnp.mean(z3f, axis=0, keepdims=True)
  rs3 = lax.rsqrt(jnp.mean(jnp.square(z3f - mu3), axis=0, keepdims=True) + eps)
  xn3 = ((z3f - mu3) * rs3 * (si_r[...] + 1.0) + bi_r[...]).astype(jnp.bfloat16)
  pgg = jnp.dot(wp_r[...], xn3, preferred_element_type=jnp.float32)
  mf = m_r[0, 0:1, :].astype(jnp.float32)
  a_r[...] = (pgg[0:c_z, :] * jax.nn.sigmoid(pgg[2 * c_z : 3 * c_z, :]) * mf).astype(jnp.bfloat16).reshape(1, c_z, bm)
  b_r[...] = (
      (pgg[c_z : 2 * c_z, :] * jax.nn.sigmoid(pgg[3 * c_z : 4 * c_z, :]) * mf).astype(jnp.bfloat16).reshape(1, c_z, bm)
  )
  go3_r[...] = jax.nn.sigmoid(pgg[4 * c_z : 5 * c_z, :]).astype(jnp.bfloat16).reshape(1, c_z, bm)


def _cross_mega_b_bwd_kernel(
    t_r,
    z1_r,
    go2_r,
    z3_r,
    m_r,
    da_r,
    db_r,
    dgo3_r,
    dz3_res_r,
    so_r,
    bo_r,
    wo_r,
    st_r,
    bt_r,
    w12_r,
    w3_r,
    si_r,
    bi_r,
    wp_r,
    dt_r,
    dz1_out_r,
    dgo2_r,
    dso_r,
    dbo_r,
    dwo_r,
    dst_r,
    dbt_r,
    dw12_r,
    dw3_r,
    dsi_r,
    dbi_r,
    dwp_r,
    *,
    eps: float,
    c_z: int,
    d_ff: int,
    bm: int,
):
  is_first = (pl.program_id(0) == 0) & (pl.program_id(1) == 0)

  @pl.when(is_first)
  def _init():
    dso_r[...] = jnp.zeros_like(dso_r)
    dbo_r[...] = jnp.zeros_like(dbo_r)
    dwo_r[...] = jnp.zeros_like(dwo_r)
    dst_r[...] = jnp.zeros_like(dst_r)
    dbt_r[...] = jnp.zeros_like(dbt_r)
    dw12_r[...] = jnp.zeros_like(dw12_r)
    dw3_r[...] = jnp.zeros_like(dw3_r)
    dsi_r[...] = jnp.zeros_like(dsi_r)
    dbi_r[...] = jnp.zeros_like(dbi_r)
    dwp_r[...] = jnp.zeros_like(dwp_r)

  z3f = z3_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu3 = jnp.mean(z3f, axis=0, keepdims=True)
  rs3 = lax.rsqrt(jnp.mean(jnp.square(z3f - mu3), axis=0, keepdims=True) + eps)
  zh3 = (z3f - mu3) * rs3
  si = si_r[...] + 1.0
  xn3 = (zh3 * si + bi_r[...]).astype(jnp.bfloat16)
  wp = wp_r[...]
  pgg = jnp.dot(wp, xn3, preferred_element_type=jnp.float32)
  pa = pgg[0:c_z, :]
  pb = pgg[c_z : 2 * c_z, :]
  ga = pgg[2 * c_z : 3 * c_z, :]
  gb = pgg[3 * c_z : 4 * c_z, :]
  go = pgg[4 * c_z : 5 * c_z, :]

  mf = m_r[0, 0:1, :].astype(jnp.float32)
  da = da_r[...].reshape(c_z, bm).astype(jnp.float32) * mf
  db = db_r[...].reshape(c_z, bm).astype(jnp.float32) * mf
  dgo3 = dgo3_r[...].reshape(c_z, bm).astype(jnp.float32)
  sga, sgb, sgo = jax.nn.sigmoid(ga), jax.nn.sigmoid(gb), jax.nn.sigmoid(go)
  dpgg = jnp.concatenate(
      [
          (da * sga).astype(jnp.bfloat16),
          (db * sgb).astype(jnp.bfloat16),
          (da * pa * sga * (1.0 - sga)).astype(jnp.bfloat16),
          (db * pb * sgb * (1.0 - sgb)).astype(jnp.bfloat16),
          (dgo3 * sgo * (1.0 - sgo)).astype(jnp.bfloat16),
      ],
      axis=0,
  )
  dwp_r[...] = dwp_r[...] + jnp.dot(dpgg, xn3.T, preferred_element_type=jnp.float32)
  dxn3 = jnp.dot(wp.T, dpgg, preferred_element_type=jnp.float32)
  dsi_r[...] = dsi_r[...] + jnp.sum(dxn3 * zh3, axis=1, keepdims=True)
  dbi_r[...] = dbi_r[...] + jnp.sum(dxn3, axis=1, keepdims=True)
  dzh3 = dxn3 * si
  dz3 = dz3_res_r[...].reshape(c_z, bm).astype(jnp.float32) + rs3 * (
      dzh3 - jnp.mean(dzh3, axis=0, keepdims=True) - zh3 * jnp.mean(dzh3 * zh3, axis=0, keepdims=True)
  )

  tf = t_r[...].reshape(c_z, bm).astype(jnp.float32)
  go2 = go2_r[...].reshape(c_z, bm).astype(jnp.float32)
  mu = jnp.mean(tf, axis=0, keepdims=True)
  rs = lax.rsqrt(jnp.mean(jnp.square(tf - mu), axis=0, keepdims=True) + eps)
  th = (tf - mu) * rs
  so = so_r[...] + 1.0
  tn = (th * so + bo_r[...]).astype(jnp.bfloat16)
  wo = wo_r[...]
  po = jnp.dot(wo, tn, preferred_element_type=jnp.float32)
  z2f = (z1_r[...].reshape(c_z, bm).astype(jnp.float32) + po * go2).astype(jnp.bfloat16).astype(jnp.float32)

  mut = jnp.mean(z2f, axis=0, keepdims=True)
  rst = lax.rsqrt(jnp.mean(jnp.square(z2f - mut), axis=0, keepdims=True) + eps)
  zht = (z2f - mut) * rst
  st = st_r[...] + 1.0
  xnt = (zht * st + bt_r[...]).astype(jnp.bfloat16)
  w12 = w12_r[...]
  w3 = w3_r[...]
  h12 = jnp.dot(w12, xnt, preferred_element_type=jnp.float32)
  h1, h2 = h12[:d_ff, :], h12[d_ff:, :]
  sig1 = jax.nn.sigmoid(h1)
  silu1 = h1 * sig1
  h = (silu1 * h2).astype(jnp.bfloat16)
  dz3_bf16 = dz3.astype(jnp.bfloat16)
  dw3_r[...] = dw3_r[...] + jnp.dot(dz3_bf16, h.T, preferred_element_type=jnp.float32)
  dh = jnp.dot(w3.T, dz3_bf16, preferred_element_type=jnp.float32)
  dh12 = jnp.concatenate(
      [
          (dh * h2 * sig1 * (1.0 + h1 * (1.0 - sig1))).astype(jnp.bfloat16),
          (dh * silu1).astype(jnp.bfloat16),
      ],
      axis=0,
  )
  dw12_r[...] = dw12_r[...] + jnp.dot(dh12, xnt.T, preferred_element_type=jnp.float32)
  dxnt = jnp.dot(w12.T, dh12, preferred_element_type=jnp.float32)
  dst_r[...] = dst_r[...] + jnp.sum(dxnt * zht, axis=1, keepdims=True)
  dbt_r[...] = dbt_r[...] + jnp.sum(dxnt, axis=1, keepdims=True)
  dzht = dxnt * st
  dz2 = dz3 + rst * (dzht - jnp.mean(dzht, axis=0, keepdims=True) - zht * jnp.mean(dzht * zht, axis=0, keepdims=True))
  dz1_out_r[...] = dz2.astype(jnp.bfloat16).reshape(1, c_z, bm)

  dgo2_r[...] = (dz2 * po).astype(jnp.bfloat16).reshape(1, c_z, bm)
  dpo = (dz2 * go2).astype(jnp.bfloat16)
  dwo_r[...] = dwo_r[...] + jnp.dot(dpo, tn.T, preferred_element_type=jnp.float32)
  dtn = jnp.dot(wo.T, dpo, preferred_element_type=jnp.float32)
  dso_r[...] = dso_r[...] + jnp.sum(dtn * th, axis=1, keepdims=True)
  dbo_r[...] = dbo_r[...] + jnp.sum(dtn, axis=1, keepdims=True)
  dth = dtn * so
  dt = rs * (dth - jnp.mean(dth, axis=0, keepdims=True) - th * jnp.mean(dth * th, axis=0, keepdims=True))
  dt_r[...] = dt.astype(jnp.bfloat16).reshape(1, c_z, bm)


def _run_final_mega_b_fwd(t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3, *, eps: float, bm: int, interpret: bool):
  b_dim, c_z, n_i, n_j = z1.shape
  d_ff = w3.shape[0]
  m = n_i * n_j
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  z3_m = pl.pallas_call(
      functools.partial(_final_mega_b_fwd_kernel, eps=eps, c_z=c_z, d_ff=d_ff, bm=bm),
      out_shape=jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
          ],
          out_specs=act_spec,
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_final_mega_b_fwd",
  )(
      t_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      z1.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      go_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      so2.astype(jnp.float32).reshape(c_z, 1),
      bo2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo2.astype(jnp.bfloat16), 0, 1),
      st.astype(jnp.float32).reshape(c_z, 1),
      bt.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w12.astype(jnp.bfloat16), 0, 1),
      jnp.swapaxes(w3.astype(jnp.bfloat16), 0, 1),
  )
  return z3_m.reshape(b_dim, c_z, n_i, n_j)


def _run_final_mega_b_bwd(
    t_in,
    z1,
    go_in,
    dz3,
    so2,
    bo2,
    wo2,
    st,
    bt,
    w12,
    w3,
    *,
    eps: float,
    bm: int,
    interpret: bool,
):
  b_dim, c_z, n_i, n_j = z1.shape
  d_ff = w3.shape[0]
  m = n_i * n_j
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"))
  dt_m, dz1_m, dgo2_m, dso2, dbo2, dwo2_t, dst, dbt, dw12_t, dw3_t = pl.pallas_call(
      functools.partial(_final_mega_b_bwd_kernel, eps=eps, c_z=c_z, d_ff=d_ff, bm=bm),
      out_shape=[
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((2 * d_ff, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, d_ff), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
          ],
          out_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
          ],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_final_mega_b_bwd",
  )(
      t_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      z1.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      go_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      dz3.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      so2.astype(jnp.float32).reshape(c_z, 1),
      bo2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo2.astype(jnp.bfloat16), 0, 1),
      st.astype(jnp.float32).reshape(c_z, 1),
      bt.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w12.astype(jnp.bfloat16), 0, 1),
      jnp.swapaxes(w3.astype(jnp.bfloat16), 0, 1),
  )
  return (
      dt_m.reshape(b_dim, c_z, n_i, n_j),
      dz1_m.reshape(b_dim, c_z, n_i, n_j),
      dgo2_m.reshape(b_dim, c_z, n_i, n_j),
      dso2.reshape(c_z).astype(so2.dtype),
      dbo2.reshape(c_z).astype(bo2.dtype),
      jnp.swapaxes(dwo2_t, 0, 1).astype(wo2.dtype),
      dst.reshape(c_z).astype(st.dtype),
      dbt.reshape(c_z).astype(bt.dtype),
      jnp.swapaxes(dw12_t, 0, 1).astype(w12.dtype),
      jnp.swapaxes(dw3_t, 0, 1).astype(w3.dtype),
  )


# ------------------------------------------------------------------------------
# 3D. Custom-VJP Primitives:
#     1. Cross-Layer Primitives (`cm_prologue`, `mega_a`, `cross_mega_b`, `final_mega_b`)
#     2. Full-Layer Compact Primitive (`pallas_pairmixer_layer_bcnn`, saving only
#        4 tensors `(z0, t_out, z1, t_in)` for B_local >= 2 up to ~95% HBM!)
# ------------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _cm_prologue_vjp(z, mask_8m, s_in, b_in, w_pgg, meta):
  out, _ = _cm_prologue_vjp_fwd(z, mask_8m, s_in, b_in, w_pgg, meta)
  return out


def _cm_prologue_vjp_fwd(z, mask_8m, s_in, b_in, w_pgg, meta):
  eps, bm, interpret = meta
  a, b, go = _run_cm_prologue_fwd(z, mask_8m, s_in, b_in, w_pgg, eps=eps, bm=bm, interpret=interpret)
  return (a, b, go), (z, mask_8m, s_in, b_in, w_pgg)


def _cm_prologue_vjp_bwd(meta, res, ct):
  eps, bm, interpret = meta
  z, mask_8m, s_in, b_in, w_pgg = res
  da, db, dgo = ct
  dz, dsi, dbi, dwp = _run_cm_prologue_bwd(
      z,
      mask_8m,
      da,
      db,
      dgo,
      jnp.zeros_like(z, dtype=jnp.bfloat16),
      s_in,
      b_in,
      w_pgg,
      eps=eps,
      bm=bm,
      interpret=interpret,
  )
  return dz.astype(z.dtype), None, dsi, dbi, dwp


_cm_prologue_vjp.defvjp(_cm_prologue_vjp_fwd, _cm_prologue_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(10,))
def _mega_a_vjp(t_out, z0, go_out, mask_8m, so1, bo1, wo1, si2, bi2, wp2, meta):
  out, _ = _mega_a_vjp_fwd(t_out, z0, go_out, mask_8m, so1, bo1, wo1, si2, bi2, wp2, meta)
  return out


def _mega_a_vjp_fwd(t_out, z0, go_out, mask_8m, so1, bo1, wo1, si2, bi2, wp2, meta):
  eps, bm, interpret = meta
  z1, a_in, b_in, go_in = _run_mega_a_fwd(
      t_out, z0, go_out, mask_8m, so1, bo1, wo1, si2, bi2, wp2, eps=eps, bm=bm, interpret=interpret
  )
  return (z1, a_in, b_in, go_in), (t_out, go_out, z1, mask_8m, so1, bo1, wo1, si2, bi2, wp2)


def _mega_a_vjp_bwd(meta, res, ct):
  eps, bm, interpret = meta
  t_out, go_out, z1, mask_8m, so1, bo1, wo1, si2, bi2, wp2 = res
  dz1_res, da_in, db_in, dgo_in = ct
  dt, dz0, dgo1, dso1, dbo1, dwo1, dsi2, dbi2, dwp2 = _run_mega_a_bwd(
      t_out,
      go_out,
      z1,
      mask_8m,
      da_in,
      db_in,
      dgo_in,
      dz1_res,
      so1,
      bo1,
      wo1,
      si2,
      bi2,
      wp2,
      eps=eps,
      bm=bm,
      interpret=interpret,
  )
  return dt, dz0, dgo1, None, dso1, dbo1, dwo1, dsi2, dbi2, dwp2


_mega_a_vjp.defvjp(_mega_a_vjp_fwd, _mega_a_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(14,))
def _cross_mega_b_vjp(t_in, z1, go_in, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3, meta):
  out, _ = _cross_mega_b_vjp_fwd(t_in, z1, go_in, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3, meta)
  return out


def _cross_mega_b_vjp_fwd(t_in, z1, go_in, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3, meta):
  eps, bm, interpret = meta
  b_dim, c_z, n_i, n_j = z1.shape
  d_ff = w3.shape[0]
  m = n_i * n_j
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  out_shape = jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16)
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  z3_m, a_m, b_m, go_m = pl.pallas_call(
      functools.partial(_cross_mega_b_fwd_kernel, eps=eps, c_z=c_z, d_ff=d_ff, bm=bm),
      out_shape=[out_shape, out_shape, out_shape, out_shape],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[act_spec, act_spec, act_spec, act_spec],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_cross_mega_b_fwd",
  )(
      t_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      z1.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      go_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      mask_8m,
      so2.astype(jnp.float32).reshape(c_z, 1),
      bo2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo2.astype(jnp.bfloat16), 0, 1),
      st.astype(jnp.float32).reshape(c_z, 1),
      bt.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w12.astype(jnp.bfloat16), 0, 1),
      jnp.swapaxes(w3.astype(jnp.bfloat16), 0, 1),
      si3.astype(jnp.float32).reshape(c_z, 1),
      bi3.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wp3.astype(jnp.bfloat16), 0, 1),
  )
  z3 = z3_m.reshape(b_dim, c_z, n_i, n_j)
  a_next = a_m.reshape(b_dim, c_z, n_i, n_j)
  b_next = b_m.reshape(b_dim, c_z, n_i, n_j)
  go_next = go_m.reshape(b_dim, c_z, n_i, n_j)
  return (z3, a_next, b_next, go_next), (t_in, z1, go_in, z3, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3)


def _cross_mega_b_vjp_bwd(meta, res, ct):
  eps, bm, interpret = meta
  t_in, z1, go_in, z3, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3 = res
  dz3_res, da_next, db_next, dgo_next = ct
  b_dim, c_z, n_i, n_j = z1.shape
  d_ff = w3.shape[0]
  m = n_i * n_j
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"))
  dt_m, dz1_m, dgo2_m, dso2, dbo2, dwo2_t, dst, dbt, dw12_t, dw3_t, dsi3, dbi3, dwp3_t = pl.pallas_call(
      functools.partial(_cross_mega_b_bwd_kernel, eps=eps, c_z=c_z, d_ff=d_ff, bm=bm),
      out_shape=[
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((2 * d_ff, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, d_ff), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((5 * c_z, c_z), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_cross_mega_b_bwd",
  )(
      t_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      z1.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      go_in.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      z3.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      mask_8m,
      da_next.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      db_next.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      dgo_next.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      dz3_res.astype(jnp.bfloat16).reshape(b_dim, c_z, m),
      so2.astype(jnp.float32).reshape(c_z, 1),
      bo2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo2.astype(jnp.bfloat16), 0, 1),
      st.astype(jnp.float32).reshape(c_z, 1),
      bt.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w12.astype(jnp.bfloat16), 0, 1),
      jnp.swapaxes(w3.astype(jnp.bfloat16), 0, 1),
      si3.astype(jnp.float32).reshape(c_z, 1),
      bi3.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wp3.astype(jnp.bfloat16), 0, 1),
  )
  return (
      dt_m.reshape(b_dim, c_z, n_i, n_j),
      dz1_m.reshape(b_dim, c_z, n_i, n_j).astype(z1.dtype),
      dgo2_m.reshape(b_dim, c_z, n_i, n_j),
      None,
      dso2.reshape(c_z).astype(so2.dtype),
      dbo2.reshape(c_z).astype(bo2.dtype),
      jnp.swapaxes(dwo2_t, 0, 1).astype(wo2.dtype),
      dst.reshape(c_z).astype(st.dtype),
      dbt.reshape(c_z).astype(bt.dtype),
      jnp.swapaxes(dw12_t, 0, 1).astype(w12.dtype),
      jnp.swapaxes(dw3_t, 0, 1).astype(w3.dtype),
      dsi3.reshape(c_z).astype(si3.dtype),
      dbi3.reshape(c_z).astype(bi3.dtype),
      jnp.swapaxes(dwp3_t, 0, 1).astype(wp3.dtype),
  )


_cross_mega_b_vjp.defvjp(_cross_mega_b_vjp_fwd, _cross_mega_b_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(10,))
def _final_mega_b_vjp(t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3, meta):
  out, _ = _final_mega_b_vjp_fwd(t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3, meta)
  return out


def _final_mega_b_vjp_fwd(t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3, meta):
  eps, bm, interpret = meta
  z3 = _run_final_mega_b_fwd(t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3, eps=eps, bm=bm, interpret=interpret)
  return z3, (t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3)


def _final_mega_b_vjp_bwd(meta, res, dz3):
  eps, bm, interpret = meta
  t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3 = res
  dt, dz1, dgo2, dso2, dbo2, dwo2, dst, dbt, dw12, dw3 = _run_final_mega_b_bwd(
      t_in, z1, go_in, dz3, so2, bo2, wo2, st, bt, w12, w3, eps=eps, bm=bm, interpret=interpret
  )
  return dt, dz1.astype(z1.dtype), dgo2, dso2, dbo2, dwo2, dst, dbt, dw12, dw3


_final_mega_b_vjp.defvjp(_final_mega_b_vjp_fwd, _final_mega_b_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(16,))
def _layer_bcnn_compact_single_device(
    z0,
    mask_8m,
    si1,
    bi1,
    wp1,
    so1,
    bo1,
    wo1,
    si2,
    bi2,
    wp2,
    so2,
    bo2,
    wo2,
    st,
    bt,
    meta,
    w12,
    w3,
):
  out, _ = _layer_bcnn_compact_fwd(
      z0, mask_8m, si1, bi1, wp1, so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, meta, w12, w3
  )
  return out


def _layer_bcnn_compact_fwd(
    z0,
    mask_8m,
    si1,
    bi1,
    wp1,
    so1,
    bo1,
    wo1,
    si2,
    bi2,
    wp2,
    so2,
    bo2,
    wo2,
    st,
    bt,
    meta,
    w12,
    w3,
):
  eps, bm, interpret, fp32_einsum, matmul_precision, z0_dtype = meta
  a_out, b_out, go_out = _run_cm_prologue_fwd(z0, mask_8m, si1, bi1, wp1, eps=eps, bm=bm, interpret=interpret)
  t_out = _tri_einsum_fwd(a_out, b_out, direction="outgoing", fp32_einsum=fp32_einsum, matmul_precision=matmul_precision)
  z1, a_in, b_in, go_in = _run_mega_a_fwd(
      t_out, z0, go_out, mask_8m, so1, bo1, wo1, si2, bi2, wp2, eps=eps, bm=bm, interpret=interpret
  )
  t_in = _tri_einsum_fwd(a_in, b_in, direction="incoming", fp32_einsum=fp32_einsum, matmul_precision=matmul_precision)
  z3 = _run_final_mega_b_fwd(t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3, eps=eps, bm=bm, interpret=interpret)
  # Save ONLY 4 activations (z0, t_out, z1, t_in) -> 6.75 GiB * B across 48 layers!
  res = (
      z0.astype(jnp.bfloat16),
      t_out,
      z1,
      t_in,
      mask_8m,
      si1,
      bi1,
      wp1,
      so1,
      bo1,
      wo1,
      si2,
      bi2,
      wp2,
      so2,
      bo2,
      wo2,
      st,
      bt,
      w12,
      w3,
  )
  return z3.astype(z0_dtype), res


def _layer_bcnn_compact_bwd(meta, res, dz3):
  eps, bm, interpret, fp32_einsum, matmul_precision, z0_dtype = meta
  (
      z0,
      t_out,
      z1,
      t_in,
      mask_8m,
      si1,
      bi1,
      wp1,
      so1,
      bo1,
      wo1,
      si2,
      bi2,
      wp2,
      so2,
      bo2,
      wo2,
      st,
      bt,
      w12,
      w3,
  ) = res
  a_in, b_in, go_in = _run_cm_prologue_fwd(z1, mask_8m, si2, bi2, wp2, eps=eps, bm=bm, interpret=interpret)
  a_out, b_out, go_out = _run_cm_prologue_fwd(z0, mask_8m, si1, bi1, wp1, eps=eps, bm=bm, interpret=interpret)

  dt_in, dz1_res, dgo_in, dso2, dbo2, dwo2, dst, dbt, dw12, dw3 = _run_final_mega_b_bwd(
      t_in, z1, go_in, dz3, so2, bo2, wo2, st, bt, w12, w3, eps=eps, bm=bm, interpret=interpret
  )
  da_in, db_in = _tri_einsum_bwd(
      dt_in, a_in, b_in, direction="incoming", fp32_einsum=fp32_einsum, matmul_precision=matmul_precision
  )
  dt_out, dz0_res, dgo_out, dso1, dbo1, dwo1, dsi2, dbi2, dwp2 = _run_mega_a_bwd(
      t_out,
      go_out,
      z1,
      mask_8m,
      da_in,
      db_in,
      dgo_in,
      dz1_res,
      so1,
      bo1,
      wo1,
      si2,
      bi2,
      wp2,
      eps=eps,
      bm=bm,
      interpret=interpret,
  )
  da_out, db_out = _tri_einsum_bwd(
      dt_out, a_out, b_out, direction="outgoing", fp32_einsum=fp32_einsum, matmul_precision=matmul_precision
  )
  dz0, dsi1, dbi1, dwp1 = _run_cm_prologue_bwd(
      z0, mask_8m, da_out, db_out, dgo_out, dz0_res, si1, bi1, wp1, eps=eps, bm=bm, interpret=interpret
  )
  return (
      dz0.astype(z0_dtype),
      None,
      dsi1,
      dbi1,
      dwp1,
      dso1,
      dbo1,
      dwo1,
      dsi2,
      dbi2,
      dwp2,
      dso2,
      dbo2,
      dwo2,
      dst,
      dbt,
      dw12,
      dw3,
  )


_layer_bcnn_compact_single_device.defvjp(_layer_bcnn_compact_fwd, _layer_bcnn_compact_bwd)


def build_channel_major_mask_8m(decoder_segment_ids: jax.Array | None, batch_size: int, seq_len: int) -> jax.Array:
  """Builds `[B, 8, N*N]` bfloat16 pair mask broadcast once before Layer 0."""
  m = seq_len * seq_len
  if decoder_segment_ids is None:
    return jnp.ones((batch_size, 8, m), dtype=jnp.bfloat16)
  tok = decoder_segment_ids != 0
  pair = (tok[:, :, None] & tok[:, None, :]).astype(jnp.bfloat16)
  return jnp.broadcast_to(pair.reshape(batch_size, 1, m), (batch_size, 8, m))


def _run_cm_prologue_m_fwd(z_m, mask_8m, s_in, b_in, w_pgg, *, eps: float, bm: int, interpret: bool):
  b_dim, c_z, m = z_m.shape
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  out_shape = jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16)
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  return pl.pallas_call(
      functools.partial(_cm_prologue_fwd_kernel, eps=eps, c_z=c_z, bm=bm),
      out_shape=[out_shape, out_shape, out_shape],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[act_spec, act_spec, act_spec],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_cm_prologue_fwd",
  )(
      z_m,
      mask_8m,
      s_in.astype(jnp.float32).reshape(c_z, 1),
      b_in.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w_pgg.astype(jnp.bfloat16), 0, 1),
  )


def _run_cm_prologue_m_bwd(
    z_m, mask_8m, da_m, db_m, dgo_m, dz_res_m, s_in, b_in, w_pgg, *, eps: float, bm: int, interpret: bool
):
  b_dim, c_z, m = z_m.shape
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"))
  dz_m, dsi, dbi, dwp_t = pl.pallas_call(
      functools.partial(_cm_prologue_bwd_kernel, eps=eps, c_z=c_z, bm=bm),
      out_shape=[
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((5 * c_z, c_z), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_cm_prologue_bwd",
  )(
      z_m,
      mask_8m,
      da_m,
      db_m,
      dgo_m,
      dz_res_m,
      s_in.astype(jnp.float32).reshape(c_z, 1),
      b_in.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w_pgg.astype(jnp.bfloat16), 0, 1),
  )
  return (
      dz_m,
      dsi.reshape(c_z).astype(s_in.dtype),
      dbi.reshape(c_z).astype(b_in.dtype),
      jnp.swapaxes(dwp_t, 0, 1).astype(w_pgg.dtype),
  )


def _run_mega_a_m_fwd(
    t_out_m, z0_m, go_out_m, mask_8m, so1, bo1, wo1, si2, bi2, wp2, *, eps: float, bm: int, interpret: bool
):
  b_dim, c_z, m = z0_m.shape
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  out_shape = jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16)
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  return pl.pallas_call(
      functools.partial(_mega_a_fwd_kernel, eps=eps, c_z=c_z, bm=bm),
      out_shape=[out_shape, out_shape, out_shape, out_shape],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[act_spec, act_spec, act_spec, act_spec],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_mega_a_fwd",
  )(
      t_out_m,
      z0_m,
      go_out_m,
      mask_8m,
      so1.astype(jnp.float32).reshape(c_z, 1),
      bo1.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo1.astype(jnp.bfloat16), 0, 1),
      si2.astype(jnp.float32).reshape(c_z, 1),
      bi2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wp2.astype(jnp.bfloat16), 0, 1),
  )


def _run_mega_a_m_bwd(
    t_out_m,
    go_out_m,
    z1_m,
    mask_8m,
    da_in_m,
    db_in_m,
    dgo_in_m,
    dz1_res_m,
    so1,
    bo1,
    wo1,
    si2,
    bi2,
    wp2,
    *,
    eps: float,
    bm: int,
    interpret: bool,
):
  b_dim, c_z, m = z1_m.shape
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"))
  dt_m, dz0_m, dgo1_m, dso1, dbo1, dwo1_t, dsi2, dbi2, dwp2_t = pl.pallas_call(
      functools.partial(_mega_a_bwd_kernel, eps=eps, c_z=c_z, bm=bm),
      out_shape=[
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((5 * c_z, c_z), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_mega_a_bwd",
  )(
      t_out_m,
      go_out_m,
      z1_m,
      mask_8m,
      da_in_m,
      db_in_m,
      dgo_in_m,
      dz1_res_m,
      so1.astype(jnp.float32).reshape(c_z, 1),
      bo1.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo1.astype(jnp.bfloat16), 0, 1),
      si2.astype(jnp.float32).reshape(c_z, 1),
      bi2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wp2.astype(jnp.bfloat16), 0, 1),
  )
  return (
      dt_m,
      dz0_m,
      dgo1_m,
      dso1.reshape(c_z).astype(so1.dtype),
      dbo1.reshape(c_z).astype(bo1.dtype),
      jnp.swapaxes(dwo1_t, 0, 1).astype(wo1.dtype),
      dsi2.reshape(c_z).astype(si2.dtype),
      dbi2.reshape(c_z).astype(bi2.dtype),
      jnp.swapaxes(dwp2_t, 0, 1).astype(wp2.dtype),
  )


def _run_final_mega_b_m_fwd(
    t_in_m, z1_m, go_in_m, so2, bo2, wo2, st, bt, w12, w3, *, eps: float, bm: int, interpret: bool
):
  b_dim, c_z, m = z1_m.shape
  d_ff = w3.shape[0]
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  return pl.pallas_call(
      functools.partial(_final_mega_b_fwd_kernel, eps=eps, c_z=c_z, d_ff=d_ff, bm=bm),
      out_shape=jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
          ],
          out_specs=act_spec,
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_final_mega_b_fwd",
  )(
      t_in_m,
      z1_m,
      go_in_m,
      so2.astype(jnp.float32).reshape(c_z, 1),
      bo2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo2.astype(jnp.bfloat16), 0, 1),
      st.astype(jnp.float32).reshape(c_z, 1),
      bt.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w12.astype(jnp.bfloat16), 0, 1),
      jnp.swapaxes(w3.astype(jnp.bfloat16), 0, 1),
  )


def _run_final_mega_b_m_bwd(
    t_in_m, z1_m, go_in_m, dz3_m, so2, bo2, wo2, st, bt, w12, w3, *, eps: float, bm: int, interpret: bool
):
  b_dim, c_z, m = z1_m.shape
  d_ff = w3.shape[0]
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"))
  dt_m, dz1_m, dgo2_m, dso2, dbo2, dwo2_t, dst, dbt, dw12_t, dw3_t = pl.pallas_call(
      functools.partial(_final_mega_b_bwd_kernel, eps=eps, c_z=c_z, d_ff=d_ff, bm=bm),
      out_shape=[
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((2 * d_ff, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, d_ff), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
          ],
          out_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
          ],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_final_mega_b_bwd",
  )(
      t_in_m,
      z1_m,
      go_in_m,
      dz3_m,
      so2.astype(jnp.float32).reshape(c_z, 1),
      bo2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo2.astype(jnp.bfloat16), 0, 1),
      st.astype(jnp.float32).reshape(c_z, 1),
      bt.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w12.astype(jnp.bfloat16), 0, 1),
      jnp.swapaxes(w3.astype(jnp.bfloat16), 0, 1),
  )
  return (
      dt_m,
      dz1_m,
      dgo2_m,
      dso2.reshape(c_z).astype(so2.dtype),
      dbo2.reshape(c_z).astype(bo2.dtype),
      jnp.swapaxes(dwo2_t, 0, 1).astype(wo2.dtype),
      dst.reshape(c_z).astype(st.dtype),
      dbt.reshape(c_z).astype(bt.dtype),
      jnp.swapaxes(dw12_t, 0, 1).astype(w12.dtype),
      jnp.swapaxes(dw3_t, 0, 1).astype(w3.dtype),
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _cm_prologue_m_vjp(z_m, mask_8m, s_in, b_in, w_pgg, meta):
  out, _ = _cm_prologue_m_vjp_fwd(z_m, mask_8m, s_in, b_in, w_pgg, meta)
  return out


def _cm_prologue_m_vjp_fwd(z_m, mask_8m, s_in, b_in, w_pgg, meta):
  eps, bm, interpret = meta
  return _run_cm_prologue_m_fwd(z_m, mask_8m, s_in, b_in, w_pgg, eps=eps, bm=bm, interpret=interpret), (
      z_m,
      mask_8m,
      s_in,
      b_in,
      w_pgg,
  )


def _cm_prologue_m_vjp_bwd(meta, res, ct):
  eps, bm, interpret = meta
  z_m, mask_8m, s_in, b_in, w_pgg = res
  da_m, db_m, dgo_m = ct
  dz_m, dsi, dbi, dwp = _run_cm_prologue_m_bwd(
      z_m,
      mask_8m,
      da_m,
      db_m,
      dgo_m,
      jnp.zeros_like(z_m, dtype=jnp.bfloat16),
      s_in,
      b_in,
      w_pgg,
      eps=eps,
      bm=bm,
      interpret=interpret,
  )
  return dz_m, None, dsi, dbi, dwp


_cm_prologue_m_vjp.defvjp(_cm_prologue_m_vjp_fwd, _cm_prologue_m_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(10,))
def _mega_a_m_vjp(t_out_m, z0_m, go_out_m, mask_8m, so1, bo1, wo1, si2, bi2, wp2, meta):
  out, _ = _mega_a_m_vjp_fwd(t_out_m, z0_m, go_out_m, mask_8m, so1, bo1, wo1, si2, bi2, wp2, meta)
  return out


def _mega_a_m_vjp_fwd(t_out_m, z0_m, go_out_m, mask_8m, so1, bo1, wo1, si2, bi2, wp2, meta):
  eps, bm, interpret = meta
  z1_m, a_in_m, b_in_m, go_in_m = _run_mega_a_m_fwd(
      t_out_m, z0_m, go_out_m, mask_8m, so1, bo1, wo1, si2, bi2, wp2, eps=eps, bm=bm, interpret=interpret
  )
  return (z1_m, a_in_m, b_in_m, go_in_m), (t_out_m, go_out_m, z1_m, mask_8m, so1, bo1, wo1, si2, bi2, wp2)


def _mega_a_m_vjp_bwd(meta, res, ct):
  eps, bm, interpret = meta
  t_out_m, go_out_m, z1_m, mask_8m, so1, bo1, wo1, si2, bi2, wp2 = res
  dz1_res_m, da_in_m, db_in_m, dgo_in_m = ct
  dt_m, dz0_m, dgo1_m, dso1, dbo1, dwo1, dsi2, dbi2, dwp2 = _run_mega_a_m_bwd(
      t_out_m,
      go_out_m,
      z1_m,
      mask_8m,
      da_in_m,
      db_in_m,
      dgo_in_m,
      dz1_res_m,
      so1,
      bo1,
      wo1,
      si2,
      bi2,
      wp2,
      eps=eps,
      bm=bm,
      interpret=interpret,
  )
  return dt_m, dz0_m, dgo1_m, None, dso1, dbo1, dwo1, dsi2, dbi2, dwp2


_mega_a_m_vjp.defvjp(_mega_a_m_vjp_fwd, _mega_a_m_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(14,))
def _cross_mega_b_m_vjp(t_in_m, z1_m, go_in_m, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3, meta):
  out, _ = _cross_mega_b_m_vjp_fwd(t_in_m, z1_m, go_in_m, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3, meta)
  return out


def _cross_mega_b_m_vjp_fwd(t_in_m, z1_m, go_in_m, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3, meta):
  eps, bm, interpret = meta
  b_dim, c_z, m = z1_m.shape
  d_ff = w3.shape[0]
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  out_shape = jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16)
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
  z3_m, a_m, b_m, go_m = pl.pallas_call(
      functools.partial(_cross_mega_b_fwd_kernel, eps=eps, c_z=c_z, d_ff=d_ff, bm=bm),
      out_shape=[out_shape, out_shape, out_shape, out_shape],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[act_spec, act_spec, act_spec, act_spec],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_cross_mega_b_fwd",
  )(
      t_in_m,
      z1_m,
      go_in_m,
      mask_8m,
      so2.astype(jnp.float32).reshape(c_z, 1),
      bo2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo2.astype(jnp.bfloat16), 0, 1),
      st.astype(jnp.float32).reshape(c_z, 1),
      bt.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w12.astype(jnp.bfloat16), 0, 1),
      jnp.swapaxes(w3.astype(jnp.bfloat16), 0, 1),
      si3.astype(jnp.float32).reshape(c_z, 1),
      bi3.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wp3.astype(jnp.bfloat16), 0, 1),
  )
  return (z3_m, a_m, b_m, go_m), (t_in_m, z1_m, go_in_m, z3_m, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3)


def _cross_mega_b_m_vjp_bwd(meta, res, ct):
  eps, bm, interpret = meta
  t_in_m, z1_m, go_in_m, z3_m, mask_8m, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3 = res
  dz3_res_m, da_next_m, db_next_m, dgo_next_m = ct
  b_dim, c_z, m = z1_m.shape
  d_ff = w3.shape[0]
  grid = (b_dim, m // bm)
  act_spec = pl.BlockSpec((1, c_z, bm), lambda bi, mi: (bi, 0, mi))
  cp = None if interpret else pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"))
  dt_m, dz1_m, dgo2_m, dso2, dbo2, dwo2_t, dst, dbt, dw12_t, dw3_t, dsi3, dbi3, dwp3_t = pl.pallas_call(
      functools.partial(_cross_mega_b_bwd_kernel, eps=eps, c_z=c_z, d_ff=d_ff, bm=bm),
      out_shape=[
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((b_dim, c_z, m), jnp.bfloat16),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((2 * d_ff, c_z), jnp.float32),
          jax.ShapeDtypeStruct((c_z, d_ff), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((c_z, 1), jnp.float32),
          jax.ShapeDtypeStruct((5 * c_z, c_z), jnp.float32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((1, 8, bm), lambda bi, mi: (bi, 0, mi)),
              act_spec,
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
          out_specs=[
              act_spec,
              act_spec,
              act_spec,
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((2 * d_ff, c_z), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, d_ff), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((c_z, 1), lambda bi, mi: (0, 0)),
              pl.BlockSpec((5 * c_z, c_z), lambda bi, mi: (0, 0)),
          ],
      ),
      compiler_params=cp,
      interpret=interpret,
      name="pairmixer_cross_mega_b_bwd",
  )(
      t_in_m,
      z1_m,
      go_in_m,
      z3_m,
      mask_8m,
      da_next_m,
      db_next_m,
      dgo_next_m,
      dz3_res_m,
      so2.astype(jnp.float32).reshape(c_z, 1),
      bo2.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wo2.astype(jnp.bfloat16), 0, 1),
      st.astype(jnp.float32).reshape(c_z, 1),
      bt.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(w12.astype(jnp.bfloat16), 0, 1),
      jnp.swapaxes(w3.astype(jnp.bfloat16), 0, 1),
      si3.astype(jnp.float32).reshape(c_z, 1),
      bi3.astype(jnp.float32).reshape(c_z, 1),
      jnp.swapaxes(wp3.astype(jnp.bfloat16), 0, 1),
  )
  return (
      dt_m,
      dz1_m,
      dgo2_m,
      None,
      dso2.reshape(c_z).astype(so2.dtype),
      dbo2.reshape(c_z).astype(bo2.dtype),
      jnp.swapaxes(dwo2_t, 0, 1).astype(wo2.dtype),
      dst.reshape(c_z).astype(st.dtype),
      dbt.reshape(c_z).astype(bt.dtype),
      jnp.swapaxes(dw12_t, 0, 1).astype(w12.dtype),
      jnp.swapaxes(dw3_t, 0, 1).astype(w3.dtype),
      dsi3.reshape(c_z).astype(si3.dtype),
      dbi3.reshape(c_z).astype(bi3.dtype),
      jnp.swapaxes(dwp3_t, 0, 1).astype(wp3.dtype),
  )


_cross_mega_b_m_vjp.defvjp(_cross_mega_b_m_vjp_fwd, _cross_mega_b_m_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(10,))
def _final_mega_b_m_vjp(t_in_m, z1_m, go_in_m, so2, bo2, wo2, st, bt, w12, w3, meta):
  out, _ = _final_mega_b_m_vjp_fwd(t_in_m, z1_m, go_in_m, so2, bo2, wo2, st, bt, w12, w3, meta)
  return out


def _final_mega_b_m_vjp_fwd(t_in_m, z1_m, go_in_m, so2, bo2, wo2, st, bt, w12, w3, meta):
  eps, bm, interpret = meta
  return _run_final_mega_b_m_fwd(
      t_in_m, z1_m, go_in_m, so2, bo2, wo2, st, bt, w12, w3, eps=eps, bm=bm, interpret=interpret
  ), (
      t_in_m,
      z1_m,
      go_in_m,
      so2,
      bo2,
      wo2,
      st,
      bt,
      w12,
      w3,
  )


def _final_mega_b_m_vjp_bwd(meta, res, dz3_m):
  eps, bm, interpret = meta
  t_in_m, z1_m, go_in_m, so2, bo2, wo2, st, bt, w12, w3 = res
  return _run_final_mega_b_m_bwd(
      t_in_m, z1_m, go_in_m, dz3_m, so2, bo2, wo2, st, bt, w12, w3, eps=eps, bm=bm, interpret=interpret
  )


_final_mega_b_m_vjp.defvjp(_final_mega_b_m_vjp_fwd, _final_mega_b_m_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(16,))
def _layer_bcm_compact_single_device(
    z0_m, mask_8m, si1, bi1, wp1, so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, meta, w12, w3
):
  out, _ = _layer_bcm_compact_fwd(
      z0_m, mask_8m, si1, bi1, wp1, so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, meta, w12, w3
  )
  return out


def _layer_bcm_compact_fwd(
    z0_m, mask_8m, si1, bi1, wp1, so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, meta, w12, w3
):
  eps, bm, interpret, n_i, n_j, save_contractions = meta
  a_out_m, b_out_m, go_out_m = _run_cm_prologue_m_fwd(z0_m, mask_8m, si1, bi1, wp1, eps=eps, bm=bm, interpret=interpret)
  t_out_m = _run_tri_bcm_fwd(a_out_m, b_out_m, n_i=n_i, n_j=n_j, direction="outgoing", interpret=interpret)
  z1_m, a_in_m, b_in_m, go_in_m = _run_mega_a_m_fwd(
      t_out_m, z0_m, go_out_m, mask_8m, so1, bo1, wo1, si2, bi2, wp2, eps=eps, bm=bm, interpret=interpret
  )
  t_in_m = _run_tri_bcm_fwd(a_in_m, b_in_m, n_i=n_i, n_j=n_j, direction="incoming", interpret=interpret)
  z3_m = _run_final_mega_b_m_fwd(
      t_in_m, z1_m, go_in_m, so2, bo2, wo2, st, bt, w12, w3, eps=eps, bm=bm, interpret=interpret
  )
  if save_contractions:
    res = (
        z0_m,
        t_out_m,
        z1_m,
        t_in_m,
        mask_8m,
        si1,
        bi1,
        wp1,
        so1,
        bo1,
        wo1,
        si2,
        bi2,
        wp2,
        so2,
        bo2,
        wo2,
        st,
        bt,
        w12,
        w3,
    )
  else:
    res = (z0_m, z1_m, mask_8m, si1, bi1, wp1, so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, w12, w3)
  return z3_m, res


def _layer_bcm_compact_bwd(meta, res, dz3_m):
  eps, bm, interpret, n_i, n_j, save_contractions = meta
  if save_contractions:
    z0_m, t_out_m, z1_m, t_in_m, mask_8m, si1, bi1, wp1, so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, w12, w3 = (
        res
    )
  else:
    z0_m, z1_m, mask_8m, si1, bi1, wp1, so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, w12, w3 = res
  a_in_m, b_in_m, go_in_m = _run_cm_prologue_m_fwd(z1_m, mask_8m, si2, bi2, wp2, eps=eps, bm=bm, interpret=interpret)
  a_out_m, b_out_m, go_out_m = _run_cm_prologue_m_fwd(z0_m, mask_8m, si1, bi1, wp1, eps=eps, bm=bm, interpret=interpret)
  if not save_contractions:
    t_in_m = _run_tri_bcm_fwd(a_in_m, b_in_m, n_i=n_i, n_j=n_j, direction="incoming", interpret=interpret)
    t_out_m = _run_tri_bcm_fwd(a_out_m, b_out_m, n_i=n_i, n_j=n_j, direction="outgoing", interpret=interpret)

  dt_in_m, dz1_res_m, dgo_in_m, dso2, dbo2, dwo2, dst, dbt, dw12, dw3 = _run_final_mega_b_m_bwd(
      t_in_m, z1_m, go_in_m, dz3_m, so2, bo2, wo2, st, bt, w12, w3, eps=eps, bm=bm, interpret=interpret
  )
  da_in_m, db_in_m = _run_tri_bcm_bwd(
      dt_in_m, a_in_m, b_in_m, n_i=n_i, n_j=n_j, direction="incoming", interpret=interpret
  )
  dt_out_m, dz0_res_m, dgo_out_m, dso1, dbo1, dwo1, dsi2, dbi2, dwp2 = _run_mega_a_m_bwd(
      t_out_m,
      go_out_m,
      z1_m,
      mask_8m,
      da_in_m,
      db_in_m,
      dgo_in_m,
      dz1_res_m,
      so1,
      bo1,
      wo1,
      si2,
      bi2,
      wp2,
      eps=eps,
      bm=bm,
      interpret=interpret,
  )
  da_out_m, db_out_m = _run_tri_bcm_bwd(
      dt_out_m, a_out_m, b_out_m, n_i=n_i, n_j=n_j, direction="outgoing", interpret=interpret
  )
  dz0_m, dsi1, dbi1, dwp1 = _run_cm_prologue_m_bwd(
      z0_m, mask_8m, da_out_m, db_out_m, dgo_out_m, dz0_res_m, si1, bi1, wp1, eps=eps, bm=bm, interpret=interpret
  )
  return dz0_m, None, dsi1, dbi1, dwp1, dso1, dbo1, dwo1, dsi2, dbi2, dwp2, dso2, dbo2, dwo2, dst, dbt, dw12, dw3


_layer_bcm_compact_single_device.defvjp(_layer_bcm_compact_fwd, _layer_bcm_compact_bwd)


def pallas_pairmixer_layer_bcnn(
    z: jax.Array,
    mask_8m: jax.Array,
    params: tuple[jax.Array, ...],
    *,
    mesh: Mesh | Any = None,
    fp32_einsum: bool = False,
    matmul_precision: str = "default",
    eps: float = 1e-5,
    block_m: int = 4096,
    save_contractions: bool = True,
    interpret: bool | None = None,
) -> jax.Array:
  """Executes one full Pairmixer layer on `[B, C, N, N]` (saving 4 tensors or 2 tensors)."""
  resolved_interpret = _resolve_interpret(interpret, mesh)
  b_global, c_z, n_i, n_j = z.shape
  m = n_i * n_j
  d_ff = params[15].shape[0]
  bm = resolve_channel_major_block_m(m, c_z, d_ff, block_m)
  use_bcm = _can_use_tri_bcm(c_z, n_i, n_j, fp32_einsum, resolved_interpret)

  if use_bcm:
    meta_bcm = (eps, bm, resolved_interpret, n_i, n_j, save_contractions)
    z_m_in = z.astype(jnp.bfloat16).reshape(b_global, c_z, m)
    if mesh is None:
      out_m = _layer_bcm_compact_single_device(z_m_in, mask_8m, *params[:14], meta_bcm, params[14], params[15])
      return out_m.reshape(b_global, c_z, n_i, n_j).astype(z.dtype)
    batch_spec = _get_batch_partition_spec(mesh, b_global)
    zm_pspec = PartitionSpec(batch_spec, None, None)
    m_pspec = PartitionSpec(batch_spec, None, None)
    p_pspecs = tuple(PartitionSpec(*([None] * p.ndim)) for p in params)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(zm_pspec, m_pspec, p_pspecs),
        out_specs=zm_pspec,
        check_vma=False,
    )
    def _mapped_bcm(zm_l, m_l, p_l):
      return _layer_bcm_compact_single_device(zm_l, m_l, *p_l[:14], meta_bcm, p_l[14], p_l[15])

    return _mapped_bcm(z_m_in, mask_8m, params).reshape(b_global, c_z, n_i, n_j).astype(z.dtype)

  meta = (eps, bm, resolved_interpret, fp32_einsum, matmul_precision, z.dtype)

  if mesh is None:
    return _layer_bcnn_compact_single_device(z, mask_8m, *params[:14], meta, params[14], params[15])

  batch_spec = _get_batch_partition_spec(mesh, z.shape[0])
  z_pspec = PartitionSpec(batch_spec, None, None, None)
  m_pspec = PartitionSpec(batch_spec, None, None)
  p_pspecs = tuple(PartitionSpec(*([None] * p.ndim)) for p in params)

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(z_pspec, m_pspec, p_pspecs),
      out_specs=z_pspec,
      check_vma=False,
  )
  def _mapped(z_l, m_l, p_l):
    return _layer_bcnn_compact_single_device(z_l, m_l, *p_l[:14], meta, p_l[14], p_l[15])

  return _mapped(z, mask_8m, params)


def pallas_pairmixer_cross_layer_trunk_bcnn(
    z0: jax.Array,
    mask_8m: jax.Array,
    all_layer_params: tuple[tuple[jax.Array, ...], ...],
    *,
    mesh: Mesh | Any = None,
    fp32_einsum: bool = False,
    matmul_precision: str = "default",
    eps: float = 1e-5,
    block_m: int = 4096,
    interpret: bool | None = None,
) -> jax.Array:
  """Executes the full L-layer Pairmixer trunk in `[B, C, M]` / `[B, C, N, N]` with Cross-Layer MegaKernel B."""
  resolved_interpret = _resolve_interpret(interpret, mesh)
  b_global, c_z, n_i, n_j = z0.shape
  m = n_i * n_j
  d_ff = all_layer_params[0][15].shape[0]
  bm = resolve_channel_major_block_m(m, c_z, d_ff, block_m)
  kmeta = (eps, bm, resolved_interpret)
  num_layers = len(all_layer_params)
  use_bcm = _can_use_tri_bcm(c_z, n_i, n_j, fp32_einsum, resolved_interpret)

  if use_bcm:
    tri_out_meta = (n_i, n_j, "outgoing", resolved_interpret)
    tri_in_meta = (n_i, n_j, "incoming", resolved_interpret)

    def _single_device_trunk_bcm(zm_cur, m_cur, layers_p):
      si1_0, bi1_0, wp1_0 = layers_p[0][0], layers_p[0][1], layers_p[0][2]
      a_out_m, b_out_m, go_out_m = _cm_prologue_m_vjp(zm_cur, m_cur, si1_0, bi1_0, wp1_0, kmeta)
      for idx in range(num_layers):
        lp = layers_p[idx]
        so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, w12, w3 = lp[3:16]
        t_out_m = _tri_bcm_vjp(a_out_m, b_out_m, tri_out_meta)
        z1_m, a_in_m, b_in_m, go_in_m = _mega_a_m_vjp(
            t_out_m, zm_cur, go_out_m, m_cur, so1, bo1, wo1, si2, bi2, wp2, kmeta
        )
        t_in_m = _tri_bcm_vjp(a_in_m, b_in_m, tri_in_meta)
        if idx < num_layers - 1:
          si3, bi3, wp3 = layers_p[idx + 1][0], layers_p[idx + 1][1], layers_p[idx + 1][2]
          zm_cur, a_out_m, b_out_m, go_out_m = _cross_mega_b_m_vjp(
              t_in_m, z1_m, go_in_m, m_cur, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3, kmeta
          )
        else:
          zm_cur = _final_mega_b_m_vjp(t_in_m, z1_m, go_in_m, so2, bo2, wo2, st, bt, w12, w3, kmeta)
      return zm_cur

    z0_m = z0.astype(jnp.bfloat16).reshape(b_global, c_z, m)
    if mesh is None:
      return _single_device_trunk_bcm(z0_m, mask_8m, all_layer_params).reshape(b_global, c_z, n_i, n_j).astype(z0.dtype)

    batch_spec = _get_batch_partition_spec(mesh, b_global)
    zm_pspec = PartitionSpec(batch_spec, None, None)
    m_pspec = PartitionSpec(batch_spec, None, None)
    all_pspecs = tuple(tuple(PartitionSpec(*([None] * p.ndim)) for p in lp) for lp in all_layer_params)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(zm_pspec, m_pspec, all_pspecs),
        out_specs=zm_pspec,
        check_vma=False,
    )
    def _mapped_bcm(zm_l, m_l, lp_l):
      return _single_device_trunk_bcm(zm_l, m_l, lp_l)

    return _mapped_bcm(z0_m, mask_8m, all_layer_params).reshape(b_global, c_z, n_i, n_j).astype(z0.dtype)

  def _single_device_trunk(z_cur, m_cur, layers_p):
    si1_0, bi1_0, wp1_0 = layers_p[0][0], layers_p[0][1], layers_p[0][2]
    a_out, b_out, go_out = _cm_prologue_vjp(z_cur.astype(jnp.bfloat16), m_cur, si1_0, bi1_0, wp1_0, kmeta)
    for idx in range(num_layers):
      lp = layers_p[idx]
      so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, w12, w3 = lp[3:16]
      t_out = _tri_einsum_fwd(
          a_out, b_out, direction="outgoing", fp32_einsum=fp32_einsum, matmul_precision=matmul_precision
      )
      z1, a_in, b_in, go_in = _mega_a_vjp(
          t_out, z_cur.astype(jnp.bfloat16), go_out, m_cur, so1, bo1, wo1, si2, bi2, wp2, kmeta
      )
      t_in = _tri_einsum_fwd(a_in, b_in, direction="incoming", fp32_einsum=fp32_einsum, matmul_precision=matmul_precision)
      if idx < num_layers - 1:
        si3, bi3, wp3 = layers_p[idx + 1][0], layers_p[idx + 1][1], layers_p[idx + 1][2]
        z_cur, a_out, b_out, go_out = _cross_mega_b_vjp(
            t_in, z1, go_in, m_cur, so2, bo2, wo2, st, bt, w12, w3, si3, bi3, wp3, kmeta
        )
      else:
        z_cur = _final_mega_b_vjp(t_in, z1, go_in, so2, bo2, wo2, st, bt, w12, w3, kmeta)
    return z_cur.astype(z0.dtype)

  if mesh is None:
    return _single_device_trunk(z0, mask_8m, all_layer_params)

  batch_spec = _get_batch_partition_spec(mesh, z0.shape[0])
  z_pspec = PartitionSpec(batch_spec, None, None, None)
  m_pspec = PartitionSpec(batch_spec, None, None)
  all_pspecs = tuple(tuple(PartitionSpec(*([None] * p.ndim)) for p in lp) for lp in all_layer_params)

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(z_pspec, m_pspec, all_pspecs),
      out_specs=z_pspec,
      check_vma=False,
  )
  def _mapped(z_l, m_l, lp_l):
    return _single_device_trunk(z_l, m_l, lp_l)

  return _mapped(z0, mask_8m, all_layer_params)


# ---------------------------------------------------------------------------
# Regime 2: Systolic Channel-Major Trunk (`c_z >= 512`, `256 x 256` MXU)
# ---------------------------------------------------------------------------


def pallas_pairmixer_systolic_trunk_bcnn(
    z0: jax.Array,
    mask_8m: jax.Array,
    all_layer_params: tuple[tuple[jax.Array, ...], ...],
    *,
    mesh: Mesh | Any = None,
    fp32_einsum: bool = False,
    matmul_precision: str = "default",
    eps: float = 1e-5,
    remat_all: bool = False,
    interpret: bool | None = None,
) -> jax.Array:
  """Executes the L-layer Pairmixer trunk on `[B, C, M]` using native 256x256 systolic contractions (`c_z >= 512`)."""
  resolved_interpret = _resolve_interpret(interpret, mesh)
  b_global, c_z, n_i, n_j = z0.shape
  m = n_i * n_j
  d_ff = all_layer_params[0][15].shape[0]
  num_layers = len(all_layer_params)
  out_dt = jnp.float32 if fp32_einsum else jnp.bfloat16
  dot_out_dt = jnp.float32 if resolved_interpret else jnp.bfloat16
  tri_out_dt = jnp.float32 if (resolved_interpret or fp32_einsum) else jnp.bfloat16

  def _dot_in(x: jax.Array) -> jax.Array:
    x_bf16 = x.astype(jnp.bfloat16)
    return x_bf16.astype(jnp.float32) if resolved_interpret else x_bf16

  if mesh is None:
    b_shards = 1
  else:
    bspec = _get_batch_partition_spec(mesh, b_global)
    if bspec is None:
      b_shards = 1
    elif isinstance(bspec, tuple):
      b_shards = 1
      for ax in bspec:
        b_shards *= mesh.shape[ax]
    else:
      b_shards = mesh.shape[bspec]
  b_local = max(1, b_global // max(1, b_shards))

  # Auto-budget dot-checkpointed vs whole-layer-checkpointed layers against
  # physical TPU v6e PJRT reservable HBM (30.81 GiB total; target <= 28.5 GiB).
  param_state_bytes = num_layers * (12 * c_z * c_z + 3 * c_z * d_ff) * 16
  avail_bytes = max(0, int(28.5 * (1024**3)) - param_state_bytes)
  work_bytes = b_local * (12 * c_z + 2 * d_ff) * m * 2

  def _ln_mc(x_mc: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    xf = x_mc.astype(jnp.float32)
    mu = jnp.mean(xf, axis=-1, keepdims=True)
    rs = lax.rsqrt(jnp.mean(jnp.square(xf - mu), axis=-1, keepdims=True) + eps)
    return ((xf - mu) * rs * (scale.astype(jnp.float32) + 1.0) + bias.astype(jnp.float32)).astype(jnp.bfloat16)

  dot_policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable

  def _sub_tri_out(z_mc, mask_mc, si1, bi1, wp1, so1, bo1, wo1):
    b_cur = z_mc.shape[0]
    xn1 = _ln_mc(z_mc, si1, bi1)
    pgg1 = jnp.dot(_dot_in(xn1), _dot_in(wp1), precision=matmul_precision, preferred_element_type=dot_out_dt)
    pa, pb, ga, gb, go = jnp.split(pgg1.reshape(b_cur, m, 5, c_z), 5, axis=2)
    mf = mask_mc.astype(jnp.float32)
    a1 = (pa[:, :, 0, :].astype(jnp.float32) * jax.nn.sigmoid(ga[:, :, 0, :].astype(jnp.float32)) * mf).astype(
        jnp.bfloat16
    )
    b1 = (pb[:, :, 0, :].astype(jnp.float32) * jax.nn.sigmoid(gb[:, :, 0, :].astype(jnp.float32)) * mf).astype(
        jnp.bfloat16
    )
    go1 = jax.nn.sigmoid(go[:, :, 0, :].astype(jnp.float32)).astype(jnp.bfloat16)
    t1 = (
        jnp.einsum(
            "bikc,bjkc->bijc",
            _dot_in(a1).reshape(b_cur, n_i, n_j, c_z),
            _dot_in(b1).reshape(b_cur, n_i, n_j, c_z),
            precision=matmul_precision,
            preferred_element_type=tri_out_dt,
        )
        .astype(out_dt)
        .astype(jnp.bfloat16)
        .reshape(b_cur, m, c_z)
    )
    po1 = jnp.dot(
        _dot_in(_ln_mc(t1, so1, bo1)), _dot_in(wo1), precision=matmul_precision, preferred_element_type=dot_out_dt
    )
    return (z_mc.astype(jnp.float32) + po1.astype(jnp.float32) * go1.astype(jnp.float32)).astype(jnp.bfloat16)

  def _sub_tri_in(z_mc, mask_mc, si2, bi2, wp2, so2, bo2, wo2):
    b_cur = z_mc.shape[0]
    xn2 = _ln_mc(z_mc, si2, bi2)
    pgg2 = jnp.dot(_dot_in(xn2), _dot_in(wp2), precision=matmul_precision, preferred_element_type=dot_out_dt)
    pa, pb, ga, gb, go = jnp.split(pgg2.reshape(b_cur, m, 5, c_z), 5, axis=2)
    mf = mask_mc.astype(jnp.float32)
    a2 = (pa[:, :, 0, :].astype(jnp.float32) * jax.nn.sigmoid(ga[:, :, 0, :].astype(jnp.float32)) * mf).astype(
        jnp.bfloat16
    )
    b2 = (pb[:, :, 0, :].astype(jnp.float32) * jax.nn.sigmoid(gb[:, :, 0, :].astype(jnp.float32)) * mf).astype(
        jnp.bfloat16
    )
    go2 = jax.nn.sigmoid(go[:, :, 0, :].astype(jnp.float32)).astype(jnp.bfloat16)
    t2 = (
        jnp.einsum(
            "bkic,bkjc->bijc",
            _dot_in(a2).reshape(b_cur, n_i, n_j, c_z),
            _dot_in(b2).reshape(b_cur, n_i, n_j, c_z),
            precision=matmul_precision,
            preferred_element_type=tri_out_dt,
        )
        .astype(out_dt)
        .astype(jnp.bfloat16)
        .reshape(b_cur, m, c_z)
    )
    po2 = jnp.dot(
        _dot_in(_ln_mc(t2, so2, bo2)), _dot_in(wo2), precision=matmul_precision, preferred_element_type=dot_out_dt
    )
    return (z_mc.astype(jnp.float32) + po2.astype(jnp.float32) * go2.astype(jnp.float32)).astype(jnp.bfloat16)

  def _sub_ffn(z_mc, st, bt, w12, w3):
    xnt = _ln_mc(z_mc, st, bt)
    h12 = jnp.dot(_dot_in(xnt), _dot_in(w12), precision=matmul_precision, preferred_element_type=dot_out_dt)
    h1, h2 = jnp.split(h12, 2, axis=-1)
    h = (jax.nn.silu(h1.astype(jnp.float32)) * h2.astype(jnp.float32)).astype(jnp.bfloat16)
    mlp_out = jnp.dot(_dot_in(h), _dot_in(w3), precision=matmul_precision, preferred_element_type=dot_out_dt)
    return (z_mc.astype(jnp.float32) + mlp_out.astype(jnp.float32)).astype(jnp.bfloat16)

  dot_tri_out = jax.checkpoint(_sub_tri_out, policy=dot_policy)
  dot_tri_in = jax.checkpoint(_sub_tri_in, policy=dot_policy)
  dot_ffn = jax.checkpoint(_sub_ffn, policy=dot_policy)

  def _dotckpt_systolic_layer(z_mc, mask_mc, lp):
    si1, bi1, wp1, so1, bo1, wo1, si2, bi2, wp2, so2, bo2, wo2, st, bt, w12, w3 = lp
    z_mc = dot_tri_out(z_mc, mask_mc, si1, bi1, wp1, so1, bo1, wo1)
    z_mc = dot_tri_in(z_mc, mask_mc, si2, bi2, wp2, so2, bo2, wo2)
    return dot_ffn(z_mc, st, bt, w12, w3)

  fullckpt_layer = jax.checkpoint(_dotckpt_systolic_layer)

  # Each BF16 dot-checkpointed layer retains non-batched matmul outputs plus sublayer inputs
  # (~(16*c_z + 2*d_ff)*m*2B), fitting safely inside physical v6e PJRT free HBM (24.44 GiB).
  dot_layer_bytes = b_local * (16 * c_z + 2 * d_ff) * m * 2
  max_dot_layers = 0 if remat_all else min(num_layers, max(0, (avail_bytes - work_bytes) // max(1, dot_layer_bytes)))
  num_fullckpt = num_layers - max_dot_layers

  # Run under GSPMD rather than jax.shard_map so gradient_accumulation_steps accumulates
  # local gradients across microsteps in HBM without per-microstep synchronous psum barriers.
  z_cur_mc = jnp.swapaxes(z0.astype(jnp.bfloat16).reshape(b_global, c_z, m), 1, 2)
  mask_mc = jnp.swapaxes(mask_8m[:, 0:1, :], 1, 2)
  for idx in range(num_layers):
    fn = fullckpt_layer if idx < num_fullckpt else _dotckpt_systolic_layer
    z_cur_mc = fn(z_cur_mc, mask_mc, all_layer_params[idx])
  return jnp.swapaxes(z_cur_mc, 1, 2).reshape(b_global, c_z, n_i, n_j).astype(z0.dtype)
