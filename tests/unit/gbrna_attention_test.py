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

"""Tests for the fused GB.RNA RoPE + attention Pallas kernel."""

import unittest

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels import gbrna_attention as ga


def _inputs(b, s, h, dtype, seed=0):
  keys = jax.random.split(jax.random.PRNGKey(seed), 4)
  shape = (b, s, h * 64)
  q, k, v, do = (jax.random.normal(kk, shape, jnp.float32).astype(dtype) for kk in keys)
  return q, k, v, do


class RotateHalfMatrixTest(unittest.TestCase):

  def test_matches_rotate_half(self):
    x = np.random.RandomState(0).randn(5, 128).astype(np.float32)
    p = ga.rotate_half_matrix()
    want = np.concatenate([np.concatenate([-x[:, 32:64], x[:, 0:32]], -1), np.concatenate([-x[:, 96:], x[:, 64:96]], -1)], -1)
    np.testing.assert_allclose(x @ p, want, atol=0)


class GbRnaAttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("g2_q256", 2, 256, 256),
      ("g4_q128", 4, 128, 128),
      ("g4_qfull_bwd64", 4, 256, 64),
  )
  def test_fwd_and_grads_fp32(self, heads_per_step, block_q, block_q_bwd):
    q, k, v, do = _inputs(2, 256, 8, jnp.float32)
    fn = lambda q, k, v: ga.gbrna_attention(q, k, v, 0.125, heads_per_step, block_q, block_q_bwd, 96 * 2**20, True)
    out, vjp = jax.vjp(fn, q, k, v)
    ref_out, ref_vjp = jax.vjp(ga.reference_attention, q, k, v)
    np.testing.assert_allclose(out, ref_out, atol=2e-5, rtol=1e-4)
    for got, want in zip(vjp(do), ref_vjp(do)):
      np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-3)

  def test_bf16_close_to_fp32_reference(self):
    q, k, v, do = _inputs(1, 256, 4, jnp.bfloat16, seed=1)
    fn = lambda q, k, v: ga.gbrna_attention(q, k, v, 0.125, 4, 256, 256, 96 * 2**20, True)
    out, vjp = jax.vjp(fn, q, k, v)
    f32 = lambda x: x.astype(jnp.float32)
    ref_out, ref_vjp = jax.vjp(ga.reference_attention, f32(q), f32(k), f32(v))
    np.testing.assert_allclose(f32(out), ref_out, atol=3e-2, rtol=3e-2)
    for got, want in zip(vjp(do), ref_vjp(f32(do))):
      scale = float(jnp.max(jnp.abs(want)))
      np.testing.assert_allclose(f32(got) / scale, want / scale, atol=3e-2)

  def test_unit_scale_matches_reference(self):
    q, k, v, do = _inputs(1, 128, 2, jnp.float32, seed=3)
    q = q * 0.125  # MaxText folds the scale into the query projection.
    fn = lambda q, k, v: ga.gbrna_attention(q, k, v, 1.0, 2, 128, 128, 96 * 2**20, True)
    ref = lambda q, k, v: ga.reference_attention(q, k, v, 1.0)
    out, vjp = jax.vjp(fn, q, k, v)
    ref_out, ref_vjp = jax.vjp(ref, q, k, v)
    np.testing.assert_allclose(out, ref_out, atol=2e-5, rtol=1e-4)
    for got, want in zip(vjp(do), ref_vjp(do)):
      np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-3)

  def test_bidirectional(self):
    q, k, v, _ = _inputs(1, 256, 2, jnp.float32, seed=2)
    fn = lambda k: ga.gbrna_attention(q, k, v, 0.125, 2, 256, 256, 96 * 2**20, True)
    k2 = k.at[:, -1].set(0.0)
    self.assertGreater(float(jnp.max(jnp.abs(fn(k)[:, 0] - fn(k2)[:, 0]))), 1e-6)


if __name__ == "__main__":
  unittest.main()
