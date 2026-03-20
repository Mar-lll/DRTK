# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests that DRTK's Python reference implementations support second-order
(higher-order) automatic differentiation.

The primary CUDA-backed operations (render, interpolate, edge_grad_estimator)
only support first-order gradients.  The *_ref variants use pure PyTorch ops so
the autograd tape is maintained through both the forward pass and the backward
pass.  Calling .backward(create_graph=True) (or torch.autograd.grad with
create_graph=True) on the reference pipeline therefore yields differentiable
first-order gradients, and a second call to .backward() / autograd.grad()
produces valid second-order gradients (Hessian-vector products).

This file exercises that property for:
  - interpolate_ref
  - render_ref
  - edge_grad_estimator_ref  (the key function: it previously used the CUDA
                               interpolate kernel internally, which blocked
                               second-order gradients)
"""

import torch as th
import torch.nn.functional as thf
from drtk import edge_grad_estimator_ref, interpolate_ref, render_ref, rasterize


def _make_simple_mesh(device):
    """Two triangles, 6 vertices, in pixel-space coordinates."""
    v = th.tensor(
        [
            [10.0, 200.0, 100.0],
            [300.0, 50.0, 100.0],
            [400.0, 500.0, 100.0],
            [50.0, 400.0, 200.0],
            [400.0, 50.0, 50.0],
            [300.0, 500.0, 200.0],
        ],
        dtype=th.float64,
        device=device,
    )

    vi = th.arange(6, device=device).int().view(2, 3)
    return v, vi


def test_interpolate_ref_second_order(device="cpu"):
    """interpolate_ref should support second-order gradients."""
    v = th.randn(1, 4, 3, dtype=th.float64, device=device, requires_grad=True)
    bary_img = th.rand(1, 3, 8, 8, dtype=th.float64, device=device)
    # Normalize bary coords so they sum to 1
    bary_img = bary_img / bary_img.sum(dim=1, keepdim=True)
    vi = th.tensor([[0, 1, 2], [1, 2, 3]], dtype=th.int32, device=device)
    index_img = th.zeros(1, 8, 8, dtype=th.int32, device=device)

    result = interpolate_ref(v, vi, index_img, bary_img)
    loss = result.sum()

    # First-order gradient with create_graph=True so the graph is retained.
    (grad,) = th.autograd.grad(loss, v, create_graph=True)
    assert grad is not None, "First-order gradient should be defined"
    assert th.isfinite(grad).all(), "First-order gradient should be finite"

    # Second-order gradient.
    (grad2,) = th.autograd.grad(grad.sum(), v)
    assert grad2 is not None, "Second-order gradient should be defined"
    assert th.isfinite(grad2).all(), "Second-order gradient should be finite"


def test_render_ref_second_order(device="cpu"):
    """render_ref should support second-order gradients."""
    v = th.tensor(
        [[[50.0, 50.0, 10.0], [200.0, 50.0, 10.0], [100.0, 200.0, 10.0]]],
        dtype=th.float64,
        device=device,
        requires_grad=True,
    )
    vi = th.tensor([[0, 1, 2]], dtype=th.int32, device=device)
    index_img = th.zeros(1, 16, 16, dtype=th.int32, device=device)

    _depth, bary = render_ref(v, vi, index_img)
    loss = bary.sum()

    (grad,) = th.autograd.grad(loss, v, create_graph=True)
    assert grad is not None, "First-order gradient should be defined"
    assert th.isfinite(grad).all(), "First-order gradient should be finite"

    (grad2,) = th.autograd.grad(grad.sum(), v)
    assert grad2 is not None, "Second-order gradient should be defined"
    assert th.isfinite(grad2).all(), "Second-order gradient should be finite"


def test_edge_grad_estimator_ref_second_order(device="cuda"):
    """
    edge_grad_estimator_ref should support second-order gradients.

    This test requires CUDA because rasterize() only runs on GPU.  The second-
    order differentiability comes from using interpolate_ref (pure PyTorch) for
    the internal v_pix_img computation instead of the CUDA interpolate kernel.
    """
    if not th.cuda.is_available():
        print(
            "SKIP test_edge_grad_estimator_ref_second_order: CUDA not available"
        )
        return

    v, vi = _make_simple_mesh(device)
    v = v[None]  # add batch dim -> [1, 6, 3]
    v = v.requires_grad_(True)
    vt = th.zeros(1, 6, 2, device=device, dtype=th.float64)
    vt[:, 3:6, 0] = 1.0

    h, w = 64, 64
    tex = th.ones(1, 3, 16, 16, device=device, dtype=th.float64)
    tex[:, :, :, 8:] = 0.5

    index_img = rasterize(v.float(), vi, h, w)

    # Use render_ref so bary_img is also differentiable.
    _, bary_img = render_ref(v, vi, index_img)

    vt_img = interpolate_ref(vt, vi, index_img, bary_img)
    img = thf.grid_sample(
        tex.float(),
        vt_img.permute(0, 2, 3, 1).float(),
        padding_mode="border",
        align_corners=False,
    ).double()
    img = img * (index_img != -1)[:, None].double()

    img_out = edge_grad_estimator_ref(
        v_pix=v,
        vi=vi,
        bary_img=bary_img,
        img=img,
        index_img=index_img,
    )

    loss = img_out.sum()

    # First-order gradient, keeping the graph.
    (grad,) = th.autograd.grad(loss, v, create_graph=True)
    assert grad is not None, "First-order gradient should be defined"
    assert th.isfinite(grad).all(), "First-order gradient should be finite"

    # Second-order gradient.
    (grad2,) = th.autograd.grad(grad.sum(), v)
    assert grad2 is not None, "Second-order gradient should be defined"
    assert th.isfinite(grad2).all(), "Second-order gradient should be finite"

    print("test_edge_grad_estimator_ref_second_order PASSED")


if __name__ == "__main__":
    print("Running second-order gradient tests...")

    test_interpolate_ref_second_order(device="cpu")
    print("test_interpolate_ref_second_order PASSED")

    test_render_ref_second_order(device="cpu")
    print("test_render_ref_second_order PASSED")

    test_edge_grad_estimator_ref_second_order(device="cuda")
