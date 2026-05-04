from src.transformer_modules.RMSNorm import *
import pytest


def test_output_shape_matches_input_shape():
    norm = RMSNorm(dim=32)

    x = torch.randn(4, 16, 32)
    y = norm(x)

    assert y.shape == x.shape


def test_output_dtype_matches_input_dtype():
    norm = RMSNorm(dim=32)

    x = torch.randn(4, 16, 32, dtype=torch.float32)
    y = norm(x)

    assert y.dtype == x.dtype


def test_weight_shape_is_dim():
    norm = RMSNorm(dim=32)

    assert norm.weight.shape == (32,)


def test_weight_initialized_to_ones():
    norm = RMSNorm(dim=32)

    assert torch.allclose(norm.weight, torch.ones_like(norm.weight))


def test_forward_matches_manual_computation():
    dim = 32
    eps = 1e-6

    norm = RMSNorm(dim=dim, eps=eps)

    x = torch.randn(4, 16, dim)

    y = norm(x)

    x_float = x.float()
    mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
    expected = x_float * torch.rsqrt(mean_square + eps)
    expected = expected.to(x.dtype) * norm.weight

    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-5)


def test_normalizes_last_dimension_only():
    dim = 8
    eps = 1e-6

    norm = RMSNorm(dim=dim, eps=eps)

    x = torch.randn(2, 3, 4, dim)
    y = norm(x)

    manual_mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
    expected = x.float() * torch.rsqrt(manual_mean_square + eps)
    expected = expected.to(x.dtype) * norm.weight

    assert y.shape == x.shape
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-5)


def test_no_nan_with_zero_input():
    norm = RMSNorm(dim=32)

    x = torch.zeros(4, 16, 32)
    y = norm(x)

    assert torch.isfinite(y).all()
    assert torch.allclose(y, torch.zeros_like(y))


def test_backward_produces_gradients():
    norm = RMSNorm(dim=32)

    x = torch.randn(4, 16, 32, requires_grad=True)
    y = norm(x)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


def test_supports_3d_input_BTD():
    norm = RMSNorm(dim=32)

    x = torch.randn(4, 16, 32)
    y = norm(x)

    assert y.shape == (4, 16, 32)


def test_supports_4d_input_BHTD():
    norm = RMSNorm(dim=32)

    x = torch.randn(4, 8, 16, 32)
    y = norm(x)

    assert y.shape == (4, 8, 16, 32)


def test_handles_large_values_without_nan():
    norm = RMSNorm(dim=32)

    x = torch.full((4, 16, 32), 1e6)
    y = norm(x)

    assert torch.isfinite(y).all()


def test_handles_small_values_without_nan():
    norm = RMSNorm(dim=32)

    x = torch.full((4, 16, 32), 1e-8)
    y = norm(x)

    assert torch.isfinite(y).all()


def test_eps_prevents_division_by_zero():
    norm = RMSNorm(dim=32, eps=1e-6)

    x = torch.zeros(4, 16, 32)
    y = norm(x)

    assert torch.isfinite(y).all()


def test_bfloat16_input_returns_bfloat16_if_supported():
    norm = RMSNorm(dim=32)

    x = torch.randn(4, 16, 32).to(torch.bfloat16)
    y = norm(x)

    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_float16_input_returns_float16_on_cuda_if_available():
    norm = RMSNorm(dim=32).cuda()

    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.float16)
    y = norm(x)

    assert y.dtype == torch.float16
    assert y.device.type == "cuda"
    assert torch.isfinite(y.float()).all()