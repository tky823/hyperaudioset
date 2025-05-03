from typing import Union

import torch


def mobius_add(
    input: torch.Tensor,
    other: torch.Tensor,
    curvature: Union[float, torch.Tensor] = -1,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply Mobius addition.

    Args:
        input (torch.Tensor): Vectors of shape (*, num_features).
        other (torch.Tensor): Vectors of shape (*, num_features).
        curvature (float or torch.Tensor): Secctional curvature. Default: ``-1``.

    Returns:
        torch.Tensor: Vectors of shape (*, num_features).

    """
    assert dim == -1, "Only dim=-1 is supported."

    if not isinstance(input, torch.Tensor):
        if isinstance(other, torch.Tensor):
            factory_kwargs = {
                "dtype": other.dtype,
                "device": other.device,
            }
        else:
            factory_kwargs = {}

        input = torch.tensor(input, **factory_kwargs)

    if not isinstance(other, torch.Tensor):
        if isinstance(input, torch.Tensor):
            factory_kwargs = {
                "dtype": input.dtype,
                "device": input.device,
            }
        else:
            factory_kwargs = {}

        other = torch.tensor(other, **factory_kwargs)

    target_shape = torch.broadcast_shapes(input.size(), other.size())

    if target_shape == ():
        # corner case
        target_shape = (1,)

    input = input.expand(target_shape).contiguous()
    other = other.expand(target_shape).contiguous()

    *batch_shape, num_features = input.size()

    input = input.view(-1, num_features)
    other = other.view(-1, num_features)

    dot = torch.sum(input * other, dim=-1)
    norm_input = torch.sum(input**2, dim=-1)
    norm_other = torch.sum(other**2, dim=-1)

    coeff_input = 1 - 2 * curvature * dot - curvature * norm_other
    coeff_other = 1 + curvature * norm_input
    denom = 1 - 2 * curvature * dot + (curvature**2) * norm_input * norm_other
    num = coeff_input.unsqueeze(dim=-1) * input + coeff_other.unsqueeze(dim=-1) * other
    denom = torch.clamp(denom, min=eps)
    output = num / denom.unsqueeze(dim=-1)
    output = output.view(*batch_shape, num_features)

    return output


def mobius_sub(
    input: torch.Tensor,
    other: torch.Tensor,
    curvature: Union[float, torch.Tensor] = -1,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply Mobius subtraction.

    Args:
        input (torch.Tensor): Vectors of shape (*, num_features).
        other (torch.Tensor): Vectors of shape (*, num_features).
        curvature (float or torch.Tensor): Negative curvature.

    Returns:
        torch.Tensor: Vectors of shape (*, num_features).

    """
    return mobius_add(input, -other, curvature=curvature, dim=dim, eps=eps)


def poincare_distance(
    input: torch.Tensor,
    other: torch.Tensor,
    curvature: float = -1,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Distance between two points on Poincare ball with negative curvature."""
    assert dim == -1

    _curvature = (-curvature) ** 0.5
    distance = mobius_add(-input, other, curvature=curvature, eps=eps)
    norm = _curvature * torch.linalg.vector_norm(distance, dim=dim)
    scale = 2 / _curvature
    norm = torch.clamp(norm, max=1 - eps)
    output = scale * torch.atanh(norm)

    return output
