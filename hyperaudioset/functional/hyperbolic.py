import torch


def mobius_add(
    input: torch.Tensor,
    other: torch.Tensor,
    radius: torch.Tensor | float = 1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply Mobius addition."""
    if input.size() != other.size():
        dim = max(input.dim(), other.dim())

        missing_dim = dim - input.dim()
        missing_shape = (1,) * missing_dim
        target_shape = missing_shape + input.size()
        input = input.view(*target_shape)

        missing_dim = dim - other.dim()
        missing_shape = (1,) * missing_dim
        target_shape = missing_shape + other.size()
        other = other.view(*target_shape)

        target_shape = ()

        for _input_size, _other_size in zip(input.size(), other.size()):
            target_shape = target_shape + (max(_input_size, _other_size),)

        input = input.expand(target_shape).contiguous()
        other = other.expand(target_shape).contiguous()

    *batch_shape, num_features = input.size()

    input = input.view(-1, num_features)
    other = other.view(-1, num_features)

    dot = torch.sum(input * other, dim=-1)
    norm_input = torch.sum(input**2, dim=-1)
    norm_other = torch.sum(other**2, dim=-1)

    c = 1 / (radius**2)
    coeff_input = 1 + 2 * c * dot + c * norm_other
    coeff_other = 1 - c * norm_input
    denom = 1 + 2 * c * dot + (c**2) * norm_input * norm_other
    num = coeff_input.unsqueeze(dim=-1) * input + coeff_other.unsqueeze(dim=-1) * other
    denom = torch.clamp(denom, min=eps)
    output = num / denom.unsqueeze(dim=-1)
    output = output.view(*batch_shape, num_features)

    return output


def mobius_sub(
    input: torch.Tensor,
    other: torch.Tensor,
    radius: torch.Tensor | float = 1,
    eps: float = 1e-5,
) -> torch.Tensor:
    return mobius_add(input, -other, radius=radius, eps=eps)


def poincare_distance(
    input: torch.Tensor,
    other: torch.Tensor,
    radius: float = 1,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert dim == -1

    distance = mobius_add(-input, other, radius=radius, eps=eps)
    norm = torch.linalg.vector_norm(distance, dim=dim) / radius
    scale = 2 * radius
    norm = torch.clamp(norm, max=1 - eps)
    output = scale * torch.atanh(norm)

    return output
