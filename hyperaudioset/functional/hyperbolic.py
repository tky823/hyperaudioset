import torch


def mobius_add(
    input: torch.Tensor,
    other: torch.Tensor,
    curvature: torch.Tensor | float = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
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

    coeff_input = 1 + 2 * curvature * dot + curvature * norm_other
    coeff_other = 1 - curvature * norm_input
    denom = 1 + 2 * curvature * dot + curvature * norm_input * norm_other
    num = coeff_input.unsqueeze(dim=-1) * input + coeff_other.unsqueeze(dim=-1) * other
    denom = torch.clamp(denom, min=eps)
    output = num / denom.unsqueeze(dim=-1)
    output = output.view(*batch_shape, num_features)

    return output


def poincare_distance(
    input: torch.Tensor, other: torch.Tensor, curvature: float = 1, dim: int = -1
) -> torch.Tensor:
    assert dim == -1

    distance = mobius_add(-input, other, curvature=curvature)
    norm = curvature**0.5 * torch.linalg.vector_norm(distance, dim=-1)
    scale = 2 / (curvature**0.5)
    output = scale * torch.tanh(norm)

    return output
