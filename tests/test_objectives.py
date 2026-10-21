import math

import pytest
import torch

from selfplay_opsd.objectives import advantages, alignment, grpo, sd_coefficients, signed_gap


def test_population_advantages_and_equal_groups():
    r = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    a = advantages(r)
    expected = torch.tensor([-0.25, -0.25, -0.25, 0.75]) / (math.sqrt(0.1875) + 1e-6)
    torch.testing.assert_close(a[0], expected)
    assert torch.equal(a[1:], torch.zeros(2, 4))
    assert not a.requires_grad


def test_grpo_negative_advantages_and_per_episode_normalization():
    current = torch.tensor([[0.0, 0.0, 9.0], [0.0, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    loss = grpo(current, torch.zeros_like(current), torch.tensor([-1.0, 1.0]), mask)
    loss.backward()
    torch.testing.assert_close(
        current.grad, torch.tensor([[0.25, 0.25, 0.0], [-1 / 6, -1 / 6, -1 / 6]])
    )


@pytest.mark.parametrize("gap", [-3.0, 0.0, 2.0])
def test_signed_gap_direction_and_detachment(gap):
    logits = torch.tensor([0.3, -0.2], dtype=torch.double, requires_grad=True)
    old = torch.tensor([[0.0]], requires_grad=True)
    teacher = torch.tensor([[gap]], requires_grad=True)
    _, gate, coefficient = sd_coefficients(teacher, old, True)
    loss = signed_gap(logits.log_softmax(0)[0].view(1, 1), coefficient.double(), torch.ones(1, 1))
    loss.backward()
    assert old.grad is None and teacher.grad is None and not gate.requires_grad
    assert logits.grad[0].item() * gap <= 0
    eps = 1e-5
    numeric = []
    for i in range(2):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus[i] += eps
        minus[i] -= eps
        numeric.append(
            (
                signed_gap(
                    plus.log_softmax(0)[0].view(1, 1), coefficient.double(), torch.ones(1, 1)
                )
                - signed_gap(
                    minus.log_softmax(0)[0].view(1, 1), coefficient.double(), torch.ones(1, 1)
                )
            )
            / (2 * eps)
        )
    torch.testing.assert_close(logits.grad, torch.stack(numeric), atol=1e-8, rtol=1e-6)


def test_negative_gate_attenuates_without_renormalization():
    old, teacher = torch.zeros(1, 2), torch.tensor([[-2.0, 2.0]])
    gap, gate, c = sd_coefficients(teacher, old, True)
    _, _, ungated = sd_coefficients(teacher, old, False)
    assert abs(c[0, 0]) < abs(ungated[0, 0])
    assert c[0, 1] > 0 and c[0, 0] < 0
    torch.testing.assert_close(c, gate * gap)
    current = torch.ones(1, 2, requires_grad=True)
    torch.testing.assert_close(
        signed_gap(current, ungated, torch.ones_like(current)),
        signed_gap(current, gap, torch.ones_like(current)),
    )


def test_alignment_sign_zero_and_nonfinite():
    a = alignment([torch.tensor([1.0, -2.0])], [torch.tensor([-1.0, 2.0])])
    assert a.cosine == pytest.approx(-1.0) and a.reward == pytest.approx(0.0)
    a = alignment([torch.zeros(2)], [torch.ones(2)])
    assert a.cosine is None and a.reward == 0.5 and a.exact_zero
    a = alignment([torch.full((2,), 1e-14)], [torch.ones(2)])
    assert a.near_zero and not a.exact_zero
    with pytest.raises(FloatingPointError):
        alignment([torch.tensor([float("nan")])], [torch.ones(1)])
