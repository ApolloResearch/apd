from pathlib import Path

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spd.run_spd import SPDModel, calc_lp_sparsity_loss, calc_param_match_loss, calc_topk_l2


class DummySPDModel(SPDModel):
    def __init__(self, d_in: int, d_out: int, k: int, n_instances: int | None = None):
        self.n_instances = n_instances
        self.n_param_matrices = 1
        self.d_in = d_in
        self.d_out = d_out
        self.k = k

    def all_As(self) -> list[Float[torch.Tensor, "... d_in k"]]:
        if self.n_instances is None:
            return [torch.ones(self.d_in, self.k)]
        return [torch.ones(self.n_instances, self.d_in, self.k)]

    def all_Bs(self) -> list[Float[torch.Tensor, "... k d_out"]]:
        if self.n_instances is None:
            return [torch.ones(self.k, self.d_out)]
        return [torch.ones(self.n_instances, self.k, self.d_out)]

    def forward_topk(
        self, x: Float[Tensor, "... dim"], topk_mask: Bool[Tensor, "... k"]
    ) -> tuple[
        Float[Tensor, "... dim"],
        list[Float[Tensor, "... dim"]],
        list[Float[Tensor, "... k"]],
    ]:
        return x, [x], [topk_mask]

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DummySPDModel":
        return cls(d_in=1, d_out=1, k=1)


class TestCalcTopkL2:
    def test_calc_topk_l2_single_instance_single_param_true_and_false(self):
        model = DummySPDModel(d_in=2, d_out=2, k=3)
        topk_mask: Float[Tensor, "batch=1 k=2"] = torch.tensor(
            [[True, False, False]], dtype=torch.bool
        )
        result = calc_topk_l2(model, topk_mask, device="cpu")

        # Below we write what the intermediate values are
        # A_topk = torch.tensor([[[1, 0, 0], [1, 0, 0]]])
        # AB_topk = torch.tensor([[[1, 1], [1, 1]]])
        expected = torch.tensor(1.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_topk_l2_single_instance_single_param_true_and_true(self):
        model = DummySPDModel(d_in=2, d_out=2, k=3)
        topk_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
        result = calc_topk_l2(model, topk_mask, device="cpu")

        # Below we write what the intermediate values are
        # A_topk = torch.tensor([[[1, 1, 1], [1, 1, 1]]])
        # AB_topk = torch.tensor([[[3, 3], [3, 3]]])
        expected = torch.tensor(9.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_topk_l2_multiple_instances(self):
        model = DummySPDModel(d_in=1, d_out=1, n_instances=2, k=2)
        # topk_mask: [batch=2, n_instances=2, k=2]
        topk_mask = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 1]]], dtype=torch.bool)
        result = calc_topk_l2(model, topk_mask, device="cpu")

        # Below we write what the intermediate values are
        # A: [n_instances=2, d_in=1, k=2] = torch.tensor(
        #     [
        #         [[1, 1]],
        #         [[1, 1]]
        #     ]
        # )
        # A_topk: [batch=2, n_instances=2, d_in=1, k=2] = torch.tensor([
        #     [
        #         [[1, 0]],
        #         [[0, 1]]
        #     ],
        #     [
        #         [[0, 1]],
        #         [[1, 1]]
        #     ]
        # ])
        # B: [n_instances=2, k=2, d_out=1] = torch.tensor([
        #     [
        #         [[1]],
        #         [[1]]
        #     ],
        #     [
        #         [[1]],
        #         [[1]]
        #     ]
        # ])
        # AB_topk: [batch=2, n_instances=2, d_in=1, d_out=1] = torch.tensor([
        #     [
        #         [[1]],
        #         [[1]]
        #     ],
        #     [
        #         [[1]],
        #         [[2]]
        #     ]
        # ])
        # topk_l2_penalty (pre-reduction): [batch=2, n_instances=2] = torch.tensor([
        #     [1, 1],
        #     [1, 4]
        # ])
        # topk_l2_penalty (post-reduction): [n_instances=2] = torch.tensor([1, 2.5])
        expected = torch.tensor([1.0, 2.5])
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


class TestCalcParamMatchLoss:
    def test_calc_param_match_loss_single_instance_single_param(self):
        model = DummySPDModel(d_in=2, d_out=2, k=3)
        pretrained_weights = [torch.tensor([[1.0, 1.0], [1.0, 1.0]])]
        result = calc_param_match_loss(model, pretrained_weights, device="cpu")

        # A: [2, 3], B: [3, 2], both filled with ones
        # AB: [[3, 3], [3, 3]]
        # (AB - pretrained_weights)^2: [[4, 4], [4, 4]]
        # Mean: 4
        expected = torch.tensor(4.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_param_match_loss_single_instance_multiple_params(self):
        class MultiParamDummySPDModel(DummySPDModel):
            def __init__(self, d_in: int, d_mid: int, d_out: int, k: int):
                super().__init__(d_in, d_out, k)
                self.n_param_matrices = 2
                self.d_mid = d_mid

            def all_As(self) -> list[Float[Tensor, "d_in k"]]:
                return [torch.ones(self.d_in, self.k), torch.ones(self.d_mid, self.k)]

            def all_Bs(self) -> list[Float[Tensor, "k d_out"]]:
                return [torch.ones(self.k, self.d_mid), torch.ones(self.k, self.d_out)]

        model = MultiParamDummySPDModel(d_in=2, d_mid=3, d_out=2, k=3)
        pretrained_weights = [
            torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        ]
        result = calc_param_match_loss(model, pretrained_weights, device="cpu")

        # First layer: AB1: [[3, 3, 3], [3, 3, 3]], diff^2: [[1, 1, 1], [1, 1, 1]]
        # Second layer: AB2: [[3, 3], [3, 3], [3, 3]], diff^2: [[4, 4], [4, 4], [4, 4]]
        # Average of both layers: (1 + 4) / 2 = 2.5
        expected = torch.tensor(2.5)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_param_match_loss_multiple_instances(self):
        class MultiInstanceDummySPDModel(DummySPDModel):
            def all_As(self) -> list[Float[Tensor, "n_instances d_in k"]]:
                assert self.n_instances is not None
                return [torch.ones(self.n_instances, self.d_in, self.k)]

            def all_Bs(self) -> list[Float[Tensor, "n_instances k d_out"]]:
                assert self.n_instances is not None
                return [torch.ones(self.n_instances, self.k, self.d_out)]

        model = MultiInstanceDummySPDModel(d_in=2, d_out=2, k=3, n_instances=2)
        pretrained_weights = [torch.tensor([[[2.0, 2.0], [2.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])]
        result = calc_param_match_loss(model, pretrained_weights, device="cpu")

        # AB [n_instances=2, d_in=2, d_out=2]: [[[3, 3], [3, 3]], [[3, 3], [3, 3]]]
        # diff^2: [[[1, 1], [1, 1]], [[4, 4], [4, 4]]]
        # mean: [1, 4]
        expected = torch.tensor([1.0, 4.0])
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


class TestCalcLpSparsityLoss:
    def test_calc_lp_sparsity_loss_single_instance(self):
        inner_acts: list[Float[Tensor, "batch=1 k=3"]] = [torch.tensor([[1.0, 1.0, 1.0]])]
        layer_out_params: list[Float[Tensor, "k=3 d_out=2"]] = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        ]

        # Compute layer_acts
        layer_acts = [inner_acts[0] @ layer_out_params[0]]
        # Expected layer_acts: [9.0, 12.0]

        # Compute out (assuming identity activation function)
        out = layer_acts[0]
        # Expected out: [9.0, 12.0]

        step_pnorm = 0.9

        result = calc_lp_sparsity_loss(
            out=out,
            layer_acts=layer_acts,
            inner_acts=inner_acts,
            layer_out_params=layer_out_params,
            step_pnorm=step_pnorm,
        )

        # Take derivative w.r.t each output dimension

        # d9/dlayer_acts = 1
        # d9/dinner_acts = [1, 3, 5]
        # attributions for 9 = [1, 3, 5] * [1, 1, 1] = [1, 3, 5]

        # d12/dinner_acts = [2, 4, 6]
        # attributions for 12 = [2, 4, 6] * [1, 1, 1] = [2, 4, 6]

        # Add the squared attributions = [1^2, 3^2, 5^2] + [2^2, 4^2, 6^2] = [5, 25, 61]
        # Divide by 2 (since we have two terms) = [2.5, 12.5, 30.5]
        # Take to the power of 0.9 and sqrt = [2.5^(0.9*0.5), 12.5^(0.9*0.5), 30.5^(0.9*0.5)]
        # Sum = 2.5^(0.9*0.5) + 12.5^(0.9*0.5) + 30.5^(0.9*0.5)

        expected_val = (2.5 ** (0.9 * 0.5)) + (12.5 ** (0.9 * 0.5)) + (30.5 ** (0.9 * 0.5))
        expected = torch.tensor(expected_val)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

        # Also check that the layer_out_params is in the computation graph of the sparsity result.
        result.backward()
        assert layer_out_params[0].grad is not None
