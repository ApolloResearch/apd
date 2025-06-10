import torch

from spd.losses import _calc_tensors_mse


class TestCalcParamMatchLoss:
    # Actually testing _calc_tensors_mse. calc_faithfulness_loss should fail hard in most cases, and
    # testing it would require lots of mocking the way it is currently written.
    def test_calc_faithfulness_loss_single_instance_single_param(self):
        A = torch.ones(2, 3)
        B = torch.ones(3, 2)
        n_params = 2 * 3 * 2
        spd_params = {"layer1": A @ B}
        target_params = {"layer1": torch.tensor([[1.0, 1.0], [1.0, 1.0]])}

        result = _calc_tensors_mse(
            params1=target_params,
            params2=spd_params,
            n_params=n_params,
            device="cpu",
        )

        # A: [2, 3], B: [3, 2], both filled with ones
        # AB: [[3, 3], [3, 3]]
        # (AB - pretrained_weights)^2: [[4, 4], [4, 4]]
        # Sum and divide by n_params: 16 / 12 = 4/3
        expected = torch.tensor(4.0 / 3.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_faithfulness_loss_single_instance_multiple_params(self):
        As = [torch.ones(2, 3), torch.ones(3, 3)]
        Bs = [torch.ones(3, 3), torch.ones(3, 2)]
        n_params = 2 * 3 * 3 + 3 * 3 * 2
        target_params = {
            "layer1": torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            "layer2": torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        }
        spd_params = {
            "layer1": As[0] @ Bs[0],
            "layer2": As[1] @ Bs[1],
        }
        result = _calc_tensors_mse(
            params1=target_params,
            params2=spd_params,
            n_params=n_params,
            device="cpu",
        )

        # First layer: AB1: [[3, 3, 3], [3, 3, 3]], diff^2: [[1, 1, 1], [1, 1, 1]]
        # Second layer: AB2: [[3, 3], [3, 3], [3, 3]], diff^2: [[4, 4], [4, 4], [4, 4]]
        # Add together 24 + 6 = 30
        # Divide by n_params: 30 / (18+18) = 5/6
        expected = torch.tensor(5.0 / 6.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
