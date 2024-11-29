import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.models.base import Model, SPDFullRankModel, SPDModel, SPDRankPenaltyModel
from spd.types import RootPath
from spd.utils import remove_grad_parallel_to_subnetwork_vecs


class TMSModel(Model):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        n_hidden_layers: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers

        self.W = nn.Parameter(torch.empty((n_instances, n_features, n_hidden), device=device))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(torch.zeros((n_instances, n_features), device=device))

        self.hidden_layers = None
        if n_hidden_layers > 0:
            self.hidden_layers = nn.ParameterList()
            for _ in range(n_hidden_layers):
                layer = nn.Parameter(torch.empty((n_instances, n_hidden, n_hidden), device=device))
                nn.init.xavier_normal_(layer)
                self.hidden_layers.append(layer)

    def forward(
        self, features: Float[Tensor, "... n_instances n_features"]
    ) -> tuple[
        Float[Tensor, "... n_instances n_features"],
        dict[
            str,
            Float[Tensor, "... n_instances n_features"] | Float[Tensor, "... n_instances n_hidden"],
        ],
        dict[
            str,
            Float[Tensor, "... n_instances n_features"] | Float[Tensor, "... n_instances n_hidden"],
        ],
    ]:
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = torch.einsum("...if,ifh->...ih", features, self.W)

        pre_acts = {"W": features}
        post_acts = {"W": hidden}
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                pre_acts[f"hidden_{i}"] = hidden
                hidden = torch.einsum("...ik,ikj->...ij", hidden, layer)
                post_acts[f"hidden_{i}"] = hidden

        out_pre_relu = torch.einsum("...ih,ifh->...if", hidden, self.W) + self.b_final
        out = F.relu(out_pre_relu)

        pre_acts["W_T"] = hidden
        post_acts["W_T"] = out_pre_relu
        return out, pre_acts, post_acts

    def all_decomposable_params(self) -> dict[str, Float[Tensor, "n_instances d_in d_out"]]:
        """Dictionary of all parameters which will be decomposed with SPD."""
        params = {"W": self.W, "W_T": rearrange(self.W, "i f h -> i h f")}
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                params[f"hidden_{i}"] = layer
        return params


class TMSSPDModel(SPDModel):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        k: int | None,
        bias_val: float,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.k = k if k is not None else n_features
        self.bias_val = bias_val

        self.A = nn.Parameter(torch.empty((n_instances, n_features, self.k), device=device))
        self.B = nn.Parameter(torch.empty((n_instances, self.k, n_hidden), device=device))

        bias_data = torch.zeros((n_instances, n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data)

        nn.init.xavier_normal_(self.A)
        # Fix the first instance to the identity to compare losses
        assert (
            n_features == self.k
        ), "Currently only supports n_features == k if fixing first instance to identity"
        self.A.data[0] = torch.eye(n_features, device=device)
        nn.init.xavier_normal_(self.B)

        self.n_param_matrices = 2  # Two W matrices (even though they're tied)

    def all_As_and_Bs(
        self,
    ) -> dict[str, tuple[Float[Tensor, "d_layer_in k"], Float[Tensor, "k d_layer_out"]]]:
        return {
            "W": (self.A, self.B),
            "W_T": (rearrange(self.B, "i k h -> i h k"), rearrange(self.A, "i f k -> i k f")),
        }

    def all_subnetwork_params(self) -> dict[str, Float[Tensor, "n_instances k d_in d_out"]]:
        W = torch.einsum("ifk,ikh->ikfh", self.A, self.B)
        return {"W": W, "W_T": rearrange(W, "i k f h -> i k h f")}

    def all_subnetwork_params_summed(self) -> dict[str, Float[Tensor, "n_instances d_in d_out"]]:
        """All subnetwork params summed over the subnetwork dimension. I.e. all the ABs."""
        W = torch.einsum("ifk,ikh->ifh", self.A, self.B)
        return {"W": W, "W_T": rearrange(W, "i f h -> i h f")}

    def forward(
        self, x: Float[Tensor, "... i f"], topk_mask: Bool[Tensor, "... k"] | None = None
    ) -> tuple[
        Float[Tensor, "... i f"],
        dict[str, Float[Tensor, "... i d_layer_out"]],
        dict[str, Float[Tensor, "... i k"]],
    ]:
        inner_act_0 = torch.einsum("...if,ifk->...ik", x, self.A)
        if topk_mask is not None:
            assert topk_mask.shape == inner_act_0.shape
            inner_act_0 = torch.einsum("...ik,...ik->...ik", inner_act_0, topk_mask)
        layer_act_0 = torch.einsum("...ik,ikh->...ih", inner_act_0, self.B)

        inner_act_1 = torch.einsum("...ih,ikh->...ik", layer_act_0, self.B)
        if topk_mask is not None:
            assert topk_mask.shape == inner_act_1.shape
            inner_act_1 = torch.einsum("...ik,...ik->...ik", inner_act_1, topk_mask)
        layer_act_1 = torch.einsum("...ik,ifk->...if", inner_act_1, self.A) + self.b_final

        out = F.relu(layer_act_1)
        layer_acts = {"W": layer_act_0, "W_T": layer_act_1}
        inner_acts = {"W": inner_act_0, "W_T": inner_act_1}
        return out, layer_acts, inner_acts

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDModel":  # type: ignore
        pass

    def set_matrices_to_unit_norm(self):
        self.A.data /= self.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self):
        assert self.A.grad is not None
        remove_grad_parallel_to_subnetwork_vecs(self.A.data, self.A.grad)

    def set_subnet_to_zero(self, subnet_idx: int) -> dict[str, Float[Tensor, "n_instances dim2"]]:
        stored_vals = {
            "A": self.A.data[:, :, subnet_idx].detach().clone(),
            "B": self.B.data[:, subnet_idx, :].detach().clone(),
        }
        self.A.data[:, :, subnet_idx] = 0.0
        self.B.data[:, subnet_idx, :] = 0.0
        return stored_vals

    def restore_subnet(
        self, subnet_idx: int, stored_vals: dict[str, Float[Tensor, "n_instances dim2"]]
    ) -> None:
        self.A.data[:, :, subnet_idx] = stored_vals["A"]
        self.B.data[:, subnet_idx, :] = stored_vals["B"]


class TMSSPDFullRankModel(SPDFullRankModel):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        k: int | None,
        bias_val: float,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.k = k if k is not None else n_features
        self.bias_val = bias_val

        self.subnetwork_params = nn.Parameter(
            torch.empty((n_instances, self.k, n_features, n_hidden), device=device)
        )

        bias_data = torch.zeros((n_instances, n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data)

        nn.init.xavier_normal_(self.subnetwork_params)

        self.n_param_matrices = 2  # Two W matrices (even though they're tied)

    def all_subnetwork_params(
        self,
    ) -> dict[str, Float[Tensor, "n_instances k d_layer_in d_layer_out"]]:
        return {
            "W": self.subnetwork_params,
            "W_T": rearrange(self.subnetwork_params, "i k f h -> i k h f"),
        }

    def all_subnetwork_params_summed(
        self,
    ) -> dict[str, Float[Tensor, "n_instances d_layer_in d_layer_out"]]:
        """All subnetwork params summed over the subnetwork dimension."""
        summed_params = self.subnetwork_params.sum(dim=-3)
        return {"W": summed_params, "W_T": rearrange(summed_params, "i f h -> i h f")}

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k n_features"]],
    ]:
        inner_act_0 = torch.einsum("...if,ikfh->...ikh", x, self.subnetwork_params)
        if topk_mask is not None:
            assert topk_mask.shape == inner_act_0.shape[:-1]
            inner_act_0 = torch.einsum("...ikh,...ik->...ikh", inner_act_0, topk_mask)
        layer_act_0 = torch.einsum("...ikh->...ih", inner_act_0)

        inner_act_1 = torch.einsum("...ih,ikfh->...ikf", layer_act_0, self.subnetwork_params)
        if topk_mask is not None:
            assert topk_mask.shape == inner_act_1.shape[:-1]
            inner_act_1 = torch.einsum("...ikf,...ik->...ikf", inner_act_1, topk_mask)
        layer_act_1 = torch.einsum("...ikf->...if", inner_act_1) + self.b_final

        out = F.relu(layer_act_1)
        layer_acts = {"W": layer_act_0, "W_T": layer_act_1}
        inner_acts = {"W": inner_act_0, "W_T": inner_act_1}
        return out, layer_acts, inner_acts

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDFullRankModel":  # type: ignore
        pass

    def set_handcoded_spd_params(self, target_model: TMSModel):
        # Initialize the subnetwork params such that the kth subnetwork contains a single row of W
        # and the rest of the rows are zero
        assert self.n_features == self.k
        self.subnetwork_params.data = torch.zeros_like(self.subnetwork_params.data)
        for subnet_idx in range(self.k):
            feature_idx = subnet_idx
            self.subnetwork_params.data[:, subnet_idx, feature_idx, :] = target_model.W.data[
                :, feature_idx, :
            ]
        self.b_final.data = target_model.b_final.data

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, "n_instances n_features n_hidden"]]:
        stored_vals = {
            "subnetwork_params": self.subnetwork_params.data[:, subnet_idx, :, :].detach().clone()
        }
        self.subnetwork_params.data[:, subnet_idx, :, :] = 0.0
        return stored_vals

    def restore_subnet(
        self,
        subnet_idx: int,
        stored_vals: dict[str, Float[Tensor, "n_instances n_features n_hidden"]],
    ) -> None:
        self.subnetwork_params.data[:, subnet_idx, :, :] = stored_vals["subnetwork_params"]


class TMSSPDRankPenaltyModel(SPDRankPenaltyModel):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        k: int | None,
        bias_val: float,
        m: int | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.k = k if k is not None else n_features
        self.bias_val = bias_val

        self.m = min(n_features, n_hidden) + 1 if m is None else m

        self.A = nn.Parameter(torch.empty((n_instances, self.k, n_features, self.m), device=device))
        self.B = nn.Parameter(torch.empty((n_instances, self.k, self.m, n_hidden), device=device))

        bias_data = torch.zeros((n_instances, n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data)

        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)

        self.n_param_matrices = 2  # Two W matrices (even though they're tied)

    def all_subnetwork_params(self) -> dict[str, Float[Tensor, "n_instances k d_in d_out"]]:
        """Get all subnetwork parameters."""
        W = torch.einsum("ikfm,ikmh->ikfh", self.A, self.B)
        return {"W": W, "W_T": rearrange(W, "i k f h -> i k h f")}

    def all_subnetwork_params_summed(self) -> dict[str, Float[Tensor, "n_instances d_in d_out"]]:
        """All subnetwork params summed over the subnetwork dimension."""
        W = torch.einsum("ikfm,ikmh->ifh", self.A, self.B)
        return {"W": W, "W_T": rearrange(W, "i f h -> i h f")}

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k n_features"]],
    ]:
        # First layer: x -> A -> m dimension -> B -> hidden
        pre_inner_act_0 = torch.einsum("bif,ikfm->bikm", x, self.A)
        if topk_mask is not None:
            assert topk_mask.shape == pre_inner_act_0.shape[:-1]
            pre_inner_act_0 = torch.einsum("bikm,bik->bikm", pre_inner_act_0, topk_mask)
        inner_act_0 = torch.einsum("bikm,ikmh->bikh", pre_inner_act_0, self.B)
        layer_act_0 = torch.einsum("bikh->bih", inner_act_0)

        # Second layer: hidden -> B.T -> m dimension -> A.T -> features
        pre_inner_act_1 = torch.einsum("bih,ikmh->bikm", layer_act_0, self.B)
        if topk_mask is not None:
            assert topk_mask.shape == pre_inner_act_1.shape[:-1]
            pre_inner_act_1 = torch.einsum("bikm,bik->bikm", pre_inner_act_1, topk_mask)
        inner_act_1 = torch.einsum("bikm,ikfm->bikf", pre_inner_act_1, self.A)
        layer_act_1 = torch.einsum("bikf->bif", inner_act_1) + self.b_final

        out = F.relu(layer_act_1)
        layer_acts = {"W": layer_act_0, "W_T": layer_act_1}
        inner_acts = {"W": inner_act_0, "W_T": inner_act_1}
        return out, layer_acts, inner_acts

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, "n_instances n_features m"]]:
        stored_vals = {
            "A": self.A.data[:, subnet_idx, :, :].detach().clone(),
            "B": self.B.data[:, subnet_idx, :, :].detach().clone(),
        }
        self.A.data[:, subnet_idx, :, :] = 0.0
        self.B.data[:, subnet_idx, :, :] = 0.0
        return stored_vals

    def restore_subnet(
        self,
        subnet_idx: int,
        stored_vals: dict[str, Float[Tensor, "n_instances n_features m"]],
    ) -> None:
        self.A.data[:, subnet_idx, :, :] = stored_vals["A"]
        self.B.data[:, subnet_idx, :, :] = stored_vals["B"]

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDRankPenaltyModel":  # type: ignore
        pass

    def all_As_and_Bs(
        self,
    ) -> dict[
        str, tuple[Float[Tensor, "n_instances k d_in m"], Float[Tensor, "n_instances k m d_out"]]
    ]:
        """Get all A and B matrices. Note that this won't return bias components."""
        return {
            "W": (self.A, self.B),
            "W_T": (
                rearrange(self.B, "i k m h -> i k h m"),
                rearrange(self.A, "i k f m -> i k m f"),
            ),
        }

    def set_matrices_to_unit_norm(self) -> None:
        """Set the matrices that need to be normalized to unit norm."""
        self.A.data /= self.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self) -> None:
        """Modify the gradient by subtracting it's component parallel to the activation."""
        assert self.A.grad is not None
        remove_grad_parallel_to_subnetwork_vecs(self.A.data, self.A.grad)
