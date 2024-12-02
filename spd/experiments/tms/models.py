import einops
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.models.base import Model, SPDFullRankModel, SPDRankPenaltyModel
from spd.models.components import InstancesParamComponentsRankPenalty
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
        params: dict[str, torch.Tensor] = {"W": self.W, "W_T": rearrange(self.W, "i f h -> i h f")}
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                params[f"hidden_{i}"] = layer
        return params


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
        params: dict[str, Float[Tensor, "n_instances k d_layer_in d_layer_out"]] = {
            "W": self.subnetwork_params,
            "W_T": rearrange(self.subnetwork_params, "i k f h -> i k h f"),
        }
        return params

    def all_subnetwork_params_summed(
        self,
    ) -> dict[str, Float[Tensor, "n_instances d_layer_in d_layer_out"]]:
        """All subnetwork params summed over the subnetwork dimension."""
        summed_subnet_params = self.subnetwork_params.sum(dim=-3)
        summed_params: dict[str, Float[Tensor, "n_instances d_layer_in d_layer_out"]] = {
            "W": summed_subnet_params,
            "W_T": rearrange(summed_subnet_params, "i f h -> i h f"),
        }
        return summed_params

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
        n_hidden_layers: int,
        k: int | None,
        bias_val: float,
        m: int | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.k = k if k is not None else n_features
        self.bias_val = bias_val

        self.m = min(n_features, n_hidden) + 1 if m is None else m

        self.A = nn.Parameter(torch.empty((n_instances, self.k, n_features, self.m), device=device))
        self.B = nn.Parameter(torch.empty((n_instances, self.k, self.m, n_hidden), device=device))

        bias_data = torch.zeros((n_instances, n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data)

        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)

        self.hidden_layers = None
        if n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList(
                [
                    InstancesParamComponentsRankPenalty(
                        n_instances=n_instances,
                        in_dim=n_hidden,
                        out_dim=n_hidden,
                        k=self.k,
                        bias=False,
                        init_scale=1.0,
                        m=self.m,
                    )
                    for _ in range(n_hidden_layers)
                ]
            )

    def all_subnetwork_params(self) -> dict[str, Float[Tensor, "n_instances k d_in d_out"]]:
        """Get all subnetwork parameters."""
        W = einops.einsum(
            self.A,
            self.B,
            "n_instances k n_features m, n_instances k m n_hidden -> n_instances k n_features n_hidden",
        )
        W_T = einops.rearrange(
            W, "n_instances k n_features n_hidden -> n_instances k n_hidden n_features"
        )
        params: dict[str, Float[Tensor, "n_instances k d_in d_out"]] = {"W": W, "W_T": W_T}
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                params[f"hidden_{i}"] = einops.einsum(
                    layer.A,
                    layer.B,
                    "n_instances k d_in m, n_instances k m d_out -> n_instances k d_in d_out",
                )
        return params

    def all_subnetwork_params_summed(self) -> dict[str, Float[Tensor, "n_instances d_in d_out"]]:
        """All subnetwork params summed over the subnetwork dimension."""
        summed_params: dict[str, Float[Tensor, "n_instances d_in d_out"]] = {
            p_name: p.sum(dim=1) for p_name, p in self.all_subnetwork_params().items()
        }
        return summed_params

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k n_features"]],
    ]:
        layer_acts = {}
        inner_acts = {}

        # First layer/embedding: x -> A -> m dimension -> B -> hidden
        pre_inner_act_0 = torch.einsum("bif,ikfm->bikm", x, self.A)
        if topk_mask is not None:
            assert topk_mask.shape == pre_inner_act_0.shape[:-1]
            pre_inner_act_0 = torch.einsum("bikm,bik->bikm", pre_inner_act_0, topk_mask)
        inner_acts["W"] = torch.einsum("bikm,ikmh->bikh", pre_inner_act_0, self.B)
        layer_acts["W"] = torch.einsum("bikh->bih", inner_acts["W"])
        x = layer_acts["W"]

        # Hidden layers
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                x, hidden_inner_act_i = layer(x, topk_mask)
                layer_acts[f"hidden_{i}"] = x
                inner_acts[f"hidden_{i}"] = hidden_inner_act_i

        # Second layer/unembedding: hidden -> B.T -> m dimension -> A.T -> features
        pre_inner_act_1 = torch.einsum("bih,ikmh->bikm", x, self.B)
        if topk_mask is not None:
            assert topk_mask.shape == pre_inner_act_1.shape[:-1]
            pre_inner_act_1 = torch.einsum("bikm,bik->bikm", pre_inner_act_1, topk_mask)
        inner_acts["W_T"] = torch.einsum("bikm,ikfm->bikf", pre_inner_act_1, self.A)
        layer_acts["W_T"] = torch.einsum("bikf->bif", inner_acts["W_T"]) + self.b_final

        out = F.relu(layer_acts["W_T"])
        return out, layer_acts, inner_acts

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, "n_instances in_dim m"] | Float[Tensor, "n_instances m out_dim"]]:
        stored_vals = {
            "A": self.A.data[:, subnet_idx, :, :].detach().clone(),
            "B": self.B.data[:, subnet_idx, :, :].detach().clone(),
        }
        self.A.data[:, subnet_idx, :, :] = 0.0
        self.B.data[:, subnet_idx, :, :] = 0.0
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                stored_vals[f"hidden_{i}_A"] = layer.A.data[:, subnet_idx, :, :].detach().clone()
                stored_vals[f"hidden_{i}_B"] = layer.B.data[:, subnet_idx, :, :].detach().clone()
                layer.A.data[:, subnet_idx, :, :] = 0.0
                layer.B.data[:, subnet_idx, :, :] = 0.0

        return stored_vals

    def restore_subnet(
        self,
        subnet_idx: int,
        stored_vals: dict[
            str, Float[Tensor, "n_instances in_dim m"] | Float[Tensor, "n_instances m out_dim"]
        ],
    ) -> None:
        self.A.data[:, subnet_idx, :, :] = stored_vals["A"]
        self.B.data[:, subnet_idx, :, :] = stored_vals["B"]
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                layer.A.data[:, subnet_idx, :, :] = stored_vals[f"hidden_{i}_A"]
                layer.B.data[:, subnet_idx, :, :] = stored_vals[f"hidden_{i}_B"]

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDRankPenaltyModel":  # type: ignore
        pass

    def all_As_and_Bs(
        self,
    ) -> dict[
        str, tuple[Float[Tensor, "n_instances k d_in m"], Float[Tensor, "n_instances k m d_out"]]
    ]:
        """Get all A and B matrices. Note that this won't return bias components."""
        params: dict[
            str,
            tuple[Float[Tensor, "n_instances k d_in m"], Float[Tensor, "n_instances k m d_out"]],
        ] = {
            "W": (self.A, self.B),
            "W_T": (
                rearrange(self.B, "i k m h -> i k h m"),
                rearrange(self.A, "i k f m -> i k m f"),
            ),
        }
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                params[f"hidden_{i}"] = (layer.A, layer.B)
        return params

    def set_matrices_to_unit_norm(self) -> None:
        """Set the matrices that need to be normalized to unit norm."""
        self.A.data /= self.A.data.norm(p=2, dim=-2, keepdim=True)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                layer.A.data /= layer.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self) -> None:
        """Modify the gradient by subtracting it's component parallel to the activation."""
        assert self.A.grad is not None
        remove_grad_parallel_to_subnetwork_vecs(self.A.data, self.A.grad)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                assert layer.A.grad is not None
                remove_grad_parallel_to_subnetwork_vecs(layer.A.data, layer.A.grad)
