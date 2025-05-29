import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.tms.models import TMSModel, TMSSPDModel, TMSTaskConfig
from spd.settings import REPO_ROOT

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instance_idx = 0
    # run_id = "wandb:spd-tms/runs/u359w3kq"
    # run_id = "wandb:spd-tms/runs/hrwrgei2"
    # run_id = "wandb:spd-tms/runs/3p8qgg6b"
    # pretrained_model_path = "wandb:spd-train-tms/runs/tmzweoqk"
    # run_id = "wandb:spd-tms/runs/fj68gebo"
    # target_model, target_model_train_config_dict = TMSModel.from_pretrained(pretrained_model_path)
    # spd_model, spd_model_train_config_dict = TMSSPDModel.from_pretrained(run_id)

    spd_model_path = "/mnt/polished-lake/home/lee/apd/spd/experiments/tms/out/maskrecon0.00e+00_nrandmasks1_randrecon1.00e+00_p3.00e+00_lpsp1.00e-03_m20_sd0_attr-gra_lr1.00e-03_bs2048_ft5_hid2hid-layers0_20250529_103056_132/spd_model_30000.pth"

    # Plot showing polygons for each subnet
    spd_model, config = TMSSPDModel.from_pretrained(spd_model_path)
    As = spd_model.linear1.A.detach().cpu()
    Bs = spd_model.linear1.B.detach().cpu()
    subnets = torch.einsum("I f C, I C h -> I C f h", As, Bs)

    assert isinstance(config.task_config, TMSTaskConfig)
    target_model, target_model_train_config_dict = TMSModel.from_pretrained(
        config.task_config.pretrained_model_path
    )

    out_dir = REPO_ROOT / "spd/experiments/tms/out/figures/"
    out_dir.mkdir(parents=True, exist_ok=True)

    # %%
    # Max cosine similarity between subnets and target model
    def plot_max_cosine_sim(max_cosine_sim: Float[Tensor, " n_features"]) -> plt.Figure:
        fig, ax = plt.subplots()
        # Make a bar plot of the max cosine similarity for each feature
        ax.bar(range(max_cosine_sim.shape[0]), max_cosine_sim.cpu().detach().numpy())
        # Add a grey horizontal line at 1
        ax.axhline(1, color="grey", linestyle="--")
        ax.set_xlabel("Input feature index")
        ax.set_ylabel("Max cosine similarity")
        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return fig

    cosine_sims = torch.einsum(
        "C f h, f h -> C f",
        subnets[instance_idx] / torch.norm(subnets[instance_idx], dim=-1, keepdim=True),
        target_model.linear1.weight[instance_idx]
        / torch.norm(target_model.linear1.weight[instance_idx], dim=-1, keepdim=True),
    )
    max_cosine_sim = cosine_sims.max(dim=0).values
    print(f"Max cosine similarity:\n{max_cosine_sim}")
    print(f"Mean max cosine similarity: {max_cosine_sim.mean()}")
    print(f"std max cosine similarity: {max_cosine_sim.std()}")

    # Get the subnet weights at the max cosine similarity
    subnet_weights_at_max_cosine_sim: Float[Tensor, "n_features n_hidden"] = subnets[
        instance_idx, cosine_sims.max(dim=0).indices, torch.arange(target_model.config.n_features)
    ]
    # Get the norm of the target model weights
    target_model_weights_norm = torch.norm(
        target_model.linear1.weight[instance_idx], dim=-1, keepdim=True
    )
    # Get the norm of subnet_weights_at_max_cosine_sim
    subnet_weights_at_max_cosine_sim_norm = torch.norm(
        subnet_weights_at_max_cosine_sim, dim=-1, keepdim=True
    )
    # Divide the subnet weights by the target model weights ratio
    l2_ratio = subnet_weights_at_max_cosine_sim_norm / target_model_weights_norm
    print(f"Mean L2 ratio: {l2_ratio.mean()}")
    print(f"std L2 ratio: {l2_ratio.std()}")

    # Mean bias
    print(f"Mean bias: {target_model.b_final[instance_idx].mean()}")

    pass
