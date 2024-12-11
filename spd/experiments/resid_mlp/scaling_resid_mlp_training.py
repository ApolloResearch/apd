import json
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.experiments.resid_mlp.train_resid_mlp import (
    ResidMLPTrainConfig,
    ResidualMLPConfig,
    run_train,
)
from spd.settings import REPO_ROOT
from spd.utils import set_seed


def train_on_test_data(
    n_instances: int,
    n_steps: int,
    d_embed: int,
    p: float,
    bias: bool,
    n_features: int,
    d_mlp: int,
    fixed_random_embedding: bool,
    fixed_identity_embedding: bool,
    loss_type: Literal["readoff", "resid"],
) -> dict[int, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ResidMLPTrainConfig(
        wandb_project=None,
        seed=0,
        resid_mlp_config=ResidualMLPConfig(
            n_instances=n_instances,
            n_features=n_features,
            d_embed=d_embed,
            d_mlp=d_mlp,
            n_layers=1,
            act_fn_name="relu",
            apply_output_act_fn=False,
            in_bias=bias,
            out_bias=bias,
        ),
        loss_type=loss_type,
        feature_probability=p,
        importance_val=None,
        batch_size=256,
        steps=n_steps,
        print_freq=100,
        lr=3e-3,
        lr_schedule="cosine",
        fixed_random_embedding=fixed_random_embedding,
        fixed_identity_embedding=fixed_identity_embedding,
        n_batches_final_losses=100,
    )

    set_seed(config.seed)
    loss = run_train(config, device)
    loss_dict = {i: loss[i].item() for i in range(n_instances)}
    return loss_dict


def plot_loss_curve(ax: plt.Axes, losses: dict[int, dict[int, float]], label: str):
    xvals = np.array(list(losses.keys()))
    # 2D array of y vals, due to instance dimension
    yvals = np.array([list(losses[x].values()) for x in xvals])
    yvals_mean = yvals.mean(axis=1)
    yvals_lower = yvals.min(axis=1)
    yvals_upper = yvals.max(axis=1)
    ax.plot(xvals, yvals_mean, label=label)
    ax.fill_between(xvals, yvals_lower, yvals_upper, alpha=0.2)


def naive_loss(n_features: int, d_mlp: int, p: float, bias: bool, embed: str) -> float:
    if embed == "random":
        if bias:
            return (n_features - d_mlp) * (8 - 3 * p) * p / 48
        else:
            return (n_features - d_mlp) * p / 6
    elif embed == "trained":
        if bias:
            return (n_features - d_mlp) * (4 - 3 * p) * p / 48
        else:
            return (n_features - d_mlp) * p / 12
    else:
        raise ValueError(f"Unknown embedding type {embed}")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out"
    os.makedirs(out_dir, exist_ok=True)
    n_instances = 20
    n_features = 100
    d_mlp = 50
    d_embed = None
    n_steps = None
    # Scale d_embed
    p = 0.01
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(f"Loss scaling with d_embed. Using {n_instances} instances")
    d_embeds = [2_000, 1000, 500, 200, 100, 50]
    bias = False
    for h, loss_type in enumerate(["readoff", "resid"]):
        for i, embed in enumerate(["random", "trained"]):
            print(f"Quadrant {bias=} and {embed=}")
            losses = {}
            fixed_random_embedding = embed == "random"
            fixed_identity_embedding = embed == "identity"
            for n_steps in [10_000, 1000, 100]:
                losses[n_steps] = {}
                for d_embed in d_embeds:
                    print(f"Run {n_steps} steps, {d_embed} d_embed")
                    losses[n_steps][d_embed] = train_on_test_data(
                        n_instances=n_instances,
                        n_steps=n_steps,
                        d_embed=d_embed,
                        p=p,
                        bias=bias,
                        n_features=n_features,
                        d_mlp=d_mlp,
                        fixed_random_embedding=fixed_random_embedding,
                        fixed_identity_embedding=fixed_identity_embedding,
                        loss_type=loss_type,  # type: ignore
                    )
            title_str = f"W_E={embed}_{bias=}_{n_features=}_{d_mlp=}_{p=}_{loss_type=}"
            with open(out_dir / f"losses_scale_embed_{title_str}.json", "w") as f:
                json.dump(losses, f)
            # Make plot
            ax = axes[h, i]  # type: ignore
            ax.set_title(title_str, fontsize=8)
            naive_losses = naive_loss(n_features, d_mlp, p, bias, embed)
            ax.axhline(
                naive_losses, color="k", linestyle="--", label=f"Naive loss {naive_losses:.2e}"
            )
            for n_steps in losses:
                plot_loss_curve(ax, losses[n_steps], label=f"{n_steps} steps")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("d_embed")
            ax.set_ylabel("Loss L")
            ax.legend(loc="upper center")
            fig.savefig(out_dir / "loss_scaling_resid_mlp_training_d_embed.png")
    print("Saved plot to", out_dir / "loss_scaling_resid_mlp_training_d_embed.png")
    plt.show()

    # Scale p
    d_embed = 10 * n_features
    loss_type = "readoff"
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(f"Loss scaling with p. Using {n_instances} instances")
    ps = np.array([0.01, 0.03, 0.05, 0.075, 0.1, 0.5, 1.0])
    for bias in [False, True]:
        for i, embed in enumerate(["random", "trained"]):
            losses = {}
            fixed_random_embedding = embed == "random"
            fixed_identity_embedding = embed == "identity"
            print(f"Quadrant {bias=} and {embed=}")
            for n_steps in [10_000, 1000, 100]:
                losses[n_steps] = {}
                for p in ps:
                    print(f"Run {n_steps} steps, {p} p")
                    losses[n_steps][p] = train_on_test_data(
                        n_instances=n_instances,
                        n_steps=n_steps,
                        d_embed=d_embed,
                        p=p,
                        bias=bias,
                        n_features=n_features,
                        d_mlp=d_mlp,
                        fixed_random_embedding=fixed_random_embedding,
                        fixed_identity_embedding=fixed_identity_embedding,
                        loss_type=loss_type,
                    )
            title_str = f"W_E={embed}_{bias=}_{n_features=}_{d_mlp=}_{d_embed=}_{loss_type=}"
            with open(out_dir / f"losses_scale_p_{title_str}.json", "w") as f:
                json.dump(losses, f)
            # Make plot
            ax = axes[int(bias), i]  # type: ignore
            ax.set_title(title_str, fontsize=8)
            scaled_naive_losses = [naive_loss(n_features, d_mlp, p, bias, embed) / p for p in ps]
            ax.plot(ps, scaled_naive_losses, color="k", linestyle="--", label="Naive loss (scaled)")
            for n_steps in losses:
                scaled_loss = {
                    p: {i: losses[n_steps][p][i] / p for i in range(n_instances)} for p in ps
                }
                plot_loss_curve(ax, scaled_loss, label=f"{n_steps} steps")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("p")
            ax.set_ylabel("Scaled loss L / p")
            ax.legend(loc="upper center")
            fig.savefig(out_dir / "loss_scaling_resid_mlp_training_p.png")
    print("Saved plot to", out_dir / "loss_scaling_resid_mlp_training_p.png")
    plt.show()
