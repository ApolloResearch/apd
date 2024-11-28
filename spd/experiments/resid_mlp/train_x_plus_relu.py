"""Trains a residual linear model on one-hot input vectors."""

import torch
import wandb

from spd.experiments.resid_mlp.train_resid_mlp import Config, run_train
from spd.utils import set_seed

wandb.require("core")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        seed=0,
        label_fn_seed=0,
        n_instances=20,
        n_features=100,
        d_embed=1000,
        d_mlp=10,
        n_layers=5,
        act_fn_name="relu",
        apply_output_act_fn=False,
        label_type="act_plus_resid",
        data_generation_type="at_least_zero_active",
        use_trivial_label_coeffs=True,
        in_bias=False,
        out_bias=False,
        feature_probability=0.01,
        importance_val=1,
        batch_size=2048,
        steps=1000,
        print_freq=100,
        lr=3e-3,
        lr_schedule="cosine",
        fixed_random_embedding=False,
        fixed_identity_embedding=False,
    )

    set_seed(config.seed)

    run_train(config, device)
