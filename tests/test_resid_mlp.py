from pathlib import Path

import torch

from spd.configs import Config
from spd.experiments.resid_mlp.models import (
    ResidualMLP,
    ResidualMLPConfig,
    ResidualMLPSPDConfig,
    ResidualMLPSPDModel,
    ResidualMLPTaskConfig,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import optimize
from spd.utils import DatasetGeneratedDataLoader, set_seed

# Create a simple ResidualMLP config that we can use in multiple tests
RESID_MLP_TASK_CONFIG = ResidualMLPTaskConfig(
    task_name="residual_mlp",
    feature_probability=0.333,
    data_generation_type="at_least_zero_active",
    pretrained_model_path=Path(),  # We'll create this later
)


def test_resid_mlp_decomposition_happy_path() -> None:
    # Just noting that this test will only work on 98/100 seeds. So it's possible that future
    # changes will break this test.
    set_seed(0)
    resid_mlp_config = ResidualMLPConfig(
        n_instances=2,
        n_features=3,
        d_embed=2,
        d_mlp=3,
        n_layers=1,
        act_fn_name="relu",
        apply_output_act_fn=False,
        in_bias=True,
        out_bias=True,
    )

    device = "cpu"
    config = Config(
        seed=0,
        m=2,
        random_mask_recon_coeff=1,
        n_random_masks=2,
        param_match_coeff=1.0,
        masked_recon_coeff=1,
        lp_sparsity_coeff=1.0,
        pnorm=0.9,
        lr=1e-3,
        batch_size=32,
        steps=50,  # Run only a few steps for the test
        print_freq=2,
        image_freq=5,
        save_freq=None,
        lr_warmup_pct=0.01,
        lr_schedule="cosine",
        task_config=RESID_MLP_TASK_CONFIG,
    )

    assert isinstance(config.task_config, ResidualMLPTaskConfig)
    # Create a pretrained model
    target_model = ResidualMLP(config=resid_mlp_config).to(device)

    # Create the SPD model
    spd_config = ResidualMLPSPDConfig(**resid_mlp_config.model_dump(), m=config.m)
    model = ResidualMLPSPDModel(config=spd_config).to(device)

    # Use the pretrained model's embedding matrices and don't train them further
    model.W_E.data[:, :] = target_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False
    model.W_U.data[:, :] = target_model.W_U.data.detach().clone()
    model.W_U.requires_grad = False

    # Copy the biases from the target model to the SPD model and set requires_grad to False
    for i in range(resid_mlp_config.n_layers):
        if resid_mlp_config.in_bias:
            model.layers[i].bias1.data[:, :] = target_model.layers[i].bias1.data.detach().clone()
            model.layers[i].bias1.requires_grad = False
        if resid_mlp_config.out_bias:
            model.layers[i].bias2.data[:, :] = target_model.layers[i].bias2.data.detach().clone()
            model.layers[i].bias2.requires_grad = False

    # Create dataset and dataloader
    dataset = ResidualMLPDataset(
        n_instances=model.n_instances,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type="at_least_zero_active",
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Calculate initial loss
    with torch.inference_mode():
        batch, _ = next(iter(dataloader))
        initial_out = model(batch)
        labels = target_model(batch)
        initial_loss = torch.mean((labels - initial_out) ** 2).item()

    param_names = []
    for i in range(target_model.config.n_layers):
        param_names.append(f"layers.{i}.mlp_in")
        param_names.append(f"layers.{i}.mlp_out")
    # Run optimize function
    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        out_dir=None,
        plot_results_fn=None,
    )

    # Calculate final loss
    with torch.inference_mode():
        final_out = model(batch)
        final_loss = torch.mean((labels - final_out) ** 2).item()

    print(f"Final loss: {final_loss}, initial loss: {initial_loss}")
    # Assert that the final loss is lower than the initial loss
    assert final_loss < initial_loss + 1e-3, (
        f"Expected final loss to be lower than initial loss, but got {final_loss} >= {initial_loss}"
    )

    # Show that W_E is still the same as the target model's W_E
    assert torch.allclose(model.W_E, target_model.W_E, atol=1e-6)
