from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config, TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSModelConfig, TMSSPDModel, TMSSPDModelConfig
from spd.experiments.tms.tms_decomposition import init_spd_model_from_target_model
from spd.experiments.tms.train_tms import TMSTrainConfig, get_model_and_dataloader, train
from spd.module_utils import get_nested_module_attr
from spd.run_spd import optimize
from spd.utils import DatasetGeneratedDataLoader, SparseFeatureDataset, set_seed

# Create a simple TMS config that we can use in multiple tests
TMS_TASK_CONFIG = TMSTaskConfig(
    task_name="tms",
    feature_probability=0.5,
    pretrained_model_path=Path(""),  # We'll create this later
)


def tms_spd_happy_path(config: Config, n_hidden_layers: int = 0):
    set_seed(0)
    device = "cpu"
    assert isinstance(config.task_config, TMSTaskConfig)

    # For our pretrained model, just use a randomly initialized TMS model
    tms_model_config = TMSModelConfig(
        n_features=5,
        n_hidden=2,
        n_hidden_layers=n_hidden_layers,
        device=device,
    )
    target_model = TMSModel(config=tms_model_config)

    tms_spd_model_config = TMSSPDModelConfig(**tms_model_config.model_dump(mode="json"), m=config.m)
    model = TMSSPDModel(config=tms_spd_model_config)
    # Randomly initialize the bias for the pretrained model
    target_model.b_final.data = torch.randn_like(target_model.b_final.data)
    # Manually set the bias for the SPD model from the bias in the pretrained model
    model.b_final.data[:] = target_model.b_final.data.clone()
    model.b_final.requires_grad = False

    dataset = SparseFeatureDataset(
        n_instances=target_model.config.n_instances,
        n_features=target_model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        data_generation_type=config.task_config.data_generation_type,
        value_range=(0.0, 1.0),
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    # Pick an arbitrary parameter to check that it changes
    initial_param = model.linear1.A.clone().detach()

    param_names = ["linear1", "linear2"]
    if model.hidden_layers is not None:
        for i in range(len(model.hidden_layers)):
            param_names.append(f"hidden_layers.{i}")

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

    assert not torch.allclose(initial_param, model.linear1.A), (
        "Model A matrix should have changed after optimization"
    )


def test_tms_happy_path():
    config = Config(
        m=10,
        random_mask_recon_coeff=1,
        n_random_masks=2,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        lp_sparsity_coeff=0.01,
        pnorm=0.9,
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_happy_path(config)


def test_train_tms_happy_path():
    device = "cpu"
    set_seed(0)
    # Set up a small configuration
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_instances=2,
            n_hidden_layers=0,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=5,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out = model(batch)
    initial_loss = torch.mean((labels.abs() - initial_out) ** 2)

    train(model, dataloader, steps=config.steps, print_freq=1000, log_wandb=False)

    # Calculate final loss
    final_out = model(batch)
    final_loss = torch.mean((labels.abs() - final_out) ** 2)

    # Assert that the final loss is lower than the initial loss
    assert final_loss < initial_loss, (
        f"Final loss ({final_loss:.2e}) is not lower than initial loss ({initial_loss:.2e})"
    )


def test_tms_train_fixed_identity():
    """Check that hidden layer is identity before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_instances=2,
            n_hidden_layers=2,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=True,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    eye = torch.eye(config.tms_model_config.n_hidden, device=device).expand(
        config.tms_model_config.n_instances, -1, -1
    )

    assert model.hidden_layers is not None
    # Assert that this is an identity matrix
    initial_hidden = model.hidden_layers[0].weight.data.clone()
    assert torch.allclose(initial_hidden, eye), "Initial hidden layer is not identity"

    train(model, dataloader, steps=config.steps, print_freq=1000, log_wandb=False)

    # Assert that the hidden layers remains identity
    assert torch.allclose(model.hidden_layers[0].weight.data, eye), "Hidden layer changed"


def test_tms_train_fixed_random():
    """Check that hidden layer is random before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_instances=2,
            n_hidden_layers=2,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=True,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    assert model.hidden_layers is not None
    initial_hidden = model.hidden_layers[0].weight.data.clone()

    train(model, dataloader, steps=config.steps, print_freq=1000, log_wandb=False)

    # Assert that the hidden layers are unchanged
    assert torch.allclose(model.hidden_layers[0].weight.data, initial_hidden), (
        "Hidden layer changed"
    )


def test_tms_equivalent_to_raw_model() -> None:
    device = "cpu"
    set_seed(0)
    tms_config = TMSModelConfig(
        n_instances=2,
        n_features=3,
        n_hidden=2,
        n_hidden_layers=1,
        device=device,
    )

    target_model = TMSModel(config=tms_config).to(device)

    # Create the SPD model
    tms_spd_config = TMSSPDModelConfig(
        **tms_config.model_dump(),
        m=3,  # Small m for testing
    )
    spd_model = TMSSPDModel(config=tms_spd_config).to(device)

    # Init all params to random values
    for param in spd_model.parameters():
        param.data = torch.randn_like(param.data)

    # Copy the subnetwork params from the SPD model to the target model
    target_model.linear1.weight.data[:, :, :] = spd_model.linear1.weight.data
    if target_model.hidden_layers is not None:
        for i in range(target_model.config.n_hidden_layers):
            target_layer: Tensor = get_nested_module_attr(target_model, f"hidden_layers.{i}.weight")
            spd_layer: Tensor = get_nested_module_attr(spd_model, f"hidden_layers.{i}.weight")
            target_layer.data[:, :, :] = spd_layer.data

    # Also copy the bias
    target_model.b_final.data[:, :] = spd_model.b_final.data

    # Create a random input
    batch_size = 4
    input_data: Float[torch.Tensor, "batch n_instances n_features"] = torch.rand(
        batch_size, tms_config.n_instances, tms_config.n_features, device=device
    )

    with torch.inference_mode():
        # Forward pass on target model
        target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
        target_out, target_cache = target_model.run_with_cache(
            input_data, names_filter=target_cache_filter
        )
        # Forward pass with all subnetworks
        spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
        out, spd_cache = spd_model.run_with_cache(input_data, names_filter=spd_cache_filter)

    # Assert outputs are the same
    assert torch.allclose(target_out, out, atol=1e-6), "Outputs do not match"

    # Assert that all post-acts are the same
    target_post_weight_acts = {k: v for k, v in target_cache.items() if k.endswith(".hook_post")}
    spd_post_weight_acts = {k: v for k, v in spd_cache.items() if k.endswith(".hook_post")}
    for key_name in target_post_weight_acts:
        assert torch.allclose(
            target_post_weight_acts[key_name], spd_post_weight_acts[key_name], atol=1e-6
        ), f"post-acts do not match at layer {key_name}"


def test_init_tms_spd_model_from_target() -> None:
    """Test that initializing an SPD model from a target model results in identical outputs."""
    device = "cpu"
    set_seed(0)

    # Create target model with no hidden layers (as per current limitation)
    tms_config = TMSModelConfig(
        n_instances=2,
        n_features=3,
        n_hidden=2,
        n_hidden_layers=0,
        device=device,
    )
    target_model = TMSModel(config=tms_config).to(device)

    # Create the SPD model with m equal to n_features
    tms_spd_config = TMSSPDModelConfig(
        **tms_config.model_dump(),
        m=tms_config.n_features,  # Must match n_features for initialization
    )
    spd_model = TMSSPDModel(config=tms_spd_config).to(device)

    init_spd_model_from_target_model(spd_model, target_model, m=tms_config.n_features)
    # Also copy the bias
    spd_model.b_final.data[:, :] = target_model.b_final.data

    # Create a random input
    batch_size = 4
    input_data: Float[Tensor, "batch n_instances n_features"] = torch.rand(
        batch_size, tms_config.n_instances, tms_config.n_features, device=device
    )

    with torch.inference_mode():
        target_out = target_model(input_data)
        spd_out = spd_model(input_data)

    assert torch.allclose(spd_model.linear1.weight, target_model.linear1.weight), (
        "Weights do not match"
    )

    assert torch.allclose(target_out, spd_out), "Outputs after initialization do not match"
