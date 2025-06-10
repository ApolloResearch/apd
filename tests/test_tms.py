import torch

from spd.configs import Config, TMSTaskConfig
from spd.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.experiments.tms.models import TMSModel, TMSModelConfig
from spd.experiments.tms.train_tms import TMSTrainConfig, get_model_and_dataloader, train
from spd.run_spd import optimize
from spd.utils import set_seed


def test_tms_decomposition_happy_path() -> None:
    """Test that SPD decomposition works on a TMS model."""
    set_seed(0)
    device = "cpu"

    # Create a TMS model config similar to the one in tms_config.yaml
    tms_model_config = TMSModelConfig(
        n_features=5,
        n_hidden=2,
        n_hidden_layers=1,
        tied_weights=True,
        init_bias_to_zero=False,
        device=device,
    )

    # Create config similar to tms_config.yaml
    config = Config(
        # WandB
        wandb_project=None,  # Disable wandb for testing
        wandb_run_name=None,
        wandb_run_name_prefix="",
        # General
        seed=0,
        C=10,  # Smaller C for faster testing
        n_mask_samples=1,
        n_ci_mlp_neurons=8,
        target_module_patterns=["linear1", "linear2", "hidden_layers.0"],
        # Loss Coefficients
        faithfulness_coeff=1.0,
        ci_masked_recon_coeff=None,
        stochastic_ci_masked_recon_coeff=1.0,
        layerwise_ci_masked_recon_coeff=1e-1,
        layerwise_stochastic_ci_masked_recon_coeff=1.0,
        importance_loss_coeff=3e-3,
        schatten_coeff=None,
        embedding_recon_coeff=None,
        is_embed_unembed_recon=False,
        pnorm=2.0,
        output_loss_type="mse",
        # Training
        lr=1e-3,
        batch_size=4,
        steps=3,  # Run only a few steps for the test
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.0,
        n_eval_steps=1,
        # Logging & Saving
        image_freq=None,
        image_on_first_step=True,
        print_freq=2,
        save_freq=None,
        log_ce_losses=False,
        # Pretrained model info
        pretrained_model_class="spd.experiments.tms.models.TMSModel",
        pretrained_model_path=None,
        pretrained_model_name_hf=None,
        pretrained_model_output_attr=None,
        tokenizer_name=None,
        # Task Specific
        task_config=TMSTaskConfig(
            task_name="tms",
            feature_probability=0.05,
            data_generation_type="at_least_zero_active",
        ),
    )

    # Create a pretrained model
    target_model = TMSModel(config=tms_model_config).to(device)
    target_model.eval()

    assert isinstance(config.task_config, TMSTaskConfig)
    # Create dataset
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        data_generation_type=config.task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=None,
    )

    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    tied_weights = None
    if target_model.config.tied_weights:
        tied_weights = [("linear1", "linear2")]

    # Run optimize function
    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=None,
        plot_results_fn=None,
        tied_weights=tied_weights,
    )

    # The test passes if optimize runs without errors
    print("TMS SPD optimization completed successfully")

    # Basic assertion to ensure the test ran
    assert True, "Test completed successfully"


def test_train_tms_happy_path():
    """Test training a TMS model from scratch."""
    device = "cpu"
    set_seed(0)
    # Set up a small configuration
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_hidden_layers=0,
            tied_weights=False,
            init_bias_to_zero=False,
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

    # Run training
    train(
        model,
        dataloader,
        importance=1.0,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

    # The test passes if training runs without errors
    print("TMS training completed successfully")
    assert True, "Test completed successfully"


def test_tms_train_fixed_identity():
    """Check that hidden layer is identity before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_hidden_layers=2,
            tied_weights=False,
            init_bias_to_zero=False,
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

    eye = torch.eye(config.tms_model_config.n_hidden, device=device)

    assert model.hidden_layers is not None
    # Assert that this is an identity matrix
    initial_hidden = model.hidden_layers[0].weight.data.clone()
    assert torch.allclose(initial_hidden, eye), "Initial hidden layer is not identity"

    train(
        model,
        dataloader,
        importance=1.0,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

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
            n_hidden_layers=2,
            tied_weights=False,
            init_bias_to_zero=False,
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

    train(
        model,
        dataloader,
        importance=1.0,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

    # Assert that the hidden layers are unchanged
    assert torch.allclose(model.hidden_layers[0].weight.data, initial_hidden), (
        "Hidden layer changed"
    )
