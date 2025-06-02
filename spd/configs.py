"""Config classes of various types"""

from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from spd.log import logger
from spd.types import ModelPath, Probability


class TMSTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = Field(
        default="tms",
        description="Task identifier for TMS",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for TMS training",
    )
    pretrained_model_path: ModelPath = Field(
        ...,
        description="Local path or wandb reference to the pretrained TMS model (e.g. 'wandb:spd-tms/runs/si0zbfxf')",
    )


class ResidualMLPTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["residual_mlp"] = Field(
        default="residual_mlp",
        description="Identifier for the residual-MLP decomposition task",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for residual-MLP training",
    )
    pretrained_model_path: ModelPath = Field(
        ...,
        description="Local path or wandb reference to the pretrained residual-MLP model (e.g. 'wandb:spd-resid-mlp/runs/j9kmavzi')",
    )
    # TODO: Move to main config when supported by TMS
    # List of fnmatch patterns for nn.Linear modules to decompose


class LMTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["lm"] = Field(
        default="lm",
        description="Identifier for the language-model decomposition task",
    )
    max_seq_len: PositiveInt = Field(
        default=512,
        description="Maximum sequence length to truncate or pad inputs to",
    )
    buffer_size: PositiveInt = Field(
        default=1000,
        description="Buffered sample count for streaming dataset shuffling",
    )
    dataset_name: str = Field(
        default="lennart-finke/SimpleStories",
        description="HuggingFace dataset identifier to use for the LM task",
    )
    column_name: str = Field(
        default="story",
        description="Dataset column that contains the text to train on",
    )
    train_data_split: str = Field(
        default="train",
        description="Name of the dataset split used for training",
    )
    eval_data_split: str = Field(
        default="test",
        description="Name of the dataset split used for evaluation",
    )
    # TODO: Move to main config when supported by TMS
    # List of fnmatch patterns for nn.Linear modules to decompose


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # --- WandB
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name (set to None to disable WandB logging)",
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="Explicit name for the WandB run (None generates an automatic name)",
    )
    wandb_run_name_prefix: str = Field(
        default="",
        description="Prefix prepended to an auto-generated WandB run name",
    )

    # --- General ---
    seed: int = Field(default=0, description="Random seed for reproducibility")
    unit_norm_matrices: bool = Field(
        default=False,
        description="Whether to renormalise each A matrix so every column has unit 2-norm",
    )
    m: PositiveInt = Field(
        ...,
        description="Rank of the decomposition / number of components per layer",
    )
    n_random_masks: PositiveInt = Field(
        ...,
        description="Number of random masks to sample when using random-mask reconstruction loss",
    )
    n_gate_hidden_neurons: PositiveInt | None = Field(
        default=None,
        description="Hidden dimension for the gate MLP; if None, use a single-layer gate",
    )
    init_from_target_model: bool = Field(
        default=False,
        description="Initialise SPD components directly from the target model's weights",
    )
    target_module_patterns: list[str] = Field(
        ...,
        description="List of fnmatch-style patterns that select nn.Linear / nn.Embedding modules to decompose",
    )

    # --- Loss Coefficients
    param_match_coeff: NonNegativeFloat | None = Field(
        default=1.0,
        description="Coefficient for matching parameters between components and target weights",
    )
    masked_recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for reconstruction loss with a deterministic binary mask",
    )
    random_mask_recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for reconstruction loss with random binary masks",
    )
    layerwise_recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for per-layer reconstruction loss (deterministic mask)",
    )
    layerwise_random_recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for per-layer reconstruction loss with random masks",
    )
    lp_sparsity_coeff: NonNegativeFloat = Field(
        ...,
        description="Coefficient for L_p sparsity penalty applied to the gating activations",
    )
    schatten_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for Schatten-norm regularisation (LM only)",
    )
    embedding_recon_coeff: float | None = Field(
        default=None,
        description="Coefficient for additional embedding reconstruction loss (LM only)",
    )
    is_embed_unembed_recon: bool = Field(
        default=False,
        description="If True, apply embedding reconstruction jointly to embed & unembed matrices",
    )
    pnorm: PositiveFloat = Field(
        ...,
        description="The p-value used for the L_p sparsity loss",
    )
    output_loss_type: Literal["mse", "kl"] = Field(
        ...,
        description="Metric used to measure reconstruction error between model outputs and targets",
    )

    # --- Training ---
    lr: PositiveFloat = Field(..., description="Learning rate for optimiser")
    steps: PositiveInt = Field(..., description="Total number of optimisation steps")
    batch_size: PositiveInt = Field(..., description="Mini-batch size used for optimisation")
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = Field(
        default="constant",
        description="Type of learning-rate schedule to apply",
    )
    lr_exponential_halflife: PositiveFloat | None = Field(
        default=None,
        description="Half-life parameter when using an exponential LR schedule",
    )
    lr_warmup_pct: Probability = Field(
        default=0.0,
        description="Fraction of total steps to linearly warm up the learning rate",
    )
    n_eval_steps: PositiveInt = Field(
        ...,
        description="Frequency (in optimisation steps) at which to run evaluation",
    )

    # --- Logging & Saving ---
    image_freq: PositiveInt | None = Field(
        default=None,
        description="Interval (in steps) at which to log diagnostic images to WandB",
    )
    image_on_first_step: bool = Field(
        default=True,
        description="Whether to log images at optimisation step 0",
    )
    print_freq: PositiveInt = Field(
        ...,
        description="Interval (in steps) at which to print training metrics to stdout",
    )
    save_freq: PositiveInt | None = Field(
        default=None,
        description="Interval (in steps) at which to save model checkpoints (None disables saving)",
    )
    log_ce_losses: bool = Field(
        default=False,
        description="If True, additionally track cross-entropy losses during training",
    )

    # --- Pretrained model info ---
    pretrained_model_class: str | None = Field(
        default=None,
        description="Fully-qualified class name of the pretrained model to load (e.g. 'transformers.LlamaForCausalLM')",
    )
    pretrained_model_name: str | None = Field(
        default=None,
        description="Model identifier or path recognised by the class' .from_pretrained() method",
    )
    pretrained_model_output_attr: str | None = Field(
        default=None,
        description="Name of the attribute on the forward output that contains logits or activations",
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Name or path of the tokenizer to use when loading an LM",
    )

    # --- Task Specific ---
    task_config: TMSTaskConfig | ResidualMLPTaskConfig | LMTaskConfig = Field(
        ...,
        discriminator="task_name",
        description="Nested task-specific configuration selected by the `task_name` discriminator",
    )

    DEPRECATED_CONFIG_KEYS: ClassVar[list[str]] = []
    RENAMED_CONFIG_KEYS: ClassVar[dict[str, str]] = {}

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""
        for key in list(config_dict.keys()):
            val = config_dict[key]
            if key in cls.DEPRECATED_CONFIG_KEYS:
                logger.warning(f"{key} is deprecated, but has value: {val}. Removing from config.")
                del config_dict[key]
            elif key in cls.RENAMED_CONFIG_KEYS:
                logger.info(f"Renaming {key} to {cls.RENAMED_CONFIG_KEYS[key]}")
                config_dict[cls.RENAMED_CONFIG_KEYS[key]] = val
                del config_dict[key]
        return config_dict

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Warn if neither masked_recon_coeff nor lp_sparsity_coeff is set
        if not self.masked_recon_coeff and not self.lp_sparsity_coeff:
            logger.warning("Neither masked_recon_coeff nor lp_sparsity_coeff is set")

        # If any of the coeffs are 0, raise a warning
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.masked_recon_coeff == 0:
            logger.warning(f"masked_recon_coeff {msg}")
        if self.lp_sparsity_coeff == 0:
            logger.warning(f"lp_sparsity_coeff {msg}")
        if self.param_match_coeff == 0:
            logger.warning(f"param_match_coeff {msg}")

        # Check that lr_exponential_halflife is not None if lr_schedule is "exponential"
        if self.lr_schedule == "exponential":
            assert self.lr_exponential_halflife is not None, (
                "lr_exponential_halflife must be set if lr_schedule is exponential"
            )
        # Schatten norm schould be null unless the model is an LM
        if self.task_config.task_name != "lm":
            assert self.schatten_coeff is None, (
                "schatten_coeff should be null unless the model is an LM"
            )
        return self
