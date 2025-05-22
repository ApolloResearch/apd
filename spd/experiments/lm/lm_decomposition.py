"""Language Model decomposition script."""

from datetime import datetime
from pathlib import Path

import einops
import fire
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from jaxtyping import Bool, Float
from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config, LMTaskConfig
from spd.experiments.lm.component_viz import (
    component_activation_statistics,
    plot_mean_component_activation_counts,
)
from spd.experiments.lm.models import EmbeddingComponent, LinearComponentWithBias, SSModel
from spd.log import logger
from spd.models.components import Gate, GateMLP
from spd.run_spd import (
    _calc_param_mse,
    calc_component_acts,
    calc_mask_l_zero,
    calc_masks,
    calc_random_masks,
    get_common_run_name_suffix,
)
from spd.utils import (
    get_device,
    get_lr_schedule_fn,
    get_lr_with_warmup,
    load_config,
    set_seed,
)
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(
    config: Config,
    model_size: str,
    max_seq_len: int,
) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"_lm{model_size}_seq{max_seq_len}"
    return config.wandb_run_name_prefix + run_suffix


def plot_lm_results(
    mean_component_activation_counts: dict[str, Float[Tensor, " m"]],
) -> dict[str, plt.Figure]:
    """Plotting function for LM decomposition."""
    fig_dict: dict[str, plt.Figure] = {}

    fig_dict["mean_component_activation_counts"] = plot_mean_component_activation_counts(
        mean_component_activation_counts=mean_component_activation_counts,
    )
    return fig_dict


def calc_recon_mse_lm(
    out1: Float[Tensor, "batch pos vocab"],
    out2: Float[Tensor, "batch pos vocab"],
) -> Float[Tensor, ""]:
    """Calculate the Mean Squared Error reconstruction loss for LM logits."""
    assert out1.shape == out2.shape
    # Mean over batch and sequence length, sum over vocab
    return ((out1 - out2) ** 2).sum(dim=-1).mean()


def calc_kl_divergence_lm(
    pred: Float[Tensor, "batch pos vocab"],
    target: Float[Tensor, "batch pos vocab"],
) -> Float[Tensor, ""]:
    """Calculate the KL divergence between two logits."""
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    kl = F.kl_div(log_q, p, reduction="none")  # P · (log P − log Q)
    return kl.sum(dim=-1).mean()  # Σ_vocab / (batch·seq)


def calc_param_match_loss_lm(
    components: dict[str, LinearComponentWithBias | EmbeddingComponent],
    target_model: Llama,
    n_params: int,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE loss between component parameters (A@B + bias) and target parameters."""
    target_params: dict[str, Float[Tensor, "d_in d_out"]] = {}
    component_params: dict[str, Float[Tensor, "d_in d_out"]] = {}

    for comp_name, component in components.items():
        component_params[comp_name] = component.weight
        submodule = target_model.get_submodule(comp_name)
        if isinstance(submodule, nn.Linear):
            target_params[comp_name] = submodule.weight.T
        elif isinstance(submodule, nn.Embedding):
            target_params[comp_name] = submodule.weight
        else:
            raise ValueError(f"Submodule {comp_name} is not a nn.Linear or nn.Embedding")
        assert component_params[comp_name].shape == target_params[comp_name].shape

    param_mse = _calc_param_mse(
        params1=component_params,
        params2=target_params,
        n_params=n_params,
        device=device,
    )
    return param_mse


def calc_layerwise_recon_loss_lm(
    model: SSModel,
    batch: Float[Tensor, "batch pos"],
    device: str,
    components: dict[str, LinearComponentWithBias | EmbeddingComponent],
    masks: list[dict[str, Float[Tensor, "batch pos m"]]],
    target_out: Float[Tensor, "batch pos vocab"],
) -> Float[Tensor, ""]:
    """Calculate the recon loss when augmenting the model one (masked) component at a time."""
    total_loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        for component_name, component in components.items():
            module_name = component_name.replace("-", ".")
            modified_out, _ = model.forward_with_component(
                batch,
                module_name=module_name,
                component=component,
                mask=mask_info[component_name],
            )
            loss = calc_kl_divergence_lm(pred=modified_out, target=target_out)
            total_loss += loss
    n_modified_components = len(masks[0])
    return total_loss / (n_modified_components * len(masks))


def calc_lp_sparsity_loss_lm(
    relud_masks: dict[str, Float[Tensor, "batch pos m"]], pnorm: float
) -> Float[Tensor, ""]:
    """Calculate the Lp sparsity loss on the attributions.

    Args:
        relud_masks: Dictionary of relu masks for each layer.
        pnorm: The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss.
    """
    # Initialize with zeros matching the shape of first mask
    total_loss = torch.zeros_like(next(iter(relud_masks.values())))

    for layer_relud_mask in relud_masks.values():
        total_loss = total_loss + layer_relud_mask**pnorm

    # Sum over the m dimension and mean over the batch and pos dimensions
    return total_loss.sum(dim=-1).mean(dim=[0, 1])


def calc_schatten_loss_lm(
    relud_masks: dict[str, Float[Tensor, "batch pos m"]],
    pnorm: float,
    components: dict[str, LinearComponentWithBias | EmbeddingComponent],
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the Schatten loss on the active components.

    The Schatten loss is calculated as:
        L = Σ_{components} mean(relu_mask^pnorm · (||A||_2^2 + ||B||_2^2))

    where:
        - relu_mask is the activation mask for each component
        - pnorm is the power to raise the mask to
        - A and B are the component matrices
        - ||·||_2 is the L2 norm

    Args:
        relud_masks: Dictionary of relu masks for each layer.
        pnorm: The pnorm to use for the sparsity loss. Must be positive.
        components: Dictionary of components for each layer. All components must be LinearComponentWithBias.
        device: The device to compute the loss on.

    Returns:
        The Schatten loss as a scalar tensor.
    """

    total_loss = torch.tensor(0.0, device=device)
    for component_name, component in components.items():
        A_norms = component.A.square().sum(dim=-2)
        B_norms = component.B.square().sum(dim=-1)
        schatten_norms = A_norms + B_norms
        loss = einops.einsum(
            relud_masks[component_name] ** pnorm, schatten_norms, "... m, m -> ..."
        )
        total_loss += loss.mean()
    return total_loss


def calc_embedding_recon_loss_lm(
    model: SSModel,
    batch: Float[Tensor, "batch pos"],
    component: EmbeddingComponent,
    masks: dict[str, Float[Tensor, "batch pos m"]] | None = None,
    unembed: bool = False,
) -> Float[Tensor, ""]:
    """
    Reconstruction loss that directly compares the outputs of the (optionally masked)
    ``EmbeddingComponent``(s) to the outputs of the original ``nn.Embedding`` modules.

    If ``unembed`` is ``True``, both the APD-augmented embedding output and the target embedding
    output are unembedded using the ``lm_head`` module, and the KL divergence is used as the loss.

    If ``unembed`` is ``False``, the loss is the MSE between the APD-augmented embedding output
    and the target embedding output is used as the loss.
    """
    module_name = "transformer.wte"

    # --- original embedding output --------------------------------------------------------- #
    orig_module = model.model.get_submodule(module_name)
    assert isinstance(orig_module, nn.Embedding), (
        f"Module {module_name} expected to be nn.Embedding, got {type(orig_module)}"
    )
    target_out: Float[Tensor, "batch pos d_emb"] = orig_module(batch)

    # --- APD-augmented embedding output ---------------------------------------------------- #
    if masks is not None:
        component.mask = masks[module_name]
    apd_out: Float[Tensor, "batch pos d_emb"] = component(batch)  # type: ignore[arg-type]
    component.mask = None

    if unembed:
        target_out_unembed = model.model.lm_head(target_out)
        apd_out_unembed = model.model.lm_head(apd_out)
        loss = calc_kl_divergence_lm(pred=apd_out_unembed, target=target_out_unembed)
    else:
        loss = ((apd_out - target_out) ** 2).sum(dim=-1).mean()

    return loss


def create_embed_mask_sample_table(
    masks: dict[str, Float[Tensor, "batch pos m"]],
) -> wandb.Table | None:
    """Create a wandb table visualizing embedding mask values.

    Args:
        masks: Dictionary of masks for each component.

    Returns:
        A wandb Table object or None if transformer.wte not in masks.
    """
    if "transformer.wte" not in masks:
        return None

    # Create a 20x10 table for wandb
    table_data = []
    # Add "Row Name" as the first column
    component_names = ["TokenSample"] + ["CompVal" for _ in range(10)]

    for i, ma in enumerate(masks["transformer.wte"][0, :20]):
        active_values = ma[ma > 0.1].tolist()
        # Cap at 10 components
        active_values = active_values[:10]
        formatted_values = [f"{val:.2f}" for val in active_values]
        # Pad with empty strings if fewer than 10 components
        while len(formatted_values) < 10:
            formatted_values.append("")
        # Add row name as the first element
        table_data.append([f"{i}"] + formatted_values)

    return wandb.Table(data=table_data, columns=component_names)


def optimize_lm(
    model: SSModel,
    config: Config,
    device: str,
    train_loader: DataLoader[Float[Tensor, "batch pos"]],
    eval_loader: DataLoader[Float[Tensor, "batch pos"]],
    n_eval_steps: int,
    out_dir: Path | None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponentWithBias | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    component_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []
    for name, component in components.items():
        component_params.extend(list(component.parameters()))
        gate_params.extend(list(gates[name].parameters()))

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimizer = optim.AdamW(component_params + gate_params, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    n_params = 0
    for module_name in components:
        weight = model.model.get_parameter(module_name + ".weight")
        n_params += weight.numel()

    log_data = {}
    data_iter = iter(train_loader)

    alive_components: dict[str, Bool[Tensor, " m"]] = {
        layer_name: torch.zeros(config.m, device=device).bool() for layer_name in components
    }

    # Use tqdm directly in the loop, iterate one extra step for final logging/plotting/saving
    for step in tqdm(range(config.steps + 1), ncols=0):
        # --- LR Scheduling Step --- #
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        # Manually update optimizer's learning rate
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        log_data["lr"] = step_lr

        # --- Zero Gradients --- #
        optimizer.zero_grad()

        # --- Get Batch --- #
        try:
            batch = next(data_iter)["input_ids"].to(device)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            data_iter = iter(train_loader)
            batch = next(data_iter)["input_ids"].to(device)

        (target_out, _), pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: v.A for module_name, v in components.items()}

        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)  # type: ignore

        masks, relud_masks = calc_masks(
            gates=gates,
            target_component_acts=target_component_acts,
            attributions=None,
            detach_inputs=False,
        )
        for layer_name, mask in masks.items():
            alive_components[layer_name] = alive_components[layer_name] | (mask > 0.1).any(
                dim=(0, 1)
            )

        # --- Calculate Losses --- #
        total_loss = torch.tensor(0.0, device=device)
        loss_terms = {}

        ####### param match loss #######
        param_match_loss_val = calc_param_match_loss_lm(
            components=components,
            target_model=model.model,
            n_params=n_params,
            device=device,
        )
        total_loss += config.param_match_coeff * param_match_loss_val
        loss_terms["loss/parameter_matching"] = param_match_loss_val.item()

        ####### layerwise recon loss #######
        if config.layerwise_recon_coeff is not None:
            layerwise_recon_loss = calc_layerwise_recon_loss_lm(
                model=model,
                batch=batch,
                device=device,
                components=components,
                masks=[masks],
                target_out=target_out,
            )
            total_loss += config.layerwise_recon_coeff * layerwise_recon_loss
            loss_terms["loss/layerwise_reconstruction"] = layerwise_recon_loss.item()

        ####### layerwise random recon loss #######
        if config.layerwise_random_recon_coeff is not None:
            layerwise_random_masks = calc_random_masks(
                masks=masks, n_random_masks=config.n_random_masks
            )
            layerwise_random_recon_loss = calc_layerwise_recon_loss_lm(
                model=model,
                batch=batch,
                device=device,
                components=components,
                masks=layerwise_random_masks,
                target_out=target_out,
            )
            total_loss += config.layerwise_random_recon_coeff * layerwise_random_recon_loss
            loss_terms["loss/layerwise_random_reconstruction"] = layerwise_random_recon_loss.item()

        ####### lp sparsity loss #######
        lp_sparsity_loss = calc_lp_sparsity_loss_lm(relud_masks=relud_masks, pnorm=config.pnorm)
        total_loss += config.lp_sparsity_coeff * lp_sparsity_loss
        loss_terms["loss/lp_sparsity_loss"] = lp_sparsity_loss.item()
        ####### Schatten loss #######
        if config.schatten_coeff is not None:
            schatten_loss = calc_schatten_loss_lm(
                relud_masks=relud_masks, pnorm=config.pnorm, components=components, device=device
            )
            total_loss += config.schatten_coeff * schatten_loss
            loss_terms["loss/schatten_loss"] = schatten_loss.item()
        ####### embedding recon loss #######
        if config.embedding_recon_coeff is not None:
            assert len(components) == 1, "Only one embedding component is supported"
            component = list(components.values())[0]
            assert isinstance(component, EmbeddingComponent)
            random_masks = calc_random_masks(masks=masks, n_random_masks=config.n_random_masks)
            embedding_recon_loss = calc_embedding_recon_loss_lm(
                model=model,
                batch=batch,
                component=component,
                masks=random_masks[0],
                unembed=config.is_embed_unembed_recon,
            )
            total_loss += config.embedding_recon_coeff * embedding_recon_loss
            loss_terms["loss/embedding_reconstruction"] = embedding_recon_loss.item()

        log_data["loss/total"] = total_loss.item()
        log_data.update(loss_terms)

        with torch.inference_mode():
            # --- Logging --- #
            if step % config.print_freq == 0:
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                tqdm.write(f"Total Loss: {log_data['loss/total']:.7f}")
                for name, value in loss_terms.items():
                    if value is not None:
                        tqdm.write(f"{name}: {value:.7f}")

                masked_component_logits, _ = model.forward_with_components(
                    batch, components=components, masks=masks
                )
                unmasked_component_logits, _ = model.forward_with_components(
                    batch, components=components, masks=None
                )

                for layer_name, layer_alive_components in alive_components.items():
                    if step == 0:
                        break
                    log_data[f"{layer_name}/n_alive_components_01"] = (
                        layer_alive_components.sum().item()
                    )
                    alive_components[layer_name] = torch.zeros(config.m, device=device).bool()

                target_logits, _ = model.forward(batch)

                unmasked_kl_loss = calc_kl_divergence_lm(
                    pred=unmasked_component_logits, target=target_logits
                )
                masked_kl_loss = calc_kl_divergence_lm(
                    pred=masked_component_logits, target=target_logits
                )

                ###### CE vs true labels #######
                flat_all_component_logits = einops.rearrange(
                    unmasked_component_logits, "batch pos vocab -> (batch pos) vocab"
                )
                flat_masked_component_logits = einops.rearrange(
                    masked_component_logits, "batch pos vocab -> (batch pos) vocab"
                )
                flat_batch = einops.rearrange(batch, "batch pos -> (batch pos)")
                unmasked_ce_loss = F.cross_entropy(
                    input=flat_all_component_logits[:-1], target=flat_batch[1:]
                )
                masked_ce_loss = F.cross_entropy(
                    input=flat_masked_component_logits[:-1], target=flat_batch[1:]
                )

                flat_target_logits = einops.rearrange(
                    target_logits, "batch pos vocab -> (batch pos) vocab"
                )
                target_ce_loss = F.cross_entropy(
                    input=flat_target_logits[:-1], target=flat_batch[1:]
                )

                # --- CE when every component is fully masked (all-zero masks) --- #
                zero_masks = {k: torch.zeros_like(v) for k, v in masks.items()}
                zero_masked_component_logits, _ = model.forward_with_components(
                    batch, components=components, masks=zero_masks
                )
                flat_zero_masked_component_logits = einops.rearrange(
                    zero_masked_component_logits, "batch pos vocab -> (batch pos) vocab"
                )
                zero_masked_ce_loss = F.cross_entropy(
                    input=flat_zero_masked_component_logits[:-1], target=flat_batch[1:]
                )

                embed_mask_table = create_embed_mask_sample_table(masks)
                if embed_mask_table is not None:
                    log_data["misc/embed_mask_sample"] = embed_mask_table

                log_data["misc/unmasked_kl_loss_vs_target"] = unmasked_kl_loss.item()
                log_data["misc/masked_kl_loss_vs_target"] = masked_kl_loss.item()
                log_data["misc/unmasked_ce_loss_vs_labels"] = unmasked_ce_loss.item()
                log_data["misc/masked_ce_loss_vs_labels"] = masked_ce_loss.item()
                log_data["misc/target_ce_loss_vs_labels"] = target_ce_loss.item()
                log_data["misc/zero_masked_ce_loss_vs_labels"] = zero_masked_ce_loss.item()

                if config.wandb_project:
                    mask_l_zero = calc_mask_l_zero(masks=masks)
                    for layer_name, layer_mask_l_zero in mask_l_zero.items():
                        log_data[f"{layer_name}/mask_l0"] = layer_mask_l_zero
                    wandb.log(log_data, step=step)

            # --- Plotting --- #
            if (
                config.image_freq is not None
                and step % config.image_freq == 0
                and (step > 0 or config.image_on_first_step)
            ):
                logger.info(f"Step {step}: Generating plots...")
                mean_component_activation_counts = component_activation_statistics(
                    model=model, dataloader=eval_loader, n_steps=n_eval_steps, device=device
                )[1]
                assert mean_component_activation_counts is not None
                fig_dict = plot_lm_results(
                    mean_component_activation_counts=mean_component_activation_counts,
                )

                if config.wandb_project:
                    wandb.log(
                        {k: wandb.Image(v) for k, v in fig_dict.items()},
                        step=step,
                    )
                    if out_dir is not None:
                        for k, v in fig_dict.items():
                            v.savefig(out_dir / f"{k}_{step}.png")
                            tqdm.write(f"Saved plot to {out_dir / f'{k}_{step}.png'}")

        # --- Saving Checkpoint --- #
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"model_{step}.pth"), base_path=str(out_dir), policy="now")
                wandb.save(
                    str(out_dir / f"optimizer_{step}.pth"), base_path=str(out_dir), policy="now"
                )

        # --- Backward Pass & Optimize --- #
        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            total_loss.backward(retain_graph=True)

            if step % config.print_freq == 0 and config.wandb_project:
                # Calculate gradient norm
                grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.flatten().pow(2).sum()  # type: ignore
                grad_norm_val = grad_norm.sqrt().item()
                wandb.log({"grad_norm": grad_norm_val}, step=step)

            if config.unit_norm_matrices:
                model.fix_normalized_adam_gradients()

            optimizer.step()
    logger.info("Finished training loop.")


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    set_seed(config.seed)
    logger.info(config)

    device = get_device()
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, LMTaskConfig), (
        "Task config must be LMTaskConfig for LM decomposition."
    )

    # --- Load Model --- #
    logger.info(f"Loading model: {config.task_config.model_size}")
    model_config_dict = MODEL_CONFIGS[config.task_config.model_size]
    model_path = f"chandan-sreedhara/SimpleStories-{config.task_config.model_size}"
    model = Llama.from_pretrained(model_path, model_config_dict)

    ss_model = SSModel(
        llama_model=model,
        target_module_patterns=config.task_config.target_module_patterns,
        m=config.m,
        n_gate_hidden_neurons=config.n_gate_hidden_neurons,
    )
    ss_model.to(device)
    logger.info("Model loaded.")

    # --- Setup Run Name and Output Dir --- #
    run_name = get_run_name(
        config,
        model_size=config.task_config.model_size,
        max_seq_len=config.task_config.max_seq_len,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # --- Save Config --- #
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    # --- Load Data --- #
    logger.info("Loading dataset...")
    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        tokenizer_file_path=None,
        hf_tokenizer_path=model_path,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name="story",
    )

    train_loader, tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        tokenizer_file_path=None,
        hf_tokenizer_path=model_path,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name="story",
    )
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    logger.info("Dataset and tokenizer loaded.")

    logger.info("Freezing target model parameters...")
    for param in ss_model.model.parameters():
        param.requires_grad = False
    logger.info("Target model frozen.")

    logger.info("Starting optimization...")
    optimize_lm(
        model=ss_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.task_config.n_eval_steps,
        out_dir=out_dir,
    )

    logger.info("Optimization finished.")

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
