"""Run SPD on a model."""

from collections.abc import Callable
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.log import logger
from spd.losses import (
    calc_embedding_recon_loss,
    calc_layerwise_recon_loss,
    calc_lp_sparsity_loss,
    calc_masked_recon_loss,
    calc_param_match_loss,
    calc_schatten_loss,
)
from spd.models.component_model import ComponentModel, init_As_and_Bs_
from spd.models.component_utils import (
    calc_component_acts,
    calc_mask_l_zero,
    calc_masks,
    calc_random_masks,
    component_activation_statistics,
)
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.plotting import (
    create_embed_mask_sample_table,
    plot_mask_histograms,
    plot_mean_component_activation_counts,
)
from spd.utils import (
    calc_kl_divergence_lm,
    extract_batch_data,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)


def get_common_run_name_suffix(config: Config) -> str:
    """Generate a run suffix based on Config that is common to all experiments."""
    run_suffix = ""
    if config.masked_recon_coeff is not None:
        run_suffix += f"maskrecon{config.masked_recon_coeff:.2e}_"
        run_suffix += f"nrandmasks{config.n_stochastic_masks}_"
    if config.random_mask_recon_coeff is not None:
        run_suffix += f"randrecon{config.random_mask_recon_coeff:.2e}_"
    run_suffix += f"p{config.pnorm:.2e}_"
    run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
    run_suffix += f"C{config.C}_"
    run_suffix += f"sd{config.seed}_"
    run_suffix += f"lr{config.lr:.2e}_"
    run_suffix += f"bs{config.batch_size}_"
    return run_suffix


def optimize(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    eval_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    out_dir: Path | None,
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    model = ComponentModel(
        base_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        n_gate_hidden_neurons=config.n_gate_hidden_neurons,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )

    for param in target_model.parameters():
        param.requires_grad = False
    logger.info("Target model parameters frozen.")

    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    model.to(device)
    init_As_and_Bs_(model=model, components=components)

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        for src_name, tgt_name in tied_weights:
            components[tgt_name].B.data = components[src_name].A.data.T
            components[tgt_name].A.data = components[src_name].B.data.T

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

    alive_components: dict[str, Bool[Tensor, " C"]] = {
        layer_name: torch.zeros(config.C, device=device).bool() for layer_name in components
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

        try:
            batch_item = next(data_iter)
            batch = extract_batch_data(batch_item)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            data_iter = iter(train_loader)
            batch_item = next(data_iter)
            batch = extract_batch_data(batch_item)
        batch = batch.to(device)

        target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: components[module_name].A for module_name in components}

        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)  # type: ignore

        masks, sparsity_masks = calc_masks(
            gates=gates, target_component_acts=target_component_acts, detach_inputs=False
        )
        for layer_name, mask in masks.items():
            alive_components[layer_name] = alive_components[layer_name] | (mask > 0.1).any(
                dim=(0, 1)
            )

        # --- Calculate Losses --- #
        total_loss = torch.tensor(0.0, device=device)
        loss_terms = {}

        ####### param match loss #######
        param_match_loss_val = calc_param_match_loss(
            components=components,
            target_model=model.model,
            n_params=n_params,
            device=device,
        )
        total_loss += config.param_match_coeff * param_match_loss_val
        loss_terms["loss/parameter_matching"] = param_match_loss_val.item()

        ####### masked recon loss #######
        if config.masked_recon_coeff is not None:
            masked_recon_loss = calc_masked_recon_loss(
                model=model,
                batch=batch,
                components=components,
                masks=masks,
                target_out=target_out,
                loss_type=config.output_loss_type,
            )
            total_loss += config.masked_recon_coeff * masked_recon_loss
            loss_terms["loss/masked_reconstruction"] = masked_recon_loss.item()

        ####### random mask recon loss #######
        if config.random_mask_recon_coeff is not None:
            random_masks = calc_random_masks(
                masks=masks, n_stochastic_masks=config.n_stochastic_masks
            )
            random_mask_loss = torch.tensor(0.0, device=target_out.device)
            for i in range(len(random_masks)):
                random_mask_loss += calc_masked_recon_loss(
                    model=model,
                    batch=batch,
                    components=components,
                    masks=random_masks[i],
                    target_out=target_out,
                    loss_type=config.output_loss_type,
                )
            random_mask_loss = random_mask_loss / len(random_masks)
            total_loss += config.random_mask_recon_coeff * random_mask_loss
            loss_terms["loss/random_mask_reconstruction"] = random_mask_loss.item()

        ####### layerwise recon loss #######
        if config.layerwise_recon_coeff is not None:
            layerwise_recon_loss = calc_layerwise_recon_loss(
                model=model,
                batch=batch,
                device=device,
                components=components,
                masks=[masks],
                target_out=target_out,
                loss_type=config.output_loss_type,
            )
            total_loss += config.layerwise_recon_coeff * layerwise_recon_loss
            loss_terms["loss/layerwise_reconstruction"] = layerwise_recon_loss.item()

        ####### layerwise random recon loss #######
        if config.layerwise_random_recon_coeff is not None:
            layerwise_random_masks = calc_random_masks(
                masks=masks, n_stochastic_masks=config.n_stochastic_masks
            )
            layerwise_random_recon_loss = calc_layerwise_recon_loss(
                model=model,
                batch=batch,
                device=device,
                components=components,
                masks=layerwise_random_masks,
                target_out=target_out,
                loss_type=config.output_loss_type,
            )
            total_loss += config.layerwise_random_recon_coeff * layerwise_random_recon_loss
            loss_terms["loss/layerwise_random_reconstruction"] = layerwise_random_recon_loss.item()

        ####### lp sparsity loss #######
        lp_sparsity_loss = calc_lp_sparsity_loss(sparsity_masks=sparsity_masks, pnorm=config.pnorm)
        total_loss += config.lp_sparsity_coeff * lp_sparsity_loss
        loss_terms["loss/lp_sparsity_loss"] = lp_sparsity_loss.item()

        ####### Schatten loss #######
        if config.schatten_coeff is not None:
            schatten_loss = calc_schatten_loss(
                sparsity_masks=sparsity_masks,
                pnorm=config.pnorm,
                components=components,
                device=device,
            )
            total_loss += config.schatten_coeff * schatten_loss
            loss_terms["loss/schatten_loss"] = schatten_loss.item()

        ####### output recon loss #######
        if config.out_recon_coeff is not None:
            masks_all_ones = {k: torch.ones_like(v) for k, v in masks.items()}
            out_recon_loss = calc_masked_recon_loss(
                model=model,
                batch=batch,
                components=components,
                masks=masks_all_ones,
                target_out=target_out,
                loss_type=config.output_loss_type,
            )
            total_loss += config.out_recon_coeff * out_recon_loss
            loss_terms["loss/output_reconstruction"] = out_recon_loss.item()

        ####### embedding recon loss #######
        if config.embedding_recon_coeff is not None:
            assert len(components) == 1, "Only one embedding component is supported"
            component = list(components.values())[0]
            assert isinstance(component, EmbeddingComponent)
            random_masks = calc_random_masks(
                masks=masks, n_stochastic_masks=config.n_stochastic_masks
            )
            embedding_recon_loss = calc_embedding_recon_loss(
                model=model,
                batch=batch,
                component=component,
                masks=random_masks,
                embed_module_name=next(iter(components.keys())),
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

                masked_component_logits = model.forward_with_components(
                    batch, components=components, masks=masks
                )
                unmasked_component_logits = model.forward_with_components(
                    batch, components=components, masks=None
                )

                for layer_name, layer_alive_components in alive_components.items():
                    if step == 0:
                        break
                    log_data[f"{layer_name}/n_alive_components_01"] = (
                        layer_alive_components.sum().item()
                    )
                    alive_components[layer_name] = torch.zeros(config.C, device=device).bool()

                target_logits = model(batch)

                unmasked_kl_loss = calc_kl_divergence_lm(
                    pred=unmasked_component_logits, target=target_logits
                )
                masked_kl_loss = calc_kl_divergence_lm(
                    pred=masked_component_logits, target=target_logits
                )

                if config.log_ce_losses:
                    ###### CE vs true labels #######
                    flat_all_component_logits = einops.rearrange(
                        unmasked_component_logits, "... vocab -> (...) vocab"
                    )
                    flat_masked_component_logits = einops.rearrange(
                        masked_component_logits, "... vocab -> (...) vocab"
                    )
                    flat_batch = batch.flatten()
                    unmasked_ce_loss = F.cross_entropy(
                        input=flat_all_component_logits[:-1], target=flat_batch[1:]
                    )
                    masked_ce_loss = F.cross_entropy(
                        input=flat_masked_component_logits[:-1], target=flat_batch[1:]
                    )

                    flat_target_logits = einops.rearrange(target_logits, "... vocab -> (...) vocab")
                    target_ce_loss = F.cross_entropy(
                        input=flat_target_logits[:-1], target=flat_batch[1:]
                    )

                    # --- CE when every component is fully masked (all-zero masks) --- #
                    zero_masks = {k: torch.zeros_like(v) for k, v in masks.items()}
                    zero_masked_component_logits = model.forward_with_components(
                        batch, components=components, masks=zero_masks
                    )
                    flat_zero_masked_component_logits = einops.rearrange(
                        zero_masked_component_logits, "... vocab -> (...) vocab"
                    )
                    zero_masked_ce_loss = F.cross_entropy(
                        input=flat_zero_masked_component_logits[:-1], target=flat_batch[1:]
                    )
                    log_data["misc/unmasked_ce_loss_vs_labels"] = unmasked_ce_loss.item()
                    log_data["misc/masked_ce_loss_vs_labels"] = masked_ce_loss.item()
                    log_data["misc/target_ce_loss_vs_labels"] = target_ce_loss.item()
                    log_data["misc/zero_masked_ce_loss_vs_labels"] = zero_masked_ce_loss.item()

                embed_mask_table = create_embed_mask_sample_table(masks)
                if embed_mask_table is not None:
                    log_data["misc/embed_mask_sample"] = embed_mask_table

                log_data["misc/unmasked_kl_loss_vs_target"] = unmasked_kl_loss.item()
                log_data["misc/masked_kl_loss_vs_target"] = masked_kl_loss.item()

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
                fig_dict = {}
                if plot_results_fn is not None:
                    fig_dict = plot_results_fn(
                        model=model,
                        components=components,
                        gates=gates,
                        batch_shape=batch.shape,
                        device=device,
                    )

                # plot_mask_histograms returns a dict of figures, so we need to merge it
                mask_histogram_figs = plot_mask_histograms(masks=masks)
                fig_dict.update(mask_histogram_figs)

                mean_component_activation_counts = component_activation_statistics(
                    model=model, dataloader=eval_loader, n_steps=n_eval_steps, device=device
                )[1]
                assert mean_component_activation_counts is not None
                fig_dict["mean_component_activation_counts"] = (
                    plot_mean_component_activation_counts(
                        mean_component_activation_counts=mean_component_activation_counts,
                    )
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
