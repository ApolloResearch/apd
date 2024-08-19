import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


def plot_A_matrix(x: torch.Tensor, pos_only: bool = False) -> plt.Figure:
    n_instances = x.shape[0]

    fig, axs = plt.subplots(
        1, n_instances, figsize=(2.5 * n_instances, 2), squeeze=False, sharey=True
    )

    cmap = "Blues" if pos_only else "RdBu"
    ims = []
    for i in range(n_instances):
        ax = axs[0, i]
        instance_data = x[i, :, :].detach().cpu().float().numpy()
        max_abs_val = np.abs(instance_data).max()
        vmin = 0 if pos_only else -max_abs_val
        vmax = max_abs_val
        im = ax.matshow(instance_data, vmin=vmin, vmax=vmax, cmap=cmap)
        ims.append(im)
        ax.xaxis.set_ticks_position("bottom")
        if i == 0:
            ax.set_ylabel("k", rotation=0, labelpad=10, va="center")
        else:
            ax.set_yticks([])  # Remove y-axis ticks for all but the first plot
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("n_features")

    plt.subplots_adjust(wspace=0.1, bottom=0.15, top=0.9)
    fig.subplots_adjust(bottom=0.2)

    return fig


class TMSDataset(
    Dataset[tuple[Float[Tensor, "n_instances n_features"], Float[Tensor, "n_instances n_features"]]]
):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: float,
        device: str,
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "n_instances n_features"], Float[Tensor, "n_instances n_features"]]:
        """Generate a batch of samples from the TMS distribution.

        We only keep samples that have at least one non-zero feature.
        """
        # Combine batch_size and n_instances into a single dimension and then reshape at the end.
        # This avoids multidim indexing.
        n_elements = batch_size * self.n_instances
        batch_elements: list[Float[Tensor, " n_elements n_features"]] = []
        while len(batch_elements) < n_elements:
            # Generate more than n_elements to avoid many calls to rand.
            samples = torch.rand(n_elements * 5, self.n_features, device=self.device)
            mask = torch.rand_like(samples) < self.feature_probability
            # Only keep the samples that have at least one non-zero feature
            mask = mask.any(dim=1)
            batch_elements.extend(samples[mask])
        batch_elements = batch_elements[:n_elements]
        batch = torch.stack(batch_elements).reshape(batch_size, self.n_instances, self.n_features)
        return batch, batch.clone().detach()

        samples = torch.rand(batch_size, self.n_instances, self.n_features, device=self.device)
        mask = torch.rand_like(samples) < self.feature_probability
        batch = samples * mask
        # Only keep the samples that have at least one non-zero feature
        return batch, batch.clone().detach()
