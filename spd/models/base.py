from abc import ABC, abstractmethod
from pathlib import Path

from jaxtyping import Float
from torch import Tensor, nn


class SPDModel(ABC, nn.Module):
    @abstractmethod
    def forward_topk(
        self,
        x: Float[Tensor, "... dim"],
        topk: int,
        all_grads: list[Float[Tensor, "... k"]] | None = None,
    ) -> tuple[
        Float[Tensor, "... dim"],
        list[Float[Tensor, "... dim"]],
        list[Float[Tensor, "... k"]],
    ]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str | Path) -> "SPDModel":
        pass

    @property
    @abstractmethod
    def all_As(self) -> list[Float[Tensor, "dim k"]]:
        pass

    @property
    @abstractmethod
    def all_Bs(self) -> list[Float[Tensor, "k dim"]]:
        pass


class Model(ABC, nn.Module):
    @property
    @abstractmethod
    def all_decomposable_params(self) -> list[Float[Tensor, "..."]]:
        pass
