from torch import Tensor, linspace, inference_mode, sigmoid, where
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Optimizer
from plotly.graph_objects import Figure
from tqdm import tqdm
from math import cos, pi
from dataclasses import dataclass, field
from collections.abc import Callable
from beartype import beartype
from jaxtyping import jaxtyped, Float, Int

from transformer import Transformer, TransformerConfig


@jaxtyped(typechecker=beartype)
def token_accuracy(
    logits: Float[Tensor, "batch sequence_length vocabulary_size"],
    labels: Int[Tensor, "batch sequence_length"],
) -> Float[Tensor, ""]:
    predictions = logits.argmax(-1)  # argmax along vocabulary_size dimension
    return (predictions == labels).float().mean()


@jaxtyped(typechecker=beartype)
def transformer_cross_entropy(
    logits: Float[Tensor, "batch sequence_length vocabulary_size"],
    labels: Int[Tensor, "batch sequence_length"],
    reduction: str = "mean"
) -> Float[Tensor, ""] | Float[Tensor, "batch_size sequence_length"]:
    return cross_entropy(logits.transpose(-1, -2), labels, reduction=reduction)


@beartype
def binary_classification_accuracy_with_logits(
    logits: Float[Tensor, "..."], labels: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    assert (labels == 0).logical_or(labels == 1).all()
    return ((logits >= 0.0) == (labels == 1)).float().mean()


@jaxtyped(typechecker=beartype)
def binary_cross_entropy_with_logits(
    logits: Float[Tensor, "..."], labels: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    assert (labels == 0).logical_or(labels == 1).all()

    s = sigmoid(logits)
    return where(labels == 1, 1 - s, s).mean()


@beartype
@dataclass
class TrainingStatistics:
    epochs: int
    train_losses: list[float] = field(default_factory=lambda: [])
    train_accuracies: list[float] = field(default_factory=lambda: [])
    test_losses: list[float] = field(default_factory=lambda: [])
    test_accuracies: list[float] = field(default_factory=lambda: [])

    def plot(self, title: str = "training") -> None:
        fig = Figure()
        fig.update_layout(
            title=title,
            xaxis=dict(title="epoch"),
            yaxis=dict(title="loss", type="log"),
            yaxis2=dict(
                title="accuracy",
                range=[0, 1],
                overlaying="y",
                anchor="free",
                autoshift=True,
            ),
        )

        if self.train_losses != []:
            fig.add_scatter(
                x=linspace(0, self.epochs, len(self.train_losses)).tolist(),
                y=self.train_losses,
                name="train loss",
            )

        if self.test_losses != []:
            assert len(self.test_losses) == self.epochs
            fig.add_scatter(
                x=list(range(1, self.epochs + 1)), y=self.test_losses, name="test loss"
            )

        if self.train_accuracies != []:
            fig.add_scatter(
                x=linspace(0, self.epochs, len(self.train_accuracies)).tolist(),
                y=self.train_accuracies,
                name="train accuracy",
                yaxis="y2",
            )

        if self.test_accuracies != []:
            assert len(self.test_accuracies) == self.epochs
            fig.add_scatter(
                x=list(range(1, self.epochs + 1)),
                y=self.test_accuracies,
                name="test accuracy",
                yaxis="y2",
            )

        fig.show()


@beartype
def cosine_schedule(t: float, start: float) -> float:
    assert 0 <= t <= 1
    return start * cos(t * pi / 2)


@beartype
@dataclass(frozen=True)
class TrainingConfig:
    model_config: TransformerConfig
    epochs: int
    learning_rate: float
    cosine_learning_rate_schedule: bool
    batch_size: int
    device: str

    def learning_rate_schedule(self, t: float) -> float:
        if self.cosine_learning_rate_schedule:
            return cosine_schedule(t=t, start=self.learning_rate)
        else:
            return self.learning_rate


@beartype
def train(
    cfg: TrainingConfig,
    train_dataset: Dataset,
    test_dataset: Dataset | None = None,
    verbose: bool = True,
    loss_function: Callable[[Tensor, Tensor], Tensor] = transformer_cross_entropy,
    accuracy_function: Callable[[Tensor, Tensor], Tensor] = token_accuracy,
) -> tuple[Transformer, TrainingStatistics]:
    model = Transformer(cfg.model_config).to(cfg.device)

    stats = TrainingStatistics(epochs=cfg.epochs)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate_schedule(0.0))
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=True
        )

    for epoch in tqdm(
        range(cfg.epochs), desc="training transformer", disable=not verbose
    ):  # type: ignore
        model.train()  # should be called at each iteration because thee test function calls model.eval()

        for i_datapoint, (input, output) in enumerate(
            tqdm(
                train_dataloader,
                desc=f"epoch {epoch + 1}/{cfg.epochs}",
                leave=False,
                disable=not verbose,
            )
        ):
            set_learning_rate(
                optimizer,
                cfg.learning_rate_schedule(
                    (
                        epoch / cfg.epochs
                        + i_datapoint / len(train_dataloader) / cfg.epochs
                    )
                ),
            )

            input = input.to(cfg.device)
            output = output.to(cfg.device)

            optimizer.zero_grad()
            logits = model(input)
            loss = loss_function(logits, output)
            accuracy = accuracy_function(logits, output)
            loss.backward()
            optimizer.step()

            stats.train_losses.append(loss.item())
            stats.train_accuracies.append(accuracy.item())

        if test_dataset is not None:
            test_result = test(
                model,
                test_dataloader, # type: ignore
                verbose=verbose,
                loss_function=loss_function,
                accuracy_function=accuracy_function,
            )
            stats.test_losses.append(test_result.loss)
            stats.test_accuracies.append(test_result.accuracy)

    return model, stats


@beartype
@dataclass
class LossAndAccuracy:
    loss: float
    accuracy: float


@beartype
@inference_mode()
def test(
    model: Module,
    dataloader: DataLoader,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    accuracy_function: Callable[[Tensor, Tensor], Tensor],
    verbose: bool = True,
) -> LossAndAccuracy:
    model.eval()

    device = next(iter(model.parameters())).device

    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    for input, output in tqdm(
        dataloader, desc="testing", leave=False, disable=not verbose
    ):
        input = input.to(device)
        output = output.to(device)

        logits = model(input)
        loss = loss_function(logits, output).item()
        accuracy = accuracy_function(logits, output).item()
        n_samples = input.size(0)

        total_loss += n_samples * loss
        total_accuracy += n_samples * accuracy
        total_samples += n_samples

    return LossAndAccuracy(
        loss=total_loss / total_samples, accuracy=total_accuracy / total_samples
    )



@beartype
def set_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = learning_rate


# ruff: noqa: F722
