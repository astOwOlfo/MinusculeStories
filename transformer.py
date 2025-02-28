from torch.nn import Module, Linear, Parameter, RMSNorm, Sequential, Embedding
from torch.nn.functional import relu, silu
import torch
from torch import (
    Tensor,
    arange,
    tril,
    zeros,
    ones,
    tensor,
    where,
    cat,
    stack,
    inference_mode,
    Generator,
)
from torch.distributions import Categorical
from torch.nn.init import trunc_normal_, zeros_, ones_
from einops import einsum, rearrange
from math import inf
from dataclasses import dataclass
from typing import Literal
from beartype import beartype
from jaxtyping import jaxtyped, Float, Int, Bool


@beartype
@dataclass(frozen=True)
class TransformerConfig:
    context_length: int
    vocabulary_size: int
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_mlp: int
    activation_function: Literal["relu", "silu"]
    rotary_positional_embedding_base: float | int
    task: Literal["next_token", "binary_classification"]

    weight_initialization_std: float = 0.02
    weight_initialization_cutoff: float = 0.09
    weight_initialization_seed: int = 42

@jaxtyped(typechecker=beartype)
class MLP(Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.up = Linear(cfg.d_model, cfg.d_mlp)
        self.gate = Linear(cfg.d_model, cfg.d_mlp)
        self.down = Linear(cfg.d_mlp, cfg.d_model)

    def forward(
        self, x: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        up_output = self.up(x)
        gate_output = self.gate(x)
        gate_output = gate_output
        activation_function_output = self.activation_function(gate_output)
        down_input = up_output * activation_function_output
        down_output = self.down(down_input)
        return down_output

    def activation_function(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        f = {"relu": relu, "silu": silu}[self.cfg.activation_function]
        return f(x)


@jaxtyped(typechecker=beartype)
def rotary_positonal_embedding(
    x: Float[Tensor, "batch position n_heads d_head"],
    sines: Float[Tensor, "context_length half_d_head"],
    cosines: Float[Tensor, "context_length half_d_head"],
) -> Float[Tensor, "batch position n_heads d_head"]:
    d_head = x.size(-1)
    assert d_head % 2 == 0, "d_head must be even when using rotary positiona embedding."

    sequence_length = x.size(1)
    context_length = sines.size(0)
    assert cosines.size(0) == sines.size(0)
    assert sequence_length <= context_length
    sines = sines[:sequence_length, :]
    cosines = cosines[:sequence_length, :]

    x_even = x[:, :, :, ::2]  # elements with even d_head index
    x_odd = x[:, :, :, 1::2]  # elements with odd d_head index

    p = "batch position n_heads half_d_head, position half_d_head -> batch position n_heads half_d_head"
    output_even = einsum(x_even, cosines, p) - einsum(x_odd, sines, p)
    output_odd = einsum(x_even, sines, p) + einsum(x_odd, cosines, p)

    output = rearrange(
        stack((output_even, output_odd), dim=-1),
        "batch position n_heads half_d_head parity -> batch position n_heads (half_d_head parity)",
    )

    return output


@jaxtyped(typechecker=beartype)
class Attention(Module):
    sines: Float[Tensor, "position half_d_head"]
    cosines: Float[Tensor, "position half_d_head"]

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.key_weight = Parameter(zeros(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.query_weight = Parameter(zeros(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.value_weight = Parameter(zeros(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.output_weight = Parameter(zeros(cfg.d_model, cfg.n_heads, cfg.d_head))

        angles = cfg.rotary_positional_embedding_base ** (
            -2 * arange(cfg.d_head // 2) / cfg.d_head
        )
        positions = arange(cfg.context_length)
        angles_times_positions = einsum(
            angles, positions, "half_d_head, position -> position half_d_head"
        )
        self.register_buffer("sines", angles_times_positions.sin())
        self.register_buffer("cosines", angles_times_positions.cos())

    def forward(
        self,
        x: Float[Tensor, "batch position d_model"],
    ) -> Float[Tensor, "batch position d_model"]:
        p = "n_heads d_head d_model, batch position d_model -> batch position n_heads d_head"
        keys = einsum(self.key_weight, x, p)
        queries = einsum(self.query_weight, x, p)
        values = einsum(self.value_weight, x, p)

        keys = rotary_positonal_embedding(keys, sines=self.sines, cosines=self.cosines)
        queries = rotary_positonal_embedding(
            queries, sines=self.sines, cosines=self.cosines
        )

        scores = einsum(
            keys,
            queries,
            "batch key_position n_heads d_head, batch query_position n_heads d_head -> batch n_heads query_position key_position",
        )
        mask = self.mask(sequence_length=x.size(-2), device=scores.device)
        scores = where(mask, scores, tensor(-inf))
        pattern = scores.softmax(-1)

        outputs = einsum(
            pattern,
            values,
            "batch n_heads query_position key_position, batch key_position n_heads d_head -> batch query_position n_heads d_head",
        )
        results = einsum(
            self.output_weight,
            outputs,
            "d_model n_heads d_head, batch position n_heads d_head -> batch position d_model",
        )
        return results

    def mask(
        self, sequence_length: int, device: str | torch.device
    ) -> Bool[Tensor, "query_position key_position"]:
        match self.cfg.task:
            case "binary_classification":
                return ones(
                    size=(sequence_length, sequence_length),
                    dtype=torch.bool,
                    device=device,
                )
            case "next_token":
                return tril(
                    ones(
                        sequence_length,
                        sequence_length,
                        dtype=torch.bool,
                        device=device,
                    )
                )
            case _:
                assert False, f"Unknown task {self.cfg.task}."


@jaxtyped(typechecker=beartype)
class TransformerBlock(Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.mlp = MLP(cfg)
        self.attention = Attention(cfg)
        self.mlp_layer_norm = RMSNorm(cfg.d_model)
        self.attention_layer_norm = RMSNorm(cfg.d_model)

    def forward(
        self, x: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        attention_input = self.attention_layer_norm(x)
        attention_output = self.attention(attention_input)
        x = x + attention_output

        mlp_input = self.mlp_layer_norm(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x


@jaxtyped(typechecker=beartype)
class Transformer(Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.blocks = Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.embedding = Embedding(cfg.vocabulary_size, cfg.d_model)
        self.final_layer_norm = RMSNorm(cfg.d_model)

        match cfg.task:
            case "next_token":
                self.unembedding = Linear(cfg.d_model, cfg.vocabulary_size, bias=False)
            case "binary_classification":
                self.binary_classification_head = Linear(cfg.d_model, 1, bias=True)
            case _:
                assert False, f"unknown task {self.cfg.task}"

        self.initialize_weights()

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position vocabulary_size"]:
        sequence_length = tokens.size(-1)
        assert sequence_length <= self.cfg.context_length

        x = self.embedding(tokens)
        x = self.blocks(x)
        x = self.final_layer_norm(x)

        match self.cfg.task:
            case "next_token":
                logits = self.unembedding(x)
            case "binary_classification":
                x = x.mean(-2)  # mean along context_length dimension
                logits = self.binary_classification_head(x)
            case _:
                assert False

        return logits

    def generate(
        self,
        tokens: Int[Tensor, "batch position"],
        n_new_tokens: int,
        temperature: float,
    ) -> Int[Tensor, "batch position"]:
        for _ in range(n_new_tokens):
            logits = self(tokens)
            logits = logits[:, -1, :]
            if temperature == 0:
                new_tokens = logits.argmax(-1)
            else:
                new_tokens = Categorical(logits=logits / temperature).sample()
            tokens = cat((tokens, new_tokens.unsqueeze(-1)), -1)
        return tokens

    @inference_mode()
    def initialize_weights(self) -> None:
        rng = Generator().manual_seed(self.cfg.weight_initialization_seed)

        for parameter in self.parameters():
            trunc_normal_(
                parameter.data,
                mean=0.0,
                std=self.cfg.weight_initialization_std,
                a=-self.cfg.weight_initialization_cutoff,
                b=self.cfg.weight_initialization_cutoff,
                generator=rng,
            )

        for submodule in self.modules():
            if isinstance(submodule, Linear) and submodule.bias is not None:
                zeros_(submodule.bias)
            if isinstance(submodule, RMSNorm) and submodule.weight is not None:
                ones_(submodule.weight)


@beartype
def print_parameter_counts(model: Transformer) -> None:
    n_parameters = sum(param.numel() for param in model.parameters())
    n_embedding_parameters = model.embedding.weight.numel()
    match model.cfg.task:
        case "next_token":
            n_embedding_parameters += model.unembedding.weight.numel()
            if model.unembedding.bias is not None:
                n_embedding_parameters += model.unembedding.bias.numel()
        case "binary_classification":
            n_embedding_parameters += model.binary_classification_head.weight.numel()
            if model.binary_classification_head.bias is not None:
                n_embedding_parameters += model.binary_classification_head.bias.numel()
        case _:
            assert False, f"Invalid task {model.cfg.task}."
    n_non_embedding_parameters = n_parameters - n_embedding_parameters
    print(f"number of parameters: {n_parameters}")
    print(f"number of non embedding parameters: {n_non_embedding_parameters}")


# ruff: noqa: F722
