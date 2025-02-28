from torch import Tensor, cat, full, tensor
import json
from dataclasses import dataclass
from statistics import mean
from typing import Literal
from jaxtyping import Float
from beartype import beartype

from transformer import Transformer
from train import transformer_cross_entropy
from word_tokenizer import WordTokenizer


@beartype
@dataclass(frozen=True)
class WinoGrandeQuestion:
    sentence: str
    option1: str
    option2: str
    answer: Literal[1, 2]


@beartype
def load_winogrande_questions(filename: str) -> list[WinoGrandeQuestion]:
    with open(filename) as f:
        dataset = json.load(f)

    for datapoint in dataset:
        assert (
            set(datapoint.keys()) == {"sentence", "option1", "option2", "answer"}  # type: ignore
            and datapoint["answer"] in ["1", "2"]  # type: ignore
        )
    assert all(
        set(datapoint.keys()) == {"sentence", "option1", "option2", "answer"}  # type: ignore
        and datapoint["answer"] in ["1", "2"]  # type: ignore
        for datapoint in dataset
    ), (
        "All entries of the huggingface dataset must be in the same format as allenai/winogrande."
    )

    return [
        WinoGrandeQuestion(
            sentence=datapoint["sentence"],  # type: ignore
            option1=datapoint["option1"],  # type: ignore
            option2=datapoint["option2"],  # type: ignore
            answer={"1": 1, "2": 2}[datapoint["answer"]],  # type: ignore
        )
        for datapoint in dataset
    ]


@beartype
def perplexities(
    model: Transformer, tokenizer: WordTokenizer, sequences: list[str]
) -> list[float]:
    device = next(iter(model.parameters())).device

    tokenized_sequences_list: list[list[int]] = [
        tokenizer.encode(sequence) for sequence in sequences
    ]

    assert all(
        0 < len(sequence) <= model.cfg.context_length
        for sequence in tokenized_sequences_list
    )

    for sequence in tokenized_sequences_list:
        while len(sequence) < model.cfg.context_length:
            sequence.append(tokenizer.eos_token_id)

    tokenized_sequences = tensor(tokenized_sequences_list, device=device)

    input = cat(
        (
            full(
                size=(len(sequences), 1),
                fill_value=tokenizer.eos_token_id,
                device=device,
            ),
            tokenized_sequences[:, :-1],
        ),
        dim=1,  # concatenate along sequence_length dimension
    )

    logits = model(input)

    cross_entropies: Float[Tensor, "batch_size sequence_length"] = (
        transformer_cross_entropy(logits, tokenized_sequences, reduction="none")
    )

    return [
        mean(cross_entropies_on_sequence[: len(tokenized_sequence)])
        for cross_entropies_on_sequence, tokenized_sequence in zip(
            cross_entropies.tolist(), tokenized_sequences, strict=True
        )
    ]


@beartype
def winogrande_accuracy(
    model: Transformer, tokenizer: WordTokenizer, questions: list[WinoGrandeQuestion]
) -> float:
    sentences_with_option1: list[str] = [
        question.sentence.replace("_", question.option1) for question in questions
    ]
    sentences_with_option2: list[str] = [
        question.sentence.replace("_", question.option2) for question in questions
    ]

    perplexities_with_option1 = perplexities(model, tokenizer, sentences_with_option1)
    perplexities_with_option2 = perplexities(model, tokenizer, sentences_with_option2)

    n_undistinguishable = len(
        [
            None
            for p1, p2 in zip(
                perplexities_with_option1, perplexities_with_option2, strict=True
            )
            if p1 == p2
        ]
    )
    print(
        f"Equal perplexities on both sequences for {n_undistinguishable / len(questions):.3%} of WinoGrande questions."
    )

    model_answers = [
        1 if p1 > p2 else 2
        for p1, p2 in zip(
            perplexities_with_option1, perplexities_with_option2, strict=True
        )
    ]

    return mean(
        int(model_answer == question.answer)
        for model_answer, question in zip(model_answers, questions, strict=True)
    )


# ruff: noqa: F722
