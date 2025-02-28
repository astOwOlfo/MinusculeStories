import random
from torch import full, tensor, Tensor, Generator, randperm, linspace
from datasets import load_dataset
from plotly.graph_objects import Figure
from tqdm import tqdm
from math import inf
from itertools import chain
from collections import Counter
from collections.abc import Iterable
from jaxtyping import jaxtyped, Int
import gc
from beartype import beartype


@beartype
def split_words(text: str) -> list[str]:
    if random.randint(0, 65536) == 0:
        gc.collect()

    text = text.lower()

    words: list[str] = [""]
    for char in text:
        if char.isspace():
            if words[-1] != "":
                words.append("")
            continue

        if words[-1] == "":
            words[-1] = char
            continue

        if char.isalpha() and words[-1][-1].isalpha():
            words[-1] += char
            continue

        words.append(char)

    if words[-1] == "":
        words.pop()

    return words


@beartype
class WordTokenizer:
    tokens: list[str]
    token_to_id: dict[str, int]

    def __init__(self, tokens: list[str]) -> None:
        assert self.eos_token not in tokens
        self.tokens = tokens + [self.eos_token]
        self.token_to_id = {token: id for id, token in enumerate(self.tokens)}

    @staticmethod
    def with_most_frequent_words(
        dataset: Iterable[str], vocabulary_size: int
    ) -> "WordTokenizer":
        frequencies = Counter(
            tqdm(
                chain.from_iterable(split_words(text) for text in dataset),
                desc="computing word frequencies",
            )
        )
        words_and_frequencies: list[tuple[str, int]] = list(frequencies.items())
        words_and_frequencies.sort(
            reverse=True, key=lambda word_and_freq: word_and_freq[1]
        )
        most_frequent_words = [
            word
            for word, freq in words_and_frequencies[
                : vocabulary_size - 1
            ]  # - 1 because  one of the tokens will be eos_token
        ]
        return WordTokenizer(tokens=most_frequent_words)

    def encode(self, text: str) -> list[int]:
        tokens = split_words(text)
        return [self.token_to_id[token] for token in tokens]

    def can_tokenize(self, text: str) -> bool:
        return all(token in self.token_to_id.keys() for token in split_words(text))

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return self.tokens[token_ids]
        return " ".join(self.tokens[id] for id in token_ids)

    @property
    def eos_token(self) -> str:
        return "<|eos|>"

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def vocabulary_size(self) -> int:
        return len(self.tokens)


@jaxtyped(typechecker=beartype)
def word_tokenize_dataset(
    dataset_or_huggingface_path: dict[str, list[str]] | str,
    context_length: int,
    vocabulary_size: int,
    align: bool,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    splits_for_word_counting: list[str] = ["train"],
) -> tuple[dict[str, Int[Tensor, "dataset_size vocabulary_size"]], WordTokenizer]:
    if isinstance(dataset_or_huggingface_path, str):
        raw_dataset = load_dataset(dataset_or_huggingface_path)
        dataset: dict[str, list[str]] = {
            split_name: [datapoint["text"] for datapoint in split]
            for split_name, split in raw_dataset.items()  # type: ignore
        }
    else:
        dataset = dataset_or_huggingface_path

    tokenizer = WordTokenizer.with_most_frequent_words(
        chain.from_iterable(dataset[split] for split in splits_for_word_counting),
        vocabulary_size=vocabulary_size,
    )

    rng = Generator()
    rng.manual_seed(shuffle_seed)

    tokenized_dataset: dict[str, Int[Tensor, "dataset_size vocabulary_size"]] = {}
    for split_name, split in dataset.items():
        tokenizable: list[str] = [
            text
            for text in tqdm(split, desc="filtering tokenizable")
            if tokenizer.can_tokenize(text)
        ]
        tokenized: list[list[int]] = [
            tokenizer.encode(text) for text in tqdm(tokenizable, desc="tokenizing")
        ]

        if align:
            dataset_size = len(tokenized)
            token_tensor = full(
                size=(dataset_size, context_length), fill_value=tokenizer.eos_token_id
            )
            for i, t in enumerate(tqdm(tokenized, desc="copying into a tensor")):
                t = t[: context_length - 1]
                token_tensor[i, 1 : len(t) + 1] = tensor(t)
        else:
            concatenated: list[int] = []
            for t in tqdm(tokenized, desc="concatenating"):
                concatenated += t
                concatenated.append(tokenizer.eos_token_id)
            while len(concatenated) % context_length != 0:
                concatenated.append(tokenizer.eos_token_id)
            dataset_size = len(concatenated) // context_length
            token_tensor = tensor(concatenated).reshape(dataset_size, context_length)

        if shuffle:
            token_tensor = token_tensor[randperm(dataset_size, generator=rng), :]
        tokenized_dataset[split_name] = token_tensor

    return tokenized_dataset, tokenizer


@beartype
def plot_fraction_tokenizable(
    dataset_huggingface_path: str,
    splits_for_word_counting: list[str],
    text_field: str = "text",
) -> None:
    raw_dataset = load_dataset(dataset_huggingface_path)
    dataset: dict[str, list[str]] = {
        split_name: [
            datapoint[text_field]
            for datapoint in tqdm(
                split, desc=f"converting {split_name} dataset to dict of lists"
            )
        ]
        for split_name, split in raw_dataset.items()  # type: ignore
    }

    def get_tokenized_dataset(split_name: str) -> Iterable[list[str]]:
        return (
            split_words(text)
            for text in tqdm(
                dataset[split_name], desc=f"tokenizing {split_name} dataset"
            )
        )

    word_counts = Counter(
        word
        for split_name in splits_for_word_counting
        for text in tqdm(
            get_tokenized_dataset(split_name),
            desc=f"counting words in {split_name} split",
        )
        for word in text
    )

    words_by_descending_frequency: list[str] = [
        word
        for word, count in sorted(
            list(word_counts.items()),
            reverse=True,
            key=lambda word_and_count: word_and_count[1],
        )
    ]

    for word in words_by_descending_frequency:
        print(word)

    word_to_rank: dict[str, int] = {
        word: rank for rank, word in enumerate(words_by_descending_frequency)
    }

    rank_of_least_frequent_word: dict[str, list[int | float]] = {
        split_name: [
            max(word_to_rank.get(word, inf) for word in tokenized_text)
            for tokenized_text in tqdm(
                get_tokenized_dataset(split_name), desc=f"ranking {split_name} split"
            )
        ]
        for split_name in dataset.keys()
    }

    fig = Figure()
    fig.update_layout(
        title="fraction tokenizable datapoints with different vocabulary sizes<br>note: there are probably off by ones, so don't use this to estimate what happens with a vocabulary size of only a few words (otherwise the off by ones shouldn't matter)",
        xaxis=dict(title="vocabulary size"),
        yaxis=dict(
            range=[0, 1],
            title="fraction tokenizable",
        ),
    )
    for split_name, ranks in rank_of_least_frequent_word.items():
        fig.add_scatter(
            x=sorted(ranks), y=linspace(0, 1, len(ranks)).tolist(), name=split_name
        )
    fig.show()


if __name__ == "__main__":
    # best vocabulary size: 4096
    # this will make 70% of the train and test datasets tokenizable

    plot_fraction_tokenizable(
        dataset_huggingface_path="fhswf/TinyStoriesV2_cleaned",
        splits_for_word_counting=["train"],
    )


# ruff: noqa: F722
