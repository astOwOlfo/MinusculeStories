import torch
from torch import tensor
from torch.utils.data import TensorDataset
from random import Random
import json
from beartype import beartype


from transformer import TransformerConfig, print_parameter_counts
from word_tokenizer import word_tokenize_dataset
from train import TrainingConfig, train
from evaluate_on_winogrande import load_winogrande_questions, winogrande_accuracy
from helper import run_or_load


@beartype
def load_minuscule_stories_dataset(
    filename: str = "data/minuscule-stories.jsonl",
    test_fraction: float = 0.05,
    deduplicate: bool = True,
    shuffle_seed: int = 42,
) -> dict[str, list[str]]:
    with open(filename) as f:
        stories: list[str] = [
            json.loads(line)["text"] for line in f if line.strip() != ""
        ]

    if deduplicate:
        length_before_deduplication = len(stories)
        stories = list(set(stories))
        print(
            f"Deduplicating the minuscule stories dataset removed {length_before_deduplication - len(stories)} stories ({(length_before_deduplication - len(stories)) / length_before_deduplication:.4%})."
        )

    Random(shuffle_seed).shuffle(stories)

    n_train = int((1 - test_fraction) * len(stories))
    train_stories = stories[:n_train]
    test_stories = stories[n_train:]

    return {"train": train_stories, "test": test_stories}


@beartype
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using", device)

    context_length: int = 64
    vocabulary_size: int = 256

    dataset, tokenizer = run_or_load(
        f"data/tokenized-minuscule-stories-dataset-context-length-{context_length}-vocabulary-size-{vocabulary_size}.pickle",
        word_tokenize_dataset,
        load_minuscule_stories_dataset(),
        context_length=context_length,
        vocabulary_size=vocabulary_size,
        align=True,
    )

    dataset = {
        split_name: TensorDataset(tokens[:, :-1], tokens[:, 1:])
        for split_name, tokens in dataset.items()
    }

    print(
        "dataset size:",
        {split_name: len(split) for split_name, split in dataset.items()},
    )

    model, stats = run_or_load(
        "models/minuscule_stories_small.pickle",
        train,
        TrainingConfig(
            model_config=TransformerConfig(
                context_length=context_length,
                vocabulary_size=tokenizer.vocabulary_size(),
                n_layers=4,  # 16,
                d_model=32,
                n_heads=4,
                d_head=8,
                d_mlp=128,
                activation_function="silu",
                rotary_positional_embedding_base=1_000,
                task="next_token",
            ),
            epochs=32,
            learning_rate=1e-3,
            cosine_learning_rate_schedule=True,
            batch_size=1024,
            device=device,
        ),
        train_dataset=dataset["train"],
        test_dataset=dataset["test"],
    )

    print_parameter_counts(model)

    stats.plot()

    generated = model.generate(
        tensor([[tokenizer.eos_token_id] for _ in range(16)], device=device),
        n_new_tokens=context_length,
        temperature=1.0,
    )
    for gen in generated:
        print("GENERATED:", tokenizer.decode(gen.tolist()))

    questions = load_winogrande_questions("data/minuscule-winogrande-train.json")
    print("winogrande accuracy:", winogrande_accuracy(model, tokenizer, questions))


if __name__ == "__main__":
    main()
