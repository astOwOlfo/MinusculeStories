from argparse import ArgumentParser
import asyncio
import json
from anthropic import AsyncAnthropic
from datasets import load_dataset
import re
from dataclasses import asdict, dataclass
from tqdm import tqdm
from more_itertools import chunked
from typing import Literal
from beartype import beartype

from word_tokenizer import WordTokenizer

# WinoGrande is a benchmark dataset designed to test a model's common sense reasoning abilities through pronoun resolution problems. It presents sentences with ambiguous pronouns where resolving the correct referent requires understanding context and world knowledge, making it a challenging test of machine comprehension that's difficult to solve through statistical patterns alone.


RESTRICT_TO_ALLOWED_WORDS_PROMPT = """I would like to make a benchmark similar to WinoGrande but that only uses words from a small set of words.
Please design a question with the exact same structure as the following WinoGrande question, but only using the allowed words.
Please reason out loud about your approach to designing the question and whether it's possible. After that, please write the question, answers options, and correct answer formatted with xml tags, exactly like in the example below.
If the allowed set of words is not expressive enough for this, do not generate a question and say so instead (in this case, do not use xml tags).

Before you write anything between xml tags, you should write the question in plain English and then check:
- The sentence makes sense and is not weird.
- There is only one correct answer and it is clear which one it is.
- The question you generated tests understanding of the same linguistic pattern as the original WinoGrande question you were asked to base yours on.
If any of those conditions is not verified, say so and give up without outputting any xml tags.

The words and punctuation symbols you are allowed to use are the following: {allowed_words}

Note that you are not allowed to use different forms of those words. For example, if the plural of a word is not in the list while the word is, you are allowed to use the word but not the plural.
You are only allowed to use the punctuation symbols in the list above, as well as _ for the blank.

<sentence>
{sentence}
</sentence>

<option1>
{option1}
</option1>

<option2>
{option2}
</option2>

<correct_answer>
{correct_answer}
</correct_answer>
"""


SOLVE_WINOGRANDE_QUESTION_PROMPT = """Which one of the following options is most appropriate for filling in the blank in the following sentence?
Please include either <option1/> or <option2/> in your answer, depending on which one it is.
If it is unclear, or if the sentence doesn't make sense either way, say so and do not include any xml tags in your response.

Sentence: {sentence}
Option 1: {option1}
Option 2: {option2}
"""


@beartype
@dataclass(frozen=True)
class WinoGrandeQuestion:
    sentence: str
    option1: str
    option2: str
    answer: Literal[1, 2]


@beartype
async def make_single_minuscule_winogrande_question(
    based_on_question: WinoGrandeQuestion,
    allowed_words: list[str],
    model: str,
    client: AsyncAnthropic,
) -> WinoGrandeQuestion | None:
    prompt = RESTRICT_TO_ALLOWED_WORDS_PROMPT.format(
        allowed_words=" ".join(allowed_words),
        sentence=based_on_question.sentence,
        option1=based_on_question.option1,
        option2=based_on_question.option2,
        correct_answer=[based_on_question.option1, based_on_question.option2][
            based_on_question.answer - 1
        ],
    )

    response = await client.messages.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=4096,
    )
    completion: str = response.content[0].text  # type: ignore

    sentence = extract_xml_tag_content(completion, tag_name="sentence")
    if sentence is None:
        return None

    option1 = extract_xml_tag_content(completion, tag_name="option1")
    if option1 is None:
        return None

    option2 = extract_xml_tag_content(completion, tag_name="option2")
    if option2 is None:
        return None

    correct_answer = extract_xml_tag_content(completion, tag_name="correct_answer")
    if correct_answer is None:
        return None

    if correct_answer not in [option1, option2]:
        return None

    answer = [option1, option2].index(correct_answer) + 1

    if sentence.count("_") != 1:
        return None

    for option in [option1, option2]:
        if not WordTokenizer(allowed_words).can_tokenize(sentence.replace("_", option)):
            return None

    return WinoGrandeQuestion(
        sentence=sentence,
        option1=option1,
        option2=option2,
        answer=answer,  # type: ignore
    )


@beartype
def extract_xml_tag_content(text: str, tag_name: str) -> str | None:
    matches = re.findall(f"<{tag_name}>(.*?)</{tag_name}>", text, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


@beartype
async def claude_solves_winogrande_question_correctly(
    question: WinoGrandeQuestion, model: str, client: AsyncAnthropic
) -> bool:
    prompt = SOLVE_WINOGRANDE_QUESTION_PROMPT.format(
        sentence=question.sentence, option1=question.option1, option2=question.option2
    )
    response = await client.messages.create(
        messages=[{"role": "user", "content": prompt}], model=model, max_tokens=4096
    )
    completion: str = response.content[0].text  # type: ignore

    chose_option1 = "<option1/>" in completion
    chose_option2 = "<option2/>" in completion

    if chose_option1 and chose_option2:
        return False

    match question.answer:
        case 1:
            return chose_option1
        case 2:
            return chose_option2
        case _:
            assert False


@beartype
async def make_minuscule_winogrande_questions(
    based_on_questions: list[WinoGrandeQuestion],
    allowed_words: list[str],
    model: str,
    client: AsyncAnthropic,
    batch_size: int,
) -> list[WinoGrandeQuestion]:
    minuscule_questions: list[WinoGrandeQuestion] = []

    for based_on_questions_batch in tqdm(
        list(chunked(based_on_questions, batch_size)),
        desc="generating minuscule winogrande",
    ):
        new_minuscule_questions: list[str | None] = await asyncio.gather(  # type: ignore
            *[
                make_single_minuscule_winogrande_question(
                    based_on_question=question,
                    allowed_words=allowed_words,
                    model=model,
                    client=client,
                )
                for question in based_on_questions_batch
            ]
        )  # type: ignore

        new_minuscule_questions: list[str] = [
            question for question in new_minuscule_questions if question is not None
        ]

        claude_solves: list[bool] = await asyncio.gather(
            *[
                claude_solves_winogrande_question_correctly(
                    question=question,  # type: ignore
                    model=model,
                    client=client,
                )
                for question in new_minuscule_questions
            ]
        )

        new_minuscule_questions = [
            question
            for question, solves in zip(new_minuscule_questions, claude_solves)
            if solves
        ]

        minuscule_questions += new_minuscule_questions # type: ignore

    return minuscule_questions


@beartype
def load_winogrande_questions(
    dataset_huggingface_path: str, dataset_subset: str, dataset_split: str
) -> list[WinoGrandeQuestion]:
    dataset = load_dataset(
        dataset_huggingface_path, dataset_subset, split=dataset_split
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
def save_winogrande_questions(
    questions: list[WinoGrandeQuestion], filename: str
) -> None:
    questions_as_dict = [
        {key: str(value) for key, value in asdict(question).items()}
        for question in questions
    ]

    with open(filename, "w") as f:
        json.dump(questions_as_dict, f)


@beartype
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--allowed-words-filename", type=str, required=True)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219")
    parser.add_argument("--api-key", type=str)
    parser.add_argument(
        "--based-on-dataset-huggingface-path",
        type=str,
        default="allenai/winogrande",
    )
    parser.add_argument("--dataset-subset", type=str, default="winogrande_xl")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
        with open(args.output, "w") as f:
            f.write("test")
    except Exception:
        print("Could not create output file.")
        exit(1)

    client = AsyncAnthropic(api_key=args.api_key)

    with open(args.allowed_words_filename) as f:
        allowed_words = [word.strip().lower() for word in f if word.strip() != ""]

    based_on_questions = load_winogrande_questions(
        args.based_on_dataset_huggingface_path,
        dataset_subset=args.dataset_subset,
        dataset_split=args.dataset_split,
    )

    if args.max_attempts is not None:
        based_on_questions = based_on_questions[: args.max_attempts]

    minuscule_questions = asyncio.run(
        make_minuscule_winogrande_questions(
            based_on_questions=based_on_questions,
            allowed_words=allowed_words,
            model=args.model,
            client=client,
            batch_size=args.batch_size,
        )
    )

    save_winogrande_questions(minuscule_questions, args.output)


if __name__ == "__main__":
    main()
