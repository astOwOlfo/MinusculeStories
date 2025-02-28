from openai import AsyncOpenAI
from argparse import ArgumentParser
import json
import asyncio
from tqdm import trange
from beartype import beartype

from word_tokenizer import split_words


WRITE_STORY_PROMPT = """Plaese write a short story using only the following words and punctiation symbols. You may not use any words not in the following list, even if they are variations of the words in the list, e.g. you may not use a plural of a word if the word is in the list but not the plural. You my only use punctuation symbols that are in the list.

The story should be:
- Simple - talk about simple things, use simple sentences, etc.
- Original.
- Coherent - prioritize the story making sense.

The story should be a couple sentence long.

Please output the story and no other words whatsoever.

The words and symbols you may use are: {allowed_words}"""


@beartype
async def generate_single_story(
    allowed_words: list[str], model: str, client: AsyncOpenAI
) -> str | None:
    prompt = WRITE_STORY_PROMPT.format(allowed_words=" ".join(allowed_words))

    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model, temperature=1
        )  # type: ignore
        story = response.choices[0].message.content
    except Exception as e:
        print(e)
        return None

    if story is None:
        return None

    if not all(word in allowed_words for word in split_words(story)):
        return None

    return story


@beartype
async def generate_stories(
    allowed_words: list[str],
    model: str,
    client: AsyncOpenAI,
    n_api_calls: int,
    batch_size: int,
) -> list[str]:
    stories: list[str] = []

    for _ in trange(n_api_calls // batch_size):
        new_stories: list[str | None] = await asyncio.gather(
            *[
                generate_single_story(
                    allowed_words=allowed_words, model=model, client=client
                )
                for _ in range(batch_size)
            ]
        )

        for new_story in new_stories:
            if new_story is None:
                continue
            stories.append(new_story)

    return stories


@beartype
def main() -> None:
    argparse = ArgumentParser()
    argparse.add_argument("--model", type=str, default="gpt-4o-mini")
    argparse.add_argument("--api-key", type=str)
    argparse.add_argument("--n-api-calls", type=int, required=True)
    argparse.add_argument("--allowed-words-file", type=str, required=True)
    argparse.add_argument("--output", type=str, required=True)
    argparse.add_argument("--batch-size", type=int, default=32)
    args = argparse.parse_args()

    with open(args.allowed_words_file) as f:
        allowed_words = [word.strip().lower() for word in f if word.strip() != ""]

    stories: list[str] = asyncio.run(
        generate_stories(
            allowed_words=allowed_words,
            model=args.model,
            client=AsyncOpenAI(api_key=args.api_key),
            n_api_calls=args.n_api_calls,
            batch_size=args.batch_size,
        )
    )

    print(
        f"Generated {len(stories)} stories. {len(stories) / args.n_api_calls:.2%} api calls returned valid stories."
    )

    with open(args.output, "w") as f:
        for story in stories:
            f.write(json.dumps({"text": story}))
            f.write("\n")


if __name__ == "__main__":
    main()
