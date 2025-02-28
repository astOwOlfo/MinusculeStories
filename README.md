# MinusculeStories
We attempt to make a dataset even simpler than [TinyStories](https://arxiv.org/abs/2305.07759) with the goal of being able to train extremely small models that can speak coherent English.
We expect this to be useful to people who do bottom-up interpretability. See [this discussion](https://www.lesswrong.com/posts/cCgxp3Bq4aS9z5xqd/lucius-bushnaq-s-shortform?commentId=vM6aBsXKe5PMBwLwd) for motivation on why one would want to do bottom up interpretability with the goal of understanding how very simple models speak coherent English at all.

To do this, we synthesize a dataset of 320k few sentence long stories with only the 256 most common words from the TinyStories dataset. An 83k parameter model trained on this dataset cn speak mostly coherent English (whereas the smallest TinyStories model has 1m parameters).

Additionally, we generate a dataset similar to [WinoGrande](https://arxiv.org/abs/1907.10641), but only using the allowed 256 words. We test our models on it, but they don't achieve a better accuracy than random baseline.

## Datasets

### Minuscule Stories

We collected the 256 most common words in [TinyStoriesV2_cleaned](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned) and ask GPT-4o mini to generate few sentence long stories only using those words. We then only keep the stories which actually only use those words (~25% of the stories). Generating 320k such stories cost $130.

Sample of such stories:
- Once there was a boy named Tim. He had a big dog named Max. They lived in a small house by the park. One day, they saw a bird. Tim wanted to play with it, but the bird flew away. Tim felt sad, but then he found a toy in the box. It was a pretty car. Tim smiled and played with Max. They had a good time together.
- Tom and Lily had a big dog. They liked to play at the park. One day, they saw a bird. It was blue and shiny. "Look!" Tom said. They ran to see. Max, their dog, ran too. It was fun! They felt happy.
- Once there was a big dog named Max. He had a small friend named Lily. They liked to play in the park. One day, they saw a red ball. Max ran to it and picked it up. Lily smiled and they played together. It was a happy time!
- Once there was a boy named Tom. He had a dog named Max and a cat named Lily. Tom and Max played in the park every day. One day, they saw a big bird. Tom smiled and said, "Look, Max!" Max ran to the bird, but it flew away. Tom felt sad, but then he found a toy. "This is fun!" he said. Together, they ran home.

Upon inspecting the dataset manually, we notice that it is very monotone and the stories are very similar. However, note that deduplicating it only removes 110 stories (0.034%).

### Minuscule WinoGrande

For questions from the WinoGrande dataset, an early NLP dataset designed to test whether models have a basic understanding of English, we ask Claude 3.7 Sonnet to generate questions that test knowledge of the same linguistic pattern but only use the allowed 256 words. We tell it to skip questions it is unable to rephrase. Then, we filter out the questions which Claude answers wrong. Unfortunately, upon manual inspection, a significant fraction of the questions are bad.

Here is a sample of questions we get:
- Lucy looked up and saw Sara in the tree, as _ was very high. Correct answer: Sara, Incorrect answer: Lucy
- Lucy saw the cat and the bird, but the _ flew away fast. Correct answer: bird, Incorrect answer: cat
- Tim played with the ball and his dog, but then he put the _ away. Correct answer: ball, Incorrect answer: dog
- Tom liked his small dog but he loved his big cat. He played with the _ all day. Correct answer: cat, Incorrect answer: dog

Unfortunately, we find that models we trained on Minuscule Stories do not achieve an above random baseline accuracy on Minuscule WinoGrande.

## Results

When we train an 83k parameter autoregressive transformer on 256k train minuscule stories for 32 epochs with context length, the model manages to generate mostly coherent English text. Here are examples of stories the model generated (note that the lowercase and spaces before punctuation are because of the simplified tokenizer we use, and the abrupt truncations are due to only using a context length of 64 tokens):

- once there was a boy named tim . he had a big dog named max . they went to the park to play . tim saw a shiny red ball . max ran to it and picked it up . tim laughed . they had fun together under the sun . tim felt happy . " thank you , " he said .
- once there was a girl named lily . she had a big dog named max . they played together in the park every day . one day , lily ran to a tree and saw a red bird . " look , max ! " she said . max looked too . they felt happy and wanted to play more . suddenly , the
- tom and lily went to the park . they saw a big dog and a little cat . the dog ran fast . " can we play with them ? " he said . they laughed and ran around . it was a fun day .
- once there was a big dog named max . he played with a little boy , tim , in the park . they ran and had fun and loved after a ball . one day , they saw a bird in the tree . max was scared and wanted to help . tim said , " it ' s a good friend ! "
- once upon a time , a boy named tim had a dog named max . they played every day in the park . one day , they saw a bird . tim smiled and said , " look at that bird ! " they felt happy . then , they ran back home .
-  lily had a dog named max . max was big and happy . they played every day in the park . one day , they saw a bird . the bird flew up high . lily smiled and said , " look , max ! " max ran after the bird . they had fun together .
- once upon a time , a boy named tim had a little dog . they played and found a ball in the sun . tim was happy and said , " look ! " his dog ran around the park . they were good friends under the big tree . tim ' s mom smiled at them . " what a good day !
- it was a big day . tim and his dog , max , went to the park . they saw a bird . tim said , " look ! " max ran fast . they played for a long time . tim felt happy . then they saw a shiny bug . suddenly came down . the bird flew up high . tom looked
- once there was a boy named tim . he had a big dog named max . they played in the park and had fun . one day , tim saw a bird in a tree . max ran to find it . tim looked up and smiled . " what is it ? " he asked . max , they played together for a
- once upon a time , a boy named tim had a dog named max . they played together in the park every day . one day , tim saw a big bird in a tree . he wanted to find it . " can you get it ? " he asked max . max ran up and down , " good dog ! "
- once there was a dog named max . he had a big ball and was very happy ! one day , he ran to the park to play . his friend tom was there too , too . they saw a bird and smiled . it was a new toy . max went to play with it . they felt good together .
- tom had a dog named max . they played in the park with a big ball . max ran after it and found a small bird . " good boy , " tom said , " we can run ! " they felt happy together .
- once , there was a dog named max . he had a big ball . max and his friend mia played in the park . they ran around and had fun . max saw a bird up in a tree . he wanted to help it . " can we help ? tom asked tom . she said yes ! they laughed and played
- once there was a big dog named max . he had a ball and liked to play with his friends . one day , they went to the park to run and play . they had a lot of fun and felt very happy . max saw a little bird and wanted to run . the bird looked scared , but max told him
- once upon a time , a boy named tom had a big dog . they would play in the park all day . tom loved to run and play with the ball . one day , he saw a bird in a tree and felt happy . he asked his dog to be careful . they both ran up to find a bird .

The model does not achieve above random baseline accuracy on Minuscule WinoGrande, and neither does a 562k model we trained the same way.


## Running

Setup
```
$ pip install -r requirements.txt
```

Train model, print stories it generate and evaluate in on Minuscule WinoGrande.
To modify hyperparameters, edit `train.py`.
```
$ python train.py
```

Generate Minuscule Stories:
```
$ python generate_minuscule_stories.py --n-api-calls 1250000 --output data/minuscule-stories.jsonl --allowed-words-file <(head -n 256 data/tiny-stories-words-by-frequency) --api-key <your openai api key> --batch-size 256
```

Generate Minuscule WinoGrande:
```
$ ANTHROPIC_API_KEY=<your anthropic api key> python make_minuscule_winogrande.py --output data/minuscule-winogrande-train.
json --allowed-words-filename  <(head -n 256 data/tiny-stories-words
-by-frequency) --max-attempts 2048 --dataset-split train
```
