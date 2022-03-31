# anthevec
A Python library to get word-level vectors from BERT-like transformer models

## What is anthevec?

For NLP applications or linguistic research, it is useful to represent words as continuous vectors. Continuous vector representations allow for (pseudo) semantic generalisation across vectors, and are favoured over one-hot encoding.

One source of such vectors can be BERT transformer models. Through a series of language modelling tasks, pieces of words get assigned continuous vector representations, which in BERT models are even further adapted to the context of the sentence in which the word pieces occur (so-called "contextualisation"). Notice, however, that BERT transformer models operate on the level of the *word piece*, not on the level of the word. Word pieces are great for generalisation across words, and thus for language modelling performance, but they do make it hard to ask for a vector of a specific word (since what we humans see as a word can actually consist of different pieces with their own vectors in BERT).

To solve this issue, I designed two small libraries: *FriendlyTokenizer* and *EmbeddingRetriever*, both part of the `anthevec` package (for lack of a better name). FriendlyTokenizer is a wrapper around the BERT model tokeniser, which complements this tokeniser with a spaCy tokeniser. The spaCy tokeniser is used to create a "human" tokenisation (on the word level, as we generally expect). Then, the FriendlyTokenizer wrapper matches the BERT word pieces with the tokenisation done by spaCy. This is possible because the [HuggingFace transformers library](https://huggingface.co/docs/transformers/index) provides the span range to which each word piece corresponds.

EmbeddingRetriever uses FriendlyTokenizer's correspondence between words and word pieces to piece together a vector from the different word pieces of which a word consists. To create this single vector, the average of all the word pieces is taken. It is possible to ask for the vector on a specific layer, or to take an average across multiple layers.

Note: The [*token\_to\_chars*](https://huggingface.co/docs/transformers/main_classes/tokenizer\#transformers.BatchEncoding.token_to_chars) method, which these libraries use, is only available for models which have a "fast" tokeniser. This limits the applicability of FriendlyTokenizer to only those models which have a fast tokeniser available. Fast tokenisers are written in Rust, while non-fast tokenisers are purely Pythonic.

## Installing anthevec

anthevec is not available on any Python package manager (yet). To use it, simply copy the `anthevec` folder from this repository to your Python script's directory. From there, you can simply import the libraries like you would for any other package:

```python
from anthevec.embedding_retriever import EmbeddingRetriever
from anthevec.friendly_tokenizer import FriendlyTokenizer
```

## Using anthevec

You should generally only interact with EmbeddingRetriever. EmbeddingRetriever uses FriendlyTokenizer under the hood, but by itself, FriendlyTokenizer only supplies the correspondence between spaCy tokens and their corresponding BERT word pieces.

### Prerequisites

Before using EmbeddingRetriever, you should have initialised the following three objects which will be passed as parameters:

- a HuggingFace transformers model object **with hidden state output**
- a HuggingFace transformers **fast** tokeniser
- a spaCy tokeniser

I do not currently have a `requirements.txt`, but it is obvious that you'll need both the [transformers](https://github.com/huggingface/transformers) package, as well as [spaCy](https://github.com/explosion/spaCy). In addition, your HuggingFace transformers tokeniser and spaCy tokeniser should (of course) be for the same language.

To correctly initialise your HuggingFace transformers model, refer to the following snippet:
```python
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_NAME = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
bert_model = AutoModel.from_pretrained(MODEL_NAME, config=config)
```
The snippet initialises a fast tokeniser for an Italian BERT model, and also creates a model object which will output hidden states (`output_hidden_states=True`).

To correctly initialise your spaCy tokeniser, refer to the following snippet:
```python
import spacy
nlp = spacy.load("it_core_news_sm")
```
The snippet initialises a spaCy tokeniser for an Italian pipeline. Refer to the [spaCy documentation](https://spacy.io/usage/models) to look for pipelines in your language and find out how to install them.

### Calling EmbeddingRetriever

First, create a new EmbeddingRetriever. It takes four arguments:

- your HuggingFace transformers model object
- your HuggingFace transformers fast tokeniser
- your spaCy tokeniser
- a list of sentences of which you want to get word-level vectors

```python
embedding_retriever = EmbeddingRetriever(bert_model, tokenizer, nlp, [ "Il gatto beve", "Le gatte bevono" ])
```

To get the hidden state of a specific word, use the `get_hidden_state()` method. It takes three arguments:

- the index of the sentence (for "Le gatte bevono", we would enter `1`)
- the index of the spaCy token (for "bevono", we would enter `2`)
- the hidden layer(s) which we want to base our token vector on (list!)
	- `0`: embedding layer
	- `1`-`12`: contextualised layers 1 through 12
	- if the list contains multiple layers, an average across the vectors of these layers will be made

```python
sentence_index = 1
token_index = 2
layers = [11, 12]
hidden_state = embedding_retriever.get_hidden_state(sentence_index, token_index, layers)
```

Tip: you can find out the spaCy tokenisation by using the `embedding_retriever.tokens` property. This property contains a list of all spaCy tokens, the indices of which are interesting for use in the `get_hidden_state()` method. You should refer to the [spaCy documentation for the Token type](https://spacy.io/api/token) for more information, but the snippet below shows how to use the list of tokens to find the index of a specific word in the token list:

```python
# The token corresponding to our type might be inflected, or in some other form.
# We use the spaCy tokenised forms to find the corresponding lemmas
lemmas = list(map(lambda token: token.lemma_, embedding_retriever.tokens[sentence_index]))

# The index of the token is the index of the type we are interested in
# e.g. "I am going to the supermarket"
#      "I be go to the supermarket" (lemma form)
# lemma = go, index = 2 -> we find "going" in the token list
token_index = lemmas.index(self.lemma)
```

## Future work

- improve code quality
- use the attention layers to make a weighted average of the different word pieces to form a word vector