---
title: The Annotated Transformer
summary: Transformer architecture explained with minimal PyTorch implementation line-by-line.
date: 2023-10-26
authors:
  - admin
tags:
  - Transformer
  - LLM
image:
  caption: 'Image credit: [**https://arxiv.org/pdf/1706.03762.pdf**](https://arxiv.org/pdf/1706.03762.pdf)'
---

In the ever-evolving landscape of Artificial Intelligence (AI), one architectual innovation stands out: **Transformer**. This powerful model has completely reshaped the way we approach tasks involving text, vision, audio, etc. In my mind, the biggest meaning of Transformer is that it paves the way for cross-domain applications and enabled advancements in multimodal learning, allowing us to tackle various problems by applying similar principles across different modalities.

In this blog post, we'll embark on a journey to understand what Transformer is, how it works and revolutionizes the AI world. To make our understanding concrete, we will build a Transformer from scratch to solve a English-German translation problem.

<!-- The essential equation of understanding is `understanding = intuition + math + codes`. -->

### Background

The Transformer architecture was first proposed in the paper "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" by Vaswani et al. (2017) for machine translation tasks in natural language processing (NLP). At the heart of NLP lies the challenge of understanding and generating sequences of words. Traditional approaches, like recurrent neural networks (RNNs) and long short-term memory networks (LSTMs), have served as workhorses for sequence modeling. However, they suffer from limitations such as vanishing gradients and inability to capture long-range dependencies. The Transformer addresses these limitations:

- **Long-Term Dependencies**: The self-attention mechanism allows each word to attend to all other words in the input, enabling to capture relationships between distant words in a sequence and modeling of long-term dependencies.

- **Parallelization and Scalability**: The computations of attention scores for different words can be executed concurrently. It enables the training of larger models by distributing computations across multiple processing units.

- **Simplicity and Interpretability**: The Transformer's architecture, based on the self-attention mechanism and feed-forward layers, offers simplicity in implementation compared to complex recurrent structures. Additionally, the attention mechanism provides interpretability, allowing insights into which parts of the input sequence influence specific outputs.

### Models
Here we dissect The Transformer into 6 separate parts and will dedicate a section to each part. In the end we will assemble these sections together to build the Transformer for solving the machine translation task.

> **Tokenization**

Tokenization is the first step for our task which refers to the process of breaking down a text or a sequence of characters into smaller units called tokens, tokens are atomic units used as input to the Transformer model. These tokens could be words, subwords, characters, or even phrases, depending on the granularity of the tokenization strategy. 

Start with a piece of text that you want to process using a Transformer model. This could be a sentence, a paragraph, or even an entire document. The process of tokenization is roughly as follows:

- Text Preprocessing: The text is often preprocessed to remove special characters, punctuation, and extra spaces. It might also involve converting text to lowercase for consistency.

- Tokenization: Words are split into smaller units based on certain rules or algorithms. These units can be characters or subword pieces. We will specifically dive into this part in detail later.

- Special Tokens: The tokenization process adds special tokens to the beginning and end of the token sequence. These special tokens include the `[CLS]` token (for classification tasks) and `[SEP]` token (to mark the end of a sentence or segment). For sequence-to-sequence tasks, an additional `[BOS]` (beginning of sequence) and `[EOS]` (end of sequence) token might be used.

- Vocabulary Building: A vocabulary is created by listing all unique subword tokens present in the text. Each token is then looked up in the model's vocabulary to obtain its corresponding index. The model's vocabulary contains a mapping of tokens to their unique IDs. 

- Padding and Truncation: Transformer models require fixed-length input, so sequences might need to be padded with a special padding token `[PAD]` to match a certain length. Alternatively, if the sequence is too long, it might need to be truncated or discarded.

- Attention Mask: An attention mask is generated to indicate which tokens are actual input `1` and which are padding tokens `0`. This helps the model ignore the padding tokens during computation.

There are three types of tokenization stratedies:

1. Character-level tokenization

Character-level tokenization involves breaking down a text into individual characters. This form of tokenization treats each character as a separate token. For example, the sentence `"Hello, world!"` would be tokenized into `['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']`.
- Pro: Smaller vocabulary size (~26 characters)
- Con: Longer sequences and more computationally expensive to process; Lose semantics of tokens.

2. Word-level tokenization

Word-level tokenization involves breaking down a text into individual words. Each word is treated as a separate token, which makes it easier to analyze and process the text. For example, the sentence `"transformer is a language model developed by Google"`, word-level tokenization would result in the following tokens: `["transformer", "is", "a", "language", "model", "developed", "by", "Google"]`.

- Pro: Captures the meaning of individual words
- Con: Some words might have multiple meanings depending on the context, leading to ambiguity in analysis; Inflected forms (e.g., "running" vs. "run") can lead to different tokens; vocabulary is large and hard to contain rare words.

3. Subword-level Tokenization

Subword tokenization, where words are broken down into smaller units known as subwords. This is particularly useful for handling out-of-vocabulary words and reducing the vocabulary size. For example, the sentence `"transformer transfer well"` might be tokenized into `["trans", "form", "f", "er", "well]`.

- Pro: Subword tokenization can capture variations of words due to inflections, prefixes, and suffixes, which is important for languages with complex morphology; Can handle words that are not present in the vocabulary, allowing models to process rare or out-of-vocabulary terms.
- Con: Subword tokenization often leads to a larger vocabulary compared to word-level tokenization; Some subword units might not carry clear semantic meaning on their own, leading to potential interpretation challenges.

In modern Transformers, subword-level tokenization methods are mainstream because they offer benefits such as handling out-of-vocabulary words and capturing morphological variations. The two most common subword tokenization methods used in modern Transformers are:

- Byte-Pair Encoding (BPE): BPE is a subword tokenization technique that splits words into subword units by iteratively merging the most frequent character pairs. It starts with a vocabulary of individual characters and then progressively merges pairs of characters based on their frequency in the training data. This process generates subword units that can represent both whole words and their parts. It's widely used in models like GPT-2 and GPT-3.

- WordPiece Tokenization: WordPiece is a subword tokenization technique similar to BPE, but it's designed to handle languages with clearer word boundaries. WordPiece starts with a vocabulary of words and iteratively merges the most frequent word pieces. It's widely used in models like BERT and its variants.

```python
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

from config import Config

train_dataset = list(Multi30k(split='train', language_pair=('de', 'en')))

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

de_tokens, en_tokens = [], []
for de_sentence, en_sentence in train_dataset:
    de_tokens.append(de_tokenizer(de_sentence))
    en_tokens.append(en_tokenizer(en_sentence))

de_vocab = build_vocab_from_iterator(
    de_tokens,
    specials=[Config.UNK_SYM, Config.PAD_SYM, Config.BOS_SYM, Config.EOS_SYM],
    special_first=True)
de_vocab.set_default_index(Config.UNK_IDX)

en_vocab = build_vocab_from_iterator(
    en_tokens,
    specials=[Config.UNK_SYM, Config.PAD_SYM, Config.BOS_SYM, Config.EOS_SYM],
    special_first=True)
en_vocab.set_default_index(Config.UNK_IDX)

def de_preprocess(de_sentence):
    tokens = de_tokenizer(de_sentence)
    tokens = [Config.BOS_SYM] + tokens + [Config.EOS_SYM]
    ids = de_vocab(tokens)
    return tokens, ids

def en_preprocess(en_sentence):
    tokens = en_tokenizer(en_sentence)
    tokens = [Config.BOS_SYM] + tokens + [Config.EOS_SYM]
    ids = en_vocab(tokens)
    return tokens, ids

```
