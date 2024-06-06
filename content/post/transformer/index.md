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

> **1. Tokenization**

Tokenization is a process which involves splitting input into smaller units, thereby facilitates representation learning and model training. Depend on the modality of input, tokenization have different forms, such as dividing images into smaller patches, converting audio signals into spectrograms, or breaking genome sequences into k-mers. Here we take text tokenization to explain it.

For NLP tasks, tokenization is often the first step for the task which refers to the process of breaking down a text or a sequence of characters into smaller units called tokens, tokens are atomic units used as input to the Transformer model. These tokens could be words, subwords, characters, or even phrases, depending on the granularity of the tokenization strategy. 

Start with a piece of text that you want to process using a Transformer model. This could be a sentence, a paragraph, or even an entire document. The process of tokenization is roughly as follows:

- Text Preprocessing: The text is often preprocessed to remove special characters, punctuation, and extra spaces. It might also involve converting text to lowercase for consistency.

- Tokenization: Words are split into smaller units based on certain rules or algorithms. These units can be characters or subword pieces. We will specifically dive into this part in detail later.

- Special Tokens: The tokenization process often involves adding special tokens to the start and end of a token sequence. These include the `[CLS]` token for classification tasks and the `[SEP]` token to denote the end of a sentence or segment. For sequence-to-sequence tasks, the `[BOS]` (beginning of sequence) and `[EOS]` (end of sequence) tokens are often used. Additionally, unknown tokens `[UNK]` and padding tokens `[PAD]` are incorporated as needed.

- Vocabulary Building: A vocabulary is created by listing all unique subword tokens present in the text. Each token is then looked up in the model's vocabulary to obtain its corresponding index. The model's vocabulary contains a mapping of tokens to their unique IDs. 

- Padding and Truncation: Transformer models require fixed-length input, so sequences might need to be padded with a special padding token `[PAD]` to match a certain length. Alternatively, if the sequence is too long, it might need to be truncated or discarded.

- Attention Mask: An attention mask is created to distinguish between actual input tokens and masked tokens, with `1` indicating the presence of input tokens and `0` indicating padding or subsequent tokens. This helps the model ignore the masked tokens during computation.

There are three types of tokenization stratedies:

1. Character/Byte level tokenization

Character/Byte level tokenization involves breaking down a text into individual characters. This form of tokenization treats each character as a separate token. The difference is that Byte tokenization encodes text characters into bytes using a character encoding scheme like UTF-8. For example, the character 'H' might be represented by the byte value 72 in ASCII encoding.

```python
# Character/Byte level tokenization

text = "Hello, World!"
char_tokens = list(text)
print(char_tokens)

>> ['H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!']
```

```python
# Byte level tokenization

text = "Hello, World!"
byte_tokens = list(text.encode('utf-8'))
print(byte_tokens)

>> [72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33]
```

- Pro: Smaller vocabulary size (~26 characters)
- Con: Result in longer sequences and more computationally expensive to process; Lose semantics of tokens.

2. Word level tokenization

Word-level tokenization involves breaking down a text into individual words. Each word is treated as a separate token, which makes it easier to analyze and process the text. For example, the sentence `"transformer is a language model developed by Google"`, word-level tokenization would result in the following tokens: `["transformer", "is", "a", "language", "model", "developed", "by", "Google"]`.

- Pro: Captures the meaning of individual words
- Con: Some words might have multiple meanings depending on the context, leading to ambiguity in analysis; Inflected forms (e.g., "running" vs. "run") can lead to different tokens; vocabulary is large and hard to contain rare words.

3. Subword level Tokenization

Subword tokenization, where words are broken down into smaller units known as subwords. This is particularly useful for handling out-of-vocabulary words and reducing the vocabulary size. For example, the sentence `"transformer transfer well"` might be tokenized into `["trans", "form", "f", "er", "well]`.

- Pro: Subword tokenization can capture variations of words due to inflections, prefixes, and suffixes, which is important for languages with complex morphology; Can handle words that are not present in the vocabulary, allowing models to process rare or out-of-vocabulary terms.
- Con: Subword tokenization often leads to a larger vocabulary compared to word-level tokenization; Some subword units might not carry clear semantic meaning on their own, leading to potential interpretation challenges.

In modern Transformers, subword-level tokenization methods are mainstream because they offer benefits such as handling out-of-vocabulary words and capturing morphological variations. The two most common subword tokenization methods used in modern Transformers are:

- Byte-Pair Encoding (BPE): BPE is a subword tokenization technique that splits words into subword units by iteratively merging the most frequent character pairs. It starts with a vocabulary of individual characters and then progressively merges pairs of characters based on their frequency in the training data. This process generates subword units that can represent both whole words and their parts. It's widely used in models like GPT-2 and GPT-3.

- WordPiece Tokenization: WordPiece is a subword tokenization technique similar to BPE, but it's designed to handle languages with clearer word boundaries. WordPiece starts with a vocabulary of words and iteratively merges the most frequent word pieces. It's widely used in models like BERT and its variants.

Tokenization is a field of ongoing research, but for our purposes, we rely on Spacy's tokenizer. Spacy employs a rule-based methodology to segment text into tokens, utilizing linguistic rules and heuristics to establish token boundaries within the text.

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
We can obtain the size of vocabulary:

```python
print('de vocab size:', len(de_vocab))
print('en vocab size:', len(en_vocab))

>> de vocab size: 19214
>> en vocab size: 10837
```

See an token example:
```python
de_sentence, en_sentence = train_dataset[0]
print('de preprocess:', *de_preprocess(de_sentence))
print('en preprocess:', *en_preprocess(en_sentence))

>> de preprocess: ['<bos>', 'Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.', '<eos>'] [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3]
>> en preprocess: ['<bos>', 'Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.', '<eos>'] [2, 19, 25, 15, 1169, 808, 17, 57, 84, 336, 1339, 5, 3]
```

> **Positional Encoding** 

In traditional sequence models like RNNs or LSTMs, the order of elements in a sequence is inherently maintained due to the sequential nature of their processing. However, transformers operate in a parallel and non-sequential manner, this parallelism allows transformers to be highly efficient, but it also means that the model lacks the inherent understanding of element positions. To overcome this, positional encodings are added to the input embeddings. Positional encoding is a crucial concept within transformer architectures, ensuring that sequence information is preserved and understood by the model.

Imagine you're translating the sentence "I love dogs" to another language. In a traditional sequence model, the model would process each word one after another, naturally grasping their order. In transformers, without positional encoding, the model might see "I dogs love," disrupting the meaning entirely.

Positional encoding is a technique used to inject information about the position of elements in a sequence into the transformer model. It involves adding unique positional embeddings to the input embeddings of each element. Positional embeddings are essentially fixed vectors associated with each position in the sequence. These vectors are added to the word embeddings before they're input to the transformer layers. By doing this, the model can differentiate between elements based on both their content (semantic meaning) and their position in the sequence.

Positional encoding is often represented using trigonometric functions, specifically sine and cosine functions. This choice ensures that the positional embeddings can capture different frequencies and positions without repetition.

```python
class EmbeddingWithPosition(nn.Module):

    def __init__(self, args, vocab_size, dropout=0.1):
        super().__init__()
        self.seq_emb = nn.Embedding(vocab_size, args.d_model)
        position_idx = torch.arange(0, args.max_seq_len).unsqueeze(-1)
        position_emb_fill = position_idx * torch.exp(
            -torch.arange(0, args.d_model, 2) * math.log(10000.0) / args.d_model)
        pos_encoding = torch.zeros(args.max_seq_len, args.d_model)
        pos_encoding[:, 0::2] = torch.sin(position_emb_fill)
        pos_encoding[:, 1::2] = torch.cos(position_emb_fill)
        self.register_buffer('pos_encoding', pos_encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (batch_size, seq_len)
        x = self.seq_emb(x)  # x: (batch_size, seq_len, d_model)
        x = x + self.pos_encoding.unsqueeze(0)[:, :x.size()[1], :]
        return self.dropout(x)
```

see a embedding example:

```python
from dataset import de_preprocess, de_vocab, train_dataset

emb = EmbeddingWithPosition(Config, len(de_vocab))
de_tokens, de_ids = de_preprocess(train_dataset[0][0])
de_ids_tensor = torch.tensor(de_ids, dtype=torch.long).unsqueeze(0)
emb_result = emb(de_ids_tensor)
print('de_tensor_size:', de_ids_tensor.size(), 'emb_size:', emb_result.size())

>> de_tensor_size: torch.Size([1, 15]) emb_size: torch.Size([1, 15, 512])
```

> **Attention Mechanism**

Unlike their predecessors, which relied on recurrent or convolutional layers, transformers introduce the concept of **self-attention**. This mechanism involves comparing each word's importance to the other words in the same sentence. This attention score dictates how much emphasis the model places on each word while generating an output. This core mechanism has proven to be incredibly effective for processing sequential data, such as text.

> **Transformer Architecture**

The transformer architecture consists of an encoder and a decoder, each composed of multiple layers. The encoder processes the input text and converts it into a dense representation, while the decoder generates the output. What's fascinating is that each layer operates independently, allowing for parallelization and significantly speeding up training time.

#### Pre-training and Fine-tuning
One of the most significant advantages of transformers is their ability to be pre-trained on massive text corpora. During pre-training, models learn contextualized representations of words. Fine-tuning follows pre-training, where models are adapted to specific NLP tasks, such as text classification or machine translation. This two-step process not only speeds up training but also allows for transfer learning, where pre-trained models are fine-tuned on smaller datasets. Pre-trained models like BERT, GPT, and T5 have become the backbone of many NLP projects. Researchers and practitioners can fine-tune these models on specific tasks, achieving state-of-the-art performance even with limited labeled data.

