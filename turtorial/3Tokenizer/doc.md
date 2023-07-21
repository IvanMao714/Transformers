# Tokenizer

## 理论

### 基于词的(Word-based)

基于词的(*word-based*)通常很容易设置和使用，只需几条规则，并且通常会产生不错的结果。例如，在下图中，目标是将原始文本拆分为单词并为每个单词找到一个数字表示

![](img\word_based_tokenization.svg)

有多种方法可以拆分文本。例如，我们可以通过应用Python的`split()`函数，使用空格将文本标记为单词：

```python
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
```

> ['Jim', 'Henson', 'was', 'a', 'puppeteer']

### 基于字符(Character-based)

基于字符的标记器(tokenizer)将文本拆分为字符，而不是单词。这有两个主要好处：

- 词汇量要小得多。
- 词汇外（未知）标记(token)要少得多，因为每个单词都可以从字符构建。

但是这里也出现了一些关于空格和标点符号的问题：

![](img\character_based_tokenization.svg)

## Subword tokenization

子词分词算法依赖于这样一个原则，即不应将常用词拆分为更小的子词，而应将稀有词分解为有意义的子词

![](img\bpe_subword.svg)

### 更多

- Byte-level BPE, 用于 GPT-2
- WordPiece, 用于 BERT
- SentencePiece or Unigram, 用于多个多语言模型

## 实践

### 加载和保存

加载使用与 BERT 相同的检查点训练的 BERT 标记器(tokenizer)与加载模型的方式相同

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

tokenizer("Using a Transformer network is simple")
```

> {'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
>  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
>  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

保存标记器(tokenizer)与保存模型相同:

```python
tokenizer.save_pretrained("directory_on_my_computer")
```

### Tokenization

标记化过程由标记器(tokenizer)的`tokenize()` 方法实现：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

> ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

#### 从词符(token)到输入 ID

```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

> [7993, 170, 11303, 1200, 2443, 1110, 3014]

### 解码

```python
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
```

> 'Using a Transformer network is simple'