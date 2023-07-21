# Multiple sequences

一批序列输入，会出现错误

```python
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tf.constant(ids)
# This line will fail.
model(input_ids)
```

> InvalidArgumentError: Input to reshape is a tensor with 14 values, but the requested shape has 196 [Op:Reshape]

查看分词器

```python
tokenized_inputs = tokenizer(sequence, return_tensors="tf")
print(tokenized_inputs["input_ids"])
```

> tf.Tensor: shape=(1, 16), dtype=int32, numpy=
> array([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662,
>         12172,  2607,  2026,  2878,  2166,  1012,   102]], dtype=int32)>

重试并添加一个新维度

```python
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = tf.constant([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

打印输入ID以及生成的logits-以下是输出

> Input IDs: tf.Tensor(
> [[ 1045  1005  2310  2042  3403  2005  1037 17662 12172  2607  2026  2878
>    2166  1012]], shape=(1, 14), dtype=int32)
> Logits: tf.Tensor([[-2.7276208  2.8789377]], shape=(1, 2), dtype=float32)

*Batching* 是一次通过模型发送多个句子的行为。如果你只有一句话，你可以用一个序列构建一个批次

## Attention masks

*Attention masks*是与输入ID张量形状完全相同的张量，用0和1填充：1s表示应注意相应的标记，0s表示不应注意相应的标记（即，模型的注意力层应忽略它们）

```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(tf.constant(batched_ids), attention_mask=tf.constant(attention_mask))
print(outputs.logits)
```

> tf.Tensor(
> [[ 1.5693681  -1.3894582 ]
>  [ 0.5803021  -0.41252586]], shape=(2, 2), dtype=float32)

# Whole Processes

首先准备通过模型传递的输入

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

`model_inputs` 变量包含模型良好运行所需的一切。对于DistilBERT，它包括输入 ID和注意力掩码(attention mask)。

```python
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

它还一次处理多个序列，并且API没有任何变化：

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences)
```

它可以根据几个目标进行填充：

```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

它还可以截断序列:

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

标记器对象可以处理到特定框架张量的转换，然后可以直接发送到模型

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## 特殊词符(token)

如果我们看一下标记器返回的输入 ID，我们会发现它们与之前的略有不同：

```python
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```

> [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
> [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]

一个在开始时添加了一个标记(token) ID，一个在结束时添加了一个标记(token) ID。让我们解码上面的两个ID序列，看看这是怎么回事：

```python
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
```

标记器在开头添加了特殊单词`[CLS]`，在结尾添加了特殊单词`[SEP]`。这是因为模型是用这些数据预训练的，所以为了得到相同的推理结果，我们还需要添加它们。请注意，有些模型不添加特殊单词，或者添加不同的单词；模型也可能只在开头或结尾添加这些特殊单词。在任何情况下，标记器都知道需要哪些词符，并将为您处理这些词符。

现在我们已经看到了标记器对象在应用于文本时使用的所有单独步骤，让我们最后一次看看它如何处理多个序列（填充！），非常长的序列（截断！），以及多种类型的张量及其主要API：

```python
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
output = model(**tokens)
```

** ： 调用函数时的关键字参数放入一个字典中