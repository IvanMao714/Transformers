## Pipeline

它将模型与其必要的预处理和后处理步骤连接起来，使我们能够直接输入任何文本并获得可理解的答案

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

> [{'label': 'POSITIVE', 'score': 0.9598047137260437}]

它可以放置多个句子

```python
classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
```

> [{'label': 'POSITIVE', 'score': 0.9598047137260437},
>  {'label': 'NEGATIVE', 'score': 0.9994558095932007}]

### 现在可以使用的pipeline模型

- `feature-extraction` (get the vector representation of a text)
- `fill-mask`
- `ner` (named entity recognition)
- `question-answering`
- `sentiment-analysis`
- `summarization`
- `text-generation`
- `translation`
- `zero-shot-classification`

#### Zero-shot-classification 零样本分类

对未标记的文本进行分类。这是现实世界项目中的常见场景，因为注释文本通常很耗时并且需要领域专业知识。对于这个用例，它允许您指定要使用哪些标签进行分类，因此您不必依赖预训练模型的标签

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

> {'sequence': 'This is a course about the Transformers library',
>  'labels': ['education', 'business', 'politics'],
>  'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}

#### Text generation

现在让我们看看如何使用管道生成一些文本。这里的主要思想是您提供一个提示，模型将通过生成剩余的文本来自动完成它。

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

- num_return_sequences: 生成多少个不同的序列
- max_length: 输出文本的总长度

#### 用任意模型在pipe中

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

> [{'generated_text': 'In this course, we will teach you how to manipulate the world and '
>                     'move your mental and physical capabilities to your advantage.'},
>  {'generated_text': 'In this course, we will teach you how to become an expert and '
>                     'practice realtime, and with a hands on experience on both real '
>                     'time and real'}]

#### The Inference API

所有模型都可以通过API直接在浏览器测试

#### Mask filling

这个任务的想法是填补给定文本中的空白：

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

> [{'sequence': 'This course will teach you all about mathematical models.',
>   'score': 0.19619831442832947,
>   'token': 30412,
>   'token_str': ' mathematical'},
>  {'sequence': 'This course will teach you all about computational models.',
>   'score': 0.04052725434303284,
>   'token': 38163,
>   'token_str': ' computational'}]

 top_k：控制想要显示多少种可能性

#### Named entity recognition

模型必须找到输入文本的哪些部分对应于实体，例如人、位置或组织。

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

