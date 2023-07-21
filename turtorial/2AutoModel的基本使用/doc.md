# AutoModel

AutoModel类及其所有相关项实际上是对库中各种可用模型的简单包装。它是一个聪明的包装器，因为它可以自动猜测检查点的适当模型体系结构，然后用该体系结构实例化模型

## 创建Transformer

### 加载配置对象

```python
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

```

配置包含许多用于构建模型的属性：

```python
print(config)
```

> BertConfig {
>   [...]
>   "hidden_size": 768,
>   "intermediate_size": 3072,
>   "max_position_embeddings": 512,
>   "num_attention_heads": 12,
>   "num_hidden_layers": 12,
>   [...]
> }

hidden*size属性定义了hidden*状态向量的大小

num_hidden_layers定义了Transformer模型的层数

### 不同的加载方式

#### 从默认配置创建模型会使用随机值对其进行初始化

```python
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

# Model is randomly initialized!
```

#### 加载已经训练过的Transformers模型

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

在上面的代码示例中，我们没有使用BertConfig，而是通过Bert base cased标识符加载了一个预训练模型。这是一个模型检查点，由BERT的作者自己训练

#### 保存模型

```python
model.save_pretrained("directory_on_my_computer")
```

会将两个文件保存到磁盘

> config.json pytorch_model.bin

-  config.json 文件，识别构建模型体系结构所需的属性。该文件还包含一些元数据，例如检查点的来源以及上次保存检查点时使用的🤗 Transformers版本。
- *pytorch_model.bin* 文件就是众所周知的*state dictionary*; 它包含模型的所有权重。这两个文件齐头并进；配置是了解模型体系结构所必需的，而模型权重是模型的参数。

### 使用Transformers模型进行推理

```python
sequences = ["Hello!", "Cool.", "Nice!"]
```

 tokenizer 将这些转换为词汇表索引，通常称为 input IDs . 每个序列现在都是一个数字列表！结果是

```python
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
```

这是一个编码序列列表：一个列表列表。张量只接受矩形（想想矩阵）。此“数组”已为矩形，因此将其转换为张量很容易：

```python
import torch

model_inputs = torch.tensor(encoded_sequences)
```

#### 使用张量作为模型的输入

```python
output = model(model_inputs)
```