# Mini GPT - 从零学习 LLM

一个最小化的 GPT 实现，用于学习理解大语言模型的核心原理。

## 项目结构

```
mini-gpt/
├── model.py       # GPT 模型（Transformer 架构）
├── tokenizer.py   # 字符级分词器
├── train.py       # 训练脚本
├── generate.py    # 文本生成
├── data/
│   └── input.txt  # 训练数据
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型（CPU 约 5-10 分钟）
python train.py

# 3. 生成文本
python generate.py "To be or"
```

## 核心概念

### 1. Tokenizer（分词器）
将文本转换为数字序列。这里用最简单的字符级分词。

### 2. Embedding（嵌入）
- Token Embedding: 每个 token 对应一个向量
- Position Embedding: 编码位置信息

### 3. Multi-Head Attention（多头注意力）
让模型关注序列中不同位置的信息，是 Transformer 的核心。

### 4. Feed-Forward Network（前馈网络）
两层 MLP，增加模型的表达能力。

### 5. 自回归生成
逐个预测下一个 token，直到生成完整文本。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 128 | 嵌入维度 |
| n_heads | 4 | 注意力头数 |
| n_layers | 4 | Transformer 层数 |
| d_ff | 512 | FFN 隐藏层维度 |
| max_seq_len | 128 | 最大序列长度 |

## 扩展建议

1. **更多数据**: 下载更大的文本数据集
2. **BPE 分词**: 实现 Byte-Pair Encoding
3. **学习率调度**: 添加 warmup 和 cosine decay
4. **更大模型**: 增加层数和维度（需要 GPU）
