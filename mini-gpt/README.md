# Mini GPT 中文问答

从零实现的中文问答 AI。

## 快速开始 (Colab)

```python
# 克隆项目
!git clone https://github.com/fgzhanhao/myllm.git
%cd myllm/mini-gpt

# 1. 准备数据
!python download_data.py chinese
!python download_data.py qa

# 2. 预训练（学中文）
!python train.py

# 3. 问答微调
!python train_qa.py

# 4. 测试问答
!python chat.py
```

## 文件说明

- `model.py` - GPT 模型
- `tokenizer.py` - 字符分词器
- `train.py` - 预训练脚本
- `train_qa.py` - 问答微调脚本
- `chat.py` - 交互问答
- `download_data.py` - 数据下载
