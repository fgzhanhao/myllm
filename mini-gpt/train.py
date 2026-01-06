"""
训练脚本 - 在小数据集上训练 Mini GPT
"""
import torch
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT
from tokenizer import CharTokenizer
import os

# ============ 超参数配置 ============
CONFIG = {
    # 模型参数
    'd_model': 256,      # 嵌入维度（大数据集用更大的模型）
    'n_heads': 8,        # 注意力头数
    'n_layers': 6,       # Transformer 层数
    'd_ff': 1024,        # FFN 隐藏层维度
    'max_seq_len': 256,  # 最大序列长度
    'dropout': 0.1,
    
    # 训练参数
    'batch_size': 64,
    'learning_rate': 3e-4,
    'epochs': 20,        # 大数据集不需要太多 epoch
    'eval_interval': 2,  # 每 N 个 epoch 评估一次
    
    # 数据（可选: input.txt, tiny_shakespeare.txt, shakespeare.txt, chinese_poems.txt）
    'data_path': 'data/tiny_shakespeare.txt',
    'train_split': 0.9,
}

class TextDataset(Dataset):
    """文本数据集：滑动窗口切分"""
    
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def train():
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"数据长度: {len(text)} 字符")
    
    # 构建分词器
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    tokenizer.save('tokenizer.json')
    
    # 编码数据
    data = tokenizer.encode(text)
    
    # 划分训练/验证集
    split_idx = int(len(data) * CONFIG['train_split'])
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"训练集: {len(train_data)} tokens")
    print(f"验证集: {len(val_data)} tokens")
    
    # 创建数据集
    train_dataset = TextDataset(train_data, CONFIG['max_seq_len'])
    val_dataset = TextDataset(val_data, CONFIG['max_seq_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # 创建模型
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_seq_len=CONFIG['max_seq_len'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            logits, loss = model(x, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 定期评估
        if (epoch + 1) % CONFIG['eval_interval'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
                  f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # 生成示例
            model.eval()
            start_text = "To be"
            start_ids = torch.tensor([tokenizer.encode(start_text)], device=device)
            generated = model.generate(start_ids, max_new_tokens=50, temperature=0.8)
            print(f"生成示例: {tokenizer.decode(generated[0].tolist())}\n")
    
    # 保存模型
    torch.save(model.state_dict(), 'model.pt')
    print("模型已保存到 model.pt")


if __name__ == '__main__':
    train()
