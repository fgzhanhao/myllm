"""问答微调脚本"""
import torch
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT
from tokenizer import CharTokenizer
import time

CONFIG = {
    'd_model': 128, 'n_heads': 4, 'n_layers': 4, 'max_seq_len': 64,
    'batch_size': 16, 'lr': 1e-4, 'epochs': 100,
    'data_path': 'data/qa.txt',
    'pretrain_model': 'model.pt',  # 预训练模型路径
}

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return max(1, len(self.data) - self.seq_len)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"问答数据: {len(text)} 字符")
    
    tokenizer = CharTokenizer()
    tokenizer.load('tokenizer.json')  # 加载预训练的词表
    
    # 检查是否有新字符，如果有则重新构建词表
    new_chars = set(text) - set(tokenizer.char_to_idx.keys())
    if new_chars:
        print(f"发现新字符，重新构建词表...")
        tokenizer.fit(text)
        tokenizer.save('tokenizer.json')
    
    data = tokenizer.encode(text)
    train_loader = DataLoader(TextDataset(data, CONFIG['max_seq_len']), 
                              batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = MiniGPT(tokenizer.vocab_size, CONFIG['d_model'], CONFIG['n_heads'], 
                    CONFIG['n_layers'], CONFIG['max_seq_len']).to(device)
    
    # 尝试加载预训练模型
    try:
        model.load_state_dict(torch.load(CONFIG['pretrain_model'], map_location=device, weights_only=True))
        print("已加载预训练模型")
    except:
        print("未找到预训练模型，从头训练")
    
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    print("\n开始问答微调...")
    start = time.time()
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | 用时: {time.time()-start:.1f}s")
            
            # 测试问答
            model.eval()
            prompt = "<问>中国的首都是哪里？<答>"
            ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            out = model.generate(ids, max_tokens=20)
            print(f"测试: {tokenizer.decode(out[0].tolist())}\n")
    
    torch.save(model.state_dict(), 'model_qa.pt')
    print(f"微调完成！总用时: {time.time()-start:.1f}s")

if __name__ == '__main__':
    train()
