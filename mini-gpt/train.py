"""训练脚本"""
import torch
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT
from tokenizer import CharTokenizer
import time

CONFIG = {
    'd_model': 128, 'n_heads': 4, 'n_layers': 4, 'max_seq_len': 64,
    'batch_size': 128, 'lr': 3e-4, 'epochs': 30,
    'data_path': 'data/chinese.txt',
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
    
    # 同时加载问答数据，确保词表包含所有字符
    try:
        with open('data/qa.txt', 'r', encoding='utf-8') as f:
            qa_text = f.read()
        all_text = text + qa_text
    except:
        all_text = text
    
    print(f"数据: {len(text)} 字符")
    
    tokenizer = CharTokenizer()
    tokenizer.fit(all_text)  # 用所有文本构建词表
    tokenizer.save('tokenizer.json')
    data = tokenizer.encode(text)
    
    split = int(len(data) * 0.9)
    train_loader = DataLoader(TextDataset(data[:split], CONFIG['max_seq_len']), 
                              batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = MiniGPT(tokenizer.vocab_size, CONFIG['d_model'], CONFIG['n_heads'], 
                    CONFIG['n_layers'], CONFIG['max_seq_len']).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    print("\n开始训练...")
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
        
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | 用时: {time.time()-start:.1f}s")
            
            # 生成示例
            model.eval()
            prompt = "什么是"
            ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            out = model.generate(ids, max_tokens=30)
            print(f"生成: {tokenizer.decode(out[0].tolist())}\n")
    
    torch.save(model.state_dict(), 'model.pt')
    print(f"训练完成！总用时: {time.time()-start:.1f}s")

if __name__ == '__main__':
    train()
