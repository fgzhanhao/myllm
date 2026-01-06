"""问答交互脚本"""
import torch
from model import MiniGPT
from tokenizer import CharTokenizer

def chat():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = CharTokenizer()
    tokenizer.load('tokenizer.json')
    
    model = MiniGPT(tokenizer.vocab_size, 128, 4, 4, 64).to(device)
    model.load_state_dict(torch.load('model_qa.pt', map_location=device, weights_only=True))
    model.eval()
    
    print("中文问答AI (输入 q 退出)")
    print("-" * 40)
    
    while True:
        question = input("你: ").strip()
        if question.lower() == 'q':
            break
        if not question:
            continue
        
        prompt = f"<问>{question}<答>"
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        
        with torch.no_grad():
            out = model.generate(ids, max_tokens=50, temperature=0.7)
        
        response = tokenizer.decode(out[0].tolist())
        # 提取答案部分
        if '<答>' in response:
            answer = response.split('<答>')[-1].split('<问>')[0].strip()
        else:
            answer = response[len(prompt):]
        
        print(f"AI: {answer}\n")

if __name__ == '__main__':
    chat()
