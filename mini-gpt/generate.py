"""
文本生成脚本 - 使用训练好的模型生成文本
"""
import torch
from model import MiniGPT
from tokenizer import CharTokenizer

def generate_text(prompt="To be", max_tokens=200, temperature=0.8, top_k=40):
    """生成文本"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载分词器
    tokenizer = CharTokenizer()
    tokenizer.load('tokenizer.json')
    
    # 加载模型（使用与训练相同的配置）
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=128,
        dropout=0.0  # 推理时关闭 dropout
    ).to(device)
    
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.eval()
    
    # 编码输入
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # 生成
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)
    
    return generated_text


if __name__ == '__main__':
    import sys
    
    prompt = sys.argv[1] if len(sys.argv) > 1 else "To be"
    generate_text(prompt=prompt)
