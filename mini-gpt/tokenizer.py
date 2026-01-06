"""
Character-level Tokenizer
最简单的分词器：每个字符就是一个 token
"""

class CharTokenizer:
    """字符级分词器"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, text):
        """从文本中构建词表"""
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"词表大小: {self.vocab_size}")
        print(f"字符: {''.join(chars)}")
    
    def encode(self, text):
        """文本 -> token IDs"""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, ids):
        """token IDs -> 文本"""
        return ''.join([self.idx_to_char[i] for i in ids])
    
    def save(self, path):
        """保存词表"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
            }, f, ensure_ascii=False)
    
    def load(self, path):
        """加载词表"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
            self.vocab_size = len(self.char_to_idx)
