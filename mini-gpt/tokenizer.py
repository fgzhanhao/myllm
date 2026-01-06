"""简单字符级分词器"""

class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, text):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"词表大小: {self.vocab_size}")
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, ids):
        return ''.join([self.idx_to_char.get(i, '') for i in ids])
    
    def save(self, path):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'c2i': self.char_to_idx, 'i2c': {str(k): v for k, v in self.idx_to_char.items()}}, f, ensure_ascii=False)
    
    def load(self, path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char_to_idx = data['c2i']
            self.idx_to_char = {int(k): v for k, v in data['i2c'].items()}
            self.vocab_size = len(self.char_to_idx)
