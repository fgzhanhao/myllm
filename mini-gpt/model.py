"""
Mini GPT 模型 - 最小化的 Transformer 实现
用于学习理解 LLM 核心原理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头自注意力机制 - Transformer 的核心"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V 投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 拆分成多头: (batch, seq, d_model) -> (batch, n_heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数: (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码（防止看到未来的 token）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        out = torch.matmul(attn_weights, V)
        
        # 合并多头: (batch, n_heads, seq, head_dim) -> (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """前馈神经网络 - 两层 MLP"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT 使用 GELU 激活函数
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """单个 Transformer 块 = 注意力 + FFN + 残差连接 + LayerNorm"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Pre-LayerNorm 架构（GPT-2 风格）
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Mini GPT 模型"""
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, 
                 d_ff=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token Embedding: 将 token ID 转为向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position Embedding: 让模型知道 token 的位置
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer 层
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        
        # 输出层: 预测下一个 token 的概率分布
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享: embedding 和 output 共享权重（常见技巧）
        self.token_embedding.weight = self.lm_head.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        # 创建因果掩码（下三角矩阵）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        
        # Token + Position Embedding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # 通过所有 Transformer 块
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # 计算损失（如果提供了目标）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成文本"""
        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            idx_cond = idx[:, -self.max_seq_len:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # 采样下一个 token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
