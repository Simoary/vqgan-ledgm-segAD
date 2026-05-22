import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    
    def __init__(self, num_codebook_vectors: int, d_model: int) -> None:
        super().__init__()
        
        # + 1 for Beginning of Sequence (BOS) token
        self.embedding = nn.Embedding(num_codebook_vectors + 1, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
        

class PositionalEncoding(nn.Module):
    
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(1)

        positional_encoding = torch.zeros(1, max_seq_len, d_model)

        _2i = torch.arange(0, d_model, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        positional_encoding[0, :, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        positional_encoding[0, :, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))
        
        self.register_buffer('positional_encoding', positional_encoding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        
        return x + self.positional_encoding[:, :seq_len, :]
    
class MaskedMultiHeadAttention(nn.Module):
    
    def __init__(self, max_seq_len: int, n_heads: int, d_model: int, p: float = 0.1) -> None:
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(p)

        self.out_proj = nn.Linear(d_model, d_model)
        
        mask = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool))
        self.register_buffer('mask', mask)
        
    def forward(self, x: torch.Tensor, past_cache=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        batch_size, seq_len, _ = x.shape
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        kv_cache = None
        if past_cache is not None:
            # past_kv is a tuple (past_k, past_v)
            # Each has shape [B, nh, L_past, hd]
            past_k, past_v = past_cache
            # Concatenate along the sequence length dimension (dim=2)
            K = torch.cat((past_k, K), dim=2)
            V = torch.cat((past_v, V), dim=2)
            # Store the updated K, V for the next step
            kv_cache = (K, V) # Shape [B, nh, L_past + L_k, hd]
        else:
            # Store K, V for the first time
            kv_cache = (K, V) # Shape [B, nh, L_k, hd]
        
        attn = Q @ K.transpose(2, 3)  
        
        # Apply causal mask
        attn.masked_fill_(~self.mask[:Q.shape[2], :Q.shape[2]], float("-inf")) # seq_len
        
        attn = attn / (K.shape[-1] ** 0.5) # d_k is dimensionality of keys
        
        attn = torch.softmax(attn, dim=-1)
        
        # Some Regularization
        attn = self.dropout(attn)
        
        out = (attn @ V).transpose(1, 2)
        
        out = out.contiguous().view(batch_size, seq_len, self.d_model)
        
        out = self.out_proj(out)
        
        return out, kv_cache
        
class FeedForward(nn.Module):
    
    def __init__(self, d_model: int, p: float = 0.1) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(p)
        )
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, max_seq_len: int, n_heads: int, d_model: int, p: float = 0.1) -> None:
        super().__init__()
        
        self.self_attention = MaskedMultiHeadAttention(max_seq_len, n_heads, d_model, p)
        
        self.norm1 = nn.LayerNorm(d_model)
        
        self.dense_net = FeedForward(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(p)
        
    def forward(self, x: torch.Tensor, past_cache=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        # Self-Attention Portion
        attn, cache = self.self_attention(x, past_cache=past_cache)
        x = x + self.dropout(attn)
        x = self.norm1(x)
        
        # MLP portion
        dense_out = self.dense_net(x)
        x = x + self.dropout(dense_out)
        x = self.norm2(x)
        
        return x, cache 
    
class AutoregressiveTransformer(nn.Module):
    def __init__(self, num_blocks: int, num_codebook_vectors: int, max_seq_len: int, n_heads: int, d_model: int, p: float = 0.1, device="cuda:0") -> None:
        super().__init__()
        
        self.token = TokenEmbedding(num_codebook_vectors, d_model).to(device)
        
        self.pos_embed = PositionalEncoding(max_seq_len, d_model).to(device)
        
        self.blocks = [TransformerBlock(max_seq_len, n_heads, d_model, p).to(device) for _ in range(num_blocks)]
        
        self.linear = nn.Linear(d_model, num_codebook_vectors, bias=False).to(device)
        
    def forward(self, x: torch.Tensor, past_cache=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        x = self.token(x)
        x = self.pos_embed(x)
        
        for block in self.blocks:
            x, cache = block(x, past_cache=past_cache)
            
        x = self.linear(x)

        return x, cache