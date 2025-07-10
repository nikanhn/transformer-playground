# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, GELU, ReLU, Embedding
import math
  
#main reference for this part: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

#main reference for this part: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    #casual masking
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    #if attention mask provided, apply it
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    #calculate attention weights
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight, attn_weight @ value


#main reference for this part: Karpathy's GitHub repository: https://github.com/karpathy/nanoGPT
class Attention(torch.nn.Module):

    def __init__(self, d_model: int, n_heads:int, dropout_attn: float, dropout_O: float, is_casual: bool) -> None:
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0

        self.is_casual = is_casual

        self.d_k = d_model // n_heads

        #define the linear layers for query, key, and value
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

        self.dropout_attn = dropout_attn
        self.dropout_o = dropout_O

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embed_dim = x.size()

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        #apply linear transformations and split into multiple heads
        q = q.view(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)

        #scaled dot-product attention
        att_w, y = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_attn if self.training else 0, is_causal=self.is_casual)
       
        #concatenate multiple heads and apply final linear transformation
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, embed_dim)
        y = self.W_o(y)
        y = torch.nn.functional.dropout(y, p=self.dropout_o, training=self.training)

        return att_w, y
    
class MLP(torch.nn.Module):

    def __init__(self, d_model: int, bias: bool, dropout: float, exploration: str = None):
        super().__init__()
        self.exploration = exploration
        self.c_fc    = Linear(d_model, 4 * d_model, bias=bias)
        if self.exploration == 'architectural1':
            self.gelu = nn.SELU()
        else:
            self.gelu    = GELU()
        self.c_proj  = Linear(4 * d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

        
class Transformer(torch.nn.Module):

    def __init__(self, d_model: int, n_heads:int, dropout_attn: float, dropout_O: float, is_casual: bool, bias: bool, exploration: str = None):
        super().__init__()
        #layer normalization before the attention layer
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.attn = Attention(d_model, n_heads, dropout_attn, dropout_O, is_casual)
        #l ayer normalization before the feed-forward layer
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model, bias, dropout_O, exploration=exploration)

    def forward(self, x):
        att_w, att_o = self.attn(self.ln_1(x))
        x = x + att_o
        x = x + self.mlp(self.ln_2(x))
        return att_w, x
    

class Encoder_version1(torch.nn.Module):
        
        def __init__(self, d_model: int, n_heads:int, dropout_attn: float, dropout_O: float, is_casual: bool, bias: bool, n_layers: int, d_hidden:int,  num_classes: int, vocab_size: int, block_size: int, exploration: str = None):
            super().__init__()
            self.exploration = exploration
            #embedding layer for token embedding 
            self.embedding_layer = torch.nn.Embedding(vocab_size, d_model)
            #positional encoding layer
            self.positional_embedding_layer = torch.nn.Embedding(block_size, d_model)
            self.layers = torch.nn.ModuleList([Transformer(d_model, n_heads, dropout_attn, dropout_O, is_casual, bias, exploration=exploration) for _ in range(n_layers)])
            #classifier with ReLU activation
            self.classifier = torch.nn.Sequential(Linear(d_model, d_hidden, bias=bias), nn.ReLU(), Dropout(dropout_O), 
                                                  Linear(d_hidden, num_classes, bias=bias,))
    
        def forward(self, x):
            x_e = self.embedding_layer(x)
            if self.exploration == 'architectural2':
                x_p = self.cosine_embedding_layer(x)
            else:
                x_p = self.positional_embedding_layer(torch.arange(0, x.size()[1], dtype=torch.long, device=x.device))
            #summing the token and positional embeddings
            x = x_e + x_p
            att_w_list = []
            for layer in self.layers:
                att_w, x = layer(x)
                att_w_list.append(att_w.detach())

            #average the output of the transformer layers
            x = torch.mean(x, dim=1)
            x = self.classifier(x)
            x = torch.nn.functional.softmax(x, dim=-1)

            return att_w_list, att_w_list, att_w_list, x
        
class Decoder_version1(torch.nn.Module):
        
        def __init__(self, d_model: int, 
                     n_heads:int, 
                     dropout_attn: float, 
                     dropout_O: float, 
                     is_casual: bool, 
                     bias: bool, 
                     n_layers: int, 
                     d_hidden:int,  
                     num_classes: int, 
                     vocab_size: int, 
                     block_size: int,
                     exploration: str = None):
            
            super().__init__()

            self.exploration = exploration
            self.d_hidden = d_hidden
            self.d_model = d_model
            self.block_size = block_size
            self.cosine_embedding_layer = PositionalEncoding(self.d_model, dropout=0, max_len=self.block_size)
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(vocab_size, d_model),
                # wpe = nn.Embedding(block_size, d_model),
                wpe = PositionalEncoding(d_model, dropout=dropout_attn, max_len=block_size),
                drop = nn.Dropout(dropout_attn),
                h = nn.ModuleList([Transformer(d_model, n_heads, dropout_attn, dropout_O, is_casual, bias, exploration=exploration) for _ in range(n_layers)]),
                # ln_f = LayerNorm(d_model, bias=bias),
            ))
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight


        def forward(self, idx):
            device = idx.device
            _, t = idx.size()
            # pos = torch.arange(0, t, dtype=torch.long, device=device)

            tok_emb = self.transformer.wte(idx)
            if self.exploration == 'architectural2':
                x = self.transformer.drop(self.cosine_embedding_layer(tok_emb))
            else:
                # pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(tok_emb + self.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=device)))
            att_w_list = []
            for block in self.transformer.h:
                att_w, x = block(x)
                att_w_list.append(att_w.detach())
            # x = self.transformer.ln_f(x)
            x = self.lm_head(x)

            return att_w_list, att_w_list, att_w_list, x
        
