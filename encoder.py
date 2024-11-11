# encoder.py
import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=512):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x += self.positional_encoding[:, :seq_len, :]
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Reshape Q, K, V for multi-head attention and transpose for easier computation.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_linear(attention_output)

class AddNorm(nn.Module):
    def __init__(self, embed_dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=2048):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.add_norm1 = AddNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.add_norm2 = AddNorm(embed_dim)

    def forward(self, x):
        attention_output = self.self_attention(x)
        x = self.add_norm1(x, attention_output)
        ffn_output = self.feed_forward(x)
        x = self.add_norm2(x, ffn_output)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim=2048, max_len=512):
        super(TransformerEncoder, self).__init__()
        self.embedding = Embeddings(vocab_size, embed_dim, max_len)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x
