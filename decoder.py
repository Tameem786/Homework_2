# decoder.py
import torch
import torch.nn as nn
import math
from encoder import Embeddings, AddNorm, FeedForward  # Import Embeddings class from the encoder module.

# Define the MaskedSelfAttention class, which implements masked self-attention for the decoder.
class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        # Initialize the MaskedSelfAttention with embedding dimension and number of heads.
        super(MaskedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # Calculate the dimension of each head.
        self.head_dim = embed_dim // num_heads

        # Define linear layers for transforming input into queries (Q), keys (K), and values (V).
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        # Define a linear layer to transform the concatenated multi-head attention output back to `embed_dim`.
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Get batch size and sequence length from the input tensor.
        batch_size, seq_len, _ = x.size()
        # Compute Q, K, V using linear transformations.
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Reshape Q, K, V for multi-head attention and transpose for easier computation.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores using scaled dot-product of Q and K.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Create a lower triangular mask to prevent attending to future tokens in the sequence.
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        # Apply the mask, setting the scores of future positions to negative infinity.
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # Apply softmax to convert scores into attention weights.
        attention_weights = torch.softmax(scores, dim=-1)

        # Compute the weighted sum of values (V) using attention weights.
        attention_output = torch.matmul(attention_weights, V)
        # Reshape the output back into the original tensor shape and apply the final linear layer.
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_linear(attention_output)

# Define the EncoderDecoderAttention class, which implements attention between the decoder and encoder.
class EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        # Initialize the EncoderDecoderAttention with embedding dimension and number of heads.
        super(EncoderDecoderAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # Calculate the dimension of each head.
        self.head_dim = embed_dim // num_heads

        # Define linear layers for transforming input into queries (Q), and the encoder output into keys (K) and values (V).
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        # Define a linear layer to transform the concatenated multi-head attention output back to `embed_dim`.
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, encoder_output):
        # Get batch size and sequence length from the input tensor.
        batch_size, seq_len, _ = x.size()
        # Get the sequence length of the encoder's output.
        enc_seq_len = encoder_output.size(1)

        # Compute Q from the decoder's input, and K, V from the encoder's output.
        Q = self.q_linear(x)
        K = self.k_linear(encoder_output)
        V = self.v_linear(encoder_output)

        # Reshape Q, K, V for multi-head attention and transpose for easier computation.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, enc_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, enc_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores using scaled dot-product of Q and K.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply softmax to convert scores into attention weights.
        attention_weights = torch.softmax(scores, dim=-1)

        # Compute the weighted sum of values (V) using attention weights.
        attention_output = torch.matmul(attention_weights, V)
        # Reshape the output back into the original tensor shape and apply the final linear layer.
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_linear(attention_output)

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=2048):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MaskedSelfAttention(embed_dim, num_heads)
        self.add_norm1 = AddNorm(embed_dim)
        self.encoder_decoder_attention = EncoderDecoderAttention(embed_dim, num_heads)
        self.add_norm2 = AddNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.add_norm3 = AddNorm(embed_dim)

    def forward(self, x, encoder_output):
        masked_attention_output = self.masked_self_attention(x)
        x = self.add_norm1(x, masked_attention_output)
        enc_dec_attention_output = self.encoder_decoder_attention(x, encoder_output)
        x = self.add_norm2(x, enc_dec_attention_output)
        ffn_output = self.feed_forward(x)
        x = self.add_norm3(x, ffn_output)
        return x

# Define the TransformerDecoder class, which represents the complete Transformer decoder model.
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim=2048, max_len=512):
        # Initialize the TransformerDecoder with vocab size, embedding dimension, number of heads, number of layers, hidden dimension, and max sequence length.
        super(TransformerDecoder, self).__init__()
        # Create an embedding layer with positional encoding.
        self.embedding = Embeddings(vocab_size, embed_dim, max_len)
        # Stack multiple DecoderLayer instances using ModuleList for the specified number of layers.
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output):
        # Pass the input through the embedding layer to get combined embeddings.
        x = self.embedding(x)
        # Sequentially pass the embeddings through each decoder layer, using the encoder's output.
        for layer in self.decoder_layers:
            x = layer(x, encoder_output)
        # Return the final output from the last decoder layer.
        return x
