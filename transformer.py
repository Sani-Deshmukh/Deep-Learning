import torch
import torch.nn as nn
import numpy as np
from psthree.attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, output_size):
        super(TransformerEncoder, self).__init__()
        
       # Initialize
       # Ensure you use the norm stated in class and get it from torch.nn
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Initialize Feed Forward network
        # As described in section 3.3 Attention is all you need paper
        # Note the arguments of this __init__ function 
        # Use nn.Sequential, nn.Linear, and nn.ReLU
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Initialize output layer
        # Use nn.Linear
        self.output_layer = nn.Linear(d_model, output_size)

        # Initialize softmax
        # Consider that the output of the rnn layer is (B, T, output_size)
        # Specify which dimension softmax should be taken over
        # Use nn.Softmax
        # self.softmax = nn.Softmax(dim=-1)
        
    # Implement forward pass
    def forward(self, x):

        # Self-attention with residual connection and normalization
        attn = self.attention(x)
        x = self.norm1(x + attn)
        
        # Feed-forward with residual connection and normalization
        ff = self.ff(x)
        x = self.norm2(x + ff)
        
        # Output layer and softmax
        output = self.output_layer(x)
        # output = self.softmax(output)
        
        return output

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs, dim=-1)
        return predictions

class Transformer(nn.Module):
    def __init__(self, tokenizer, embedding_size, num_heads, d_ff, output_size, max_seq_length=512):
        super(Transformer, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.max_seq_length = max_seq_length
        
        # Initialize layers
        # Use nn.Embedding for the embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You can get the vocabulary size from the tokenizer with tokenizer.vocab_size
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_size)
        
        # Positional encoding
        self.register_buffer('positional_encoding', self._create_positional_encoding())
        
        # Initialize transformer encoder
        # Use embedding_size for d_model
        self.transformer_encoder = TransformerEncoder(embedding_size, num_heads, d_ff, output_size)
    
    def _create_positional_encoding(self):
        position = torch.arange(self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_size, 2) * (-np.log(10000.0) / self.embedding_size))
        pos_encoding = torch.zeros(self.max_seq_length, self.embedding_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    # Implement forward pass
    def forward(self, x):
        # Get sequence length for current batch
        T = x.size(1)
        
        # Token embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :T, :]
        
        # Pass through transformer layer
        x = self.transformer_encoder(x)
        
        return x
