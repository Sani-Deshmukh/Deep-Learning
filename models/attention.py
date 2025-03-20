import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()

        # d_model = d_k = d_v
        self.d_model = d_model
        
        #Initialize weights and biases
        # Instead of using nn.Parameter, use nn.Linear (for speed)
        # Don't need any special initializaton
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Initialize softmax
        # In order to compute attention scores, what dimension is softmax applied to?
        # Hint: You want each row of your attention scores to be a probability distribution
        # Use nn.Softmax
        self.softmax = nn.Softmax(dim=-1) 
        
    #Implement forward pass
    # Follow the formula you wrote in the problem set
    # Note that self-attention is applied to every input embedding in x individually
    # Thus, matmuls are broadcasted across the batch and only the last two dimensions matter
    # To transpose K use torch.Tensor.transpose on the two dimensions you want to swap
    # Keep in mind that linear layers are only applied on the last dimension of x by default
    # Fun fact: Implementing this method is a real interview question for Machine Learning Engineering roles
    def forward(self, x):
        Q = self.W_q(x)  
        K = self.W_k(x)  
        V = self.W_v(x) 
        
        output = (torch.matmul(Q, K.transpose(-2, -1))) / ((self.d_model) ** 0.5)
        output = self.softmax(output) 
        output = torch.matmul(output, V)        
        
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"


        self.d_model = d_model
        self.num_heads = num_heads

        # Initialize weights and biases
        # Multi-head attention can be implemented by first applying the same linear layers as in self attention
        # Then, splitting them into different heads and computing attention scores for each head individually
        # Finally, concatenating the heads together and applying the output linear layer
        # Instead of using nn.Parameter, use nn.Linear (for speed)
        # Don't need any special initializaton
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Define d_k
        # d_k is only used when applying attention for each head individually
        self.d_k = d_model // num_heads
        
        # Initialize softmax
        # In order to compute attention scores, what dimension is softmax applied to?
        # Hint: You want each row of your attention scores to be a probability distribution
        # Use nn.Softmax
        self.softmax = nn.Softmax(dim=-1)
    
    # Implement forward pass
    # Follow these steps
    # 1) Apply the linear transformations of q, k, v
    # 2) Use torch.Tensor.reshape to reshape each one to split each vector into num_heads parts: (B, T, num_heads, d_K)
    #    This will effectively split each embedding vector into num_heads parts
    # 3) Use torch.Tensor.transpose to achieve the shape (B, num_heads, T, d_k)
    # 4) Compute the attention scores for each head
    #    Notice that in SelfAttention, the batch dimension was ignored
    #    With the same code, both the batch and num_heads dimensions are ignored
    #    This way you are effectively computing attention for each head individually
    # 5) Use torch.Tensor.transpose to achieve the shape (B, T, num_heads, d_k)
    # 6) Use torch.Tensor.reshape to effectively concatenate the heads together
    # 7) Apply the final linear transformation
    def forward(self, x):
        B, T, _ = x.shape
        
        Q = self.W_q(x)  #(B, T, d_model)
        K = self.W_k(x)  
        V = self.W_v(x) 
        
        # reshaping 
        Q = Q.reshape(B, T, self.num_heads, self.d_k)
        K = K.reshape(B, T, self.num_heads, self.d_k)
        V = V.reshape(B, T, self.num_heads, self.d_k)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, num_heads, T, T)
        attention_weights = self.softmax(attention_scores)  # (B, num_heads, T, T)
        attention_output = torch.matmul(attention_weights, V)  # (B, num_heads, T, d_k)
        attention_output = attention_output.transpose(1, 2)  # (B, T, num_heads, d_k)
        attention_output = attention_output.reshape(B, T, self.d_model)
        output = self.W_o(attention_output)  # (B, T, d_model)
        
        return output    

        