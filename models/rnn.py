import torch
import torch.nn as nn
import numpy as np

class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLayer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Initialize weights and biases 
        # Initialize weights with Xavier initialization
        # For weight matrix of shape (n_in, n_out):
        # Normal distribution with mean 0 and variance 2 / (n_in + n_out)
        # Use torch.randn
        # Initialize biases with zeros (torch.zeros)
        # Use nn.Parameter to create the weights and biases and pass in the initialziation
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5) #xavier 
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * (2 / (input_size + output_size)) ** 0.5)
        self.b_y = nn.Parameter(torch.zeros(output_size))
        
        # Initialize activation functions
        # Consider that the output of the rnn layer is (B, T, output_size)
        # Specify which dimension softmax should be taken over
        # Use nn.Tanh and nn.Softmax
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=-1)

    #Implement forward pass
    def forward(self, x):
        
        # Define dimensions
        B = x.shape[0]
        T = x.shape[1]
        H = self.hidden_size
        
        # Initialize hidden state to zeros (torch.zeros) of the right dimensions
        # This will be updated for each time step so no need to have a T dimension
        hidden = torch.zeros(B, H)
        
        # List of outputs for each time step
        outputs = []
        
        # Calculate hidden and output for each time step and update variables
        # Follow the equations in the problem set
        # Remember to softmax the output before appending to outputs
        # torch.matmul is used for matrix multiplication
        # ----------------------------------------
        for t in range(T):
            x_t = x[:, t, :]
            Wx_xt = torch.matmul(x_t, self.W_xh)
            Wh_ht = torch.matmul(hidden, self.W_hh)
            hidden = self.tanh(Wh_ht + Wx_xt + self.b_h)
            
            output = torch.matmul(hidden, self.W_hy) + self.b_y
            # output = self.softmax(output)
            outputs.append(output)
        # ----------------------------------------
        
        # Stack outputs along the time dimension to get a tensor of shape (B, T, output_size)
        # Hint: use torch.stack
        outputs = torch.stack(outputs, dim=1)
        
        return outputs

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs, dim=-1)
        return predictions

class RNN(nn.Module):
    def __init__(self, tokenizer, embedding_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #Initialize layers
        # Use nn.Embedding for the embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You can get the vocabulary size from the tokenizer with tokenizer.vocab_size
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_size)
        self.rnn_layer = RNNLayer(embedding_size, hidden_size, output_size)
    
    # Implement forward pass
    def forward(self, x):
        # Get text embedding
        x = self.embedding(x)

        # Pass through rnn layer
        x = self.rnn_layer(x)

        return x