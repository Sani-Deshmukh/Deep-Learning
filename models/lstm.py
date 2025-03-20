import torch
import torch.nn as nn
import numpy as np

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMLayer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Initialize weights and biases 
        # Initialize weights with Xavier initialization
        # For weight matrix of shape (n_in, n_out):
        # Normal distribution with mean 0 and variance 2 / (n_in + n_out)
        # Use torch.randn
        # Initialize biases with zeros (torch.zeros)
        # Use nn.Parameter to create the weights and biases and pass in the initialziation

        # Input gate weights
        self.W_xi =  nn.Parameter(torch.randn(input_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5)
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5)
        self.b_i =  nn.Parameter(torch.zeros(hidden_size))
        
        # Forget gate weights
        self.W_xf = nn.Parameter(torch.randn(input_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5)
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        
        # Output gate weights
        self.W_xo = nn.Parameter(torch.randn(input_size, hidden_size) * (2/(input_size + hidden_size)) ** 0.5)         
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
        # Cell state weights
        self.W_xc = nn.Parameter(torch.randn(input_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5)
        self.W_hc = nn.Parameter(torch.randn(hidden_size, hidden_size) * (2 / (input_size + hidden_size)) ** 0.5)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        
        # Output weights
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * (2 / (hidden_size + output_size)) ** 0.5)
        self.b_y = nn.Parameter(torch.zeros(output_size))
        
        # Initialize activation functions
        # Consider that the output of the rnn layer is (B, T, output_size)
        # Specify which dimension softmax should be taken over
        # Use nn.Tanh, nn.Sigmoid, and nn.Softmax
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=-1)  

    # Implement forward pass
    def forward(self, x):
        
        # Define dimensions
        B = x.shape[0]
        T = x.shape[1]
        H = self.hidden_size
        
        
        # Initialize h_t and c_t to zeros (torch.zeros) of the right dimensions
        # This will be updated for each time step so no need to have a T dimension
        h_t = torch.zeros(B, H)
        c_t = torch.zeros(B, H)
        
        # List of outputs for each time step
        outputs = []
        
        # Calculate h_t, c_t, and output for each time step and update variables
        # Follow the equations in the problem set
        # Remember to softmax the output before appending to outputs
        # Element-wise multiplcation can be done with just "*"
        # torch.matmul is used for matrix multiplication
        # ----------------------------------------
        for t in range(T):
            x_t =  x[:, t, :]
            
            i_t = self.sigmoid(torch.matmul(x_t, self.W_xi) + torch.matmul(h_t, self.W_hi) + self.b_i)
            f_t = self.sigmoid(torch.matmul(x_t, self.W_xf) + torch.matmul(h_t, self.W_hf) + self.b_f)
            o_t = self.sigmoid(torch.matmul(x_t, self.W_xo) + torch.matmul(h_t, self.W_ho) + self.b_o)
            c_t_candidate = self.tanh(torch.matmul(x_t, self.W_xc) + torch.matmul(h_t, self.W_hc) + self.b_c)
            
            c_t = f_t * c_t + i_t * c_t_candidate
            h_t = o_t * self.tanh(c_t)
            
            y_t = torch.matmul(h_t, self.W_hy) + self.b_y
            outputs.append(y_t)
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

class LSTM(nn.Module):
    def __init__(self, tokenizer, embedding_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize layers
        # Use nn.Embedding for the embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You can get the vocabulary size from the tokenizer with tokenizer.vocab_size
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_size)
        self.lstm_layer = LSTMLayer(embedding_size, hidden_size, output_size)
    
    # Implement forward pass
    def forward(self, x):
        # Get text embedding
        x = self.embedding(x)

        # Pass through lstm layer
        x = self.lstm_layer(x)

        return x