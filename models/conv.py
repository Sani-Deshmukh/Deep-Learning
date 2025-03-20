import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1, filters=None, biases=None):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.filters = nn.Parameter(filters if filters is not None else torch.randn(out_channels, in_channels, filter_size, filter_size))
        self.biases = nn.Parameter(biases if biases is not None else torch.randn(out_channels))


    def convolution(self, x: torch.Tensor, filters: torch.Tensor):
        """
        2D convolution of filters over x
        x: [batch_size, channels, height, width]
        filters: [out_channels, in_channels, filter_size, filter_size]
        """
        B = x.shape[0]
        C_in = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        C_out = self.out_channels
        F = self.filter_size
        P = self.padding
        S = self.stride

        H_out = (((H + 2 * P) - F) // S) + 1
        W_out = (((W + 2 * P) - F) // S) + 1

        output = torch.zeros([B, C_out, H_out, W_out])

        padded_x = torch.zeros([B, C_in, H + 2 * P, W + 2 * P])
        padded_x[:, :, P:P + H, P:P + W] = x

        for b in range(B):
            for c in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        # apply filter 
                        h_start = h * S 
                        h_end = h_start + F
                        w_start =  w * S
                        w_end = w_start + F
                        
                        region = padded_x[b, :, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = torch.sum(region * self.filters[c, :, :, :])

        return output
        

    def forward(self, x: torch.Tensor):
        x = self.convolution(x, self.filters)
        return x + self.biases.view(1, -1, 1, 1)

class FasterConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1, filters=None, biases=None):
        super(FasterConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.filters = nn.Parameter(filters if filters is not None else torch.randn(out_channels, in_channels, filter_size, filter_size))
        self.biases = nn.Parameter(biases if biases is not None else torch.randn(out_channels))


    def convolution(self, x: torch.Tensor, filters: torch.Tensor):
        """
        2D convolution of filters over x
        x: [batch_size, channels, height, width]
        filters: [out_channels, in_channels, filter_size, filter_size]
        """
        B = x.shape[0]
        C_in = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        C_out = self.out_channels
        F = self.filter_size
        P = self.padding
        S = self.stride

        H_out = (((H + 2 * P) - F) // S) + 1
        W_out = (((W + 2 * P) - F) // S) + 1

        output = torch.zeros([B, C_out, H_out, W_out])

        padded_x = torch.zeros([B, C_in, H + 2 * P, W + 2 * P])
        padded_x[:, :, P:P + H, P:P + W] = x
        
        for h in range(H_out):
            for w in range(W_out):
                # apply filter 
                h_start = h * S 
                h_end = h_start + F
                w_start =  w * S
                w_end = w_start + F
                
                region = padded_x[:, :, h_start:h_end, w_start:w_end]  # Correct batch indexing
                output[:, :, h, w] = torch.sum(region.unsqueeze(1) * self.filters.unsqueeze(0), dim=(2, 3, 4))


        return output

    def forward(self, x: torch.Tensor):
        x = self.convolution(x, self.filters)
        return x + self.biases.view(1, -1, 1, 1)
    

