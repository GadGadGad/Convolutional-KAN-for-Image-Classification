import torch
import torch.nn as nn
import torch.nn.functional as F

# Fuzzy Pooling Implementation
class FuzzyPooling(nn.Module):
    def __init__(self, kernel_size, stride, v_max=6):
        super(FuzzyPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.v_max = v_max
        self.d = self.v_max / 2
        self.c = self.v_max / 4
        self.a = 1.5
        self.m = self.v_max / 2
        self.r = self.v_max / 2
        self.q = self.r + (self.v_max / 4)

    def triangular_membership(self, x, center, width):
        """Triangular membership function."""
        return torch.clamp(1 - torch.abs(x - center) / width, min=0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Pooled tensor of shape (B, C, H', W')
        """

        B, C, H, W = x.shape
        
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        
        output = torch.zeros((B, C, H_out, W_out), device=x.device)
        
        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = x[b, c, i*self.stride:i*self.stride + self.kernel_size,
                                 j*self.stride:j*self.stride + self.kernel_size]
                                
                        patch_flattened = patch.flatten()

                        mu1 = self.triangular_membership(patch_flattened, self.c, self.d - self.c)
                        mu2 = self.triangular_membership(patch_flattened, self.m, self.a)
                        mu3 = self.triangular_membership(patch_flattened, self.r, self.q - self.r)

                        pi1 = mu1
                        pi2 = mu2
                        pi3 = mu3

                        s1 = torch.sum(pi1)
                        s2 = torch.sum(pi2)
                        s3 = torch.sum(pi3)

                        s_values = torch.stack([s1, s2, s3])
                        v_selected = torch.argmax(s_values)

                        if v_selected == 0:
                            selected_inputs = patch_flattened * mu1
                        elif v_selected == 1:
                            selected_inputs = patch_flattened * mu2
                        else:
                            selected_inputs = patch_flattened * mu3

                        sum_num = torch.sum(selected_inputs * patch_flattened)
                        sum_den = torch.sum(selected_inputs)

                        if sum_den == 0:
                            pooled_value = torch.tensor(0.0, device=x.device) 
                        else:
                            pooled_value = sum_num / sum_den


                        output[b, c, i, j] = pooled_value
                                
        return output


if __name__ == '__main__':
    batch_size = 2
    in_channels = 3
    height = 32
    width = 32
    kernel_size = 3
    stride = 2
    
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    fuzzy_pool = FuzzyPooling(kernel_size=kernel_size, stride=stride)
    output_tensor = fuzzy_pool(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)