import torch.nn as nn
import torch

# Convolutional Projector ########################################################
    
class ConvProjector(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layer1 = nn.Conv2d(inputs, outputs, kernel_size=1)
        
    def forward(self, x):
        x = self.layer1(x)
        return x

class ConvDisentangler(nn.Module):
    def __init__(self, inputs, s_size, z_size): # s is specified, z is unspecified
        super().__init__()
        self.s_projector = ConvProjector(inputs, s_size)
        self.z_projector = ConvProjector(inputs, z_size)
        
    def forward(self, x):
        s = self.s_projector(x)
        z = self.z_projector(x)
        return torch.cat([s, z], dim=1)

# Projector ######################################################## 

class Projector(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layer1 = nn.Linear(inputs, outputs)
        
    def forward(self, x):
        x = self.layer1(x)
        return x
    
class Disentangler(nn.Module):
    def __init__(self, inputs, s_size, z_size): # s is specified, z is unspecified
        super().__init__()
        self.s_projector = Projector(inputs, s_size)
        self.z_projector = Projector(inputs, z_size)
        
    def forward(self, x):
        s = self.s_projector(x)
        z = self.z_projector(x)
        return torch.cat([s, z], dim=-1)
    
# Projector Normalizer ######################################################## 
    
class ProjectorN(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layer1 = nn.Linear(inputs, 4*inputs)
        self.norm = nn.ReLU()
        self.layer2 = nn.Linear(4*inputs, inputs)
        self.norm2 = nn.functional.normalize
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.layer2(x)
        return self.norm2(x)
    
class FullProjector(nn.Module):
    def __init__(self, stain_size, morph_size):
        super().__init__()
        self.stain_size = stain_size
        self.stain = ProjectorN(stain_size)
        self.morph = ProjectorN(morph_size)
        
    def forward(self, x):
        stain = self.stain(x[:, :self.stain_size])
        morph = self.morph(x[:, self.stain_size:])
        return stain, morph

# Normalizer ######################################################## 

class Normalizer(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        self.norm = nn.functional.normalize
        
    def forward(self, x):
        return self.norm(x)
    
class FullNormalizer(nn.Module):
    def __init__(self, stain_size, morph_size):
        super().__init__()
        self.stain_size = stain_size
        self.stain = Normalizer(stain_size)
        self.morph = Normalizer(morph_size)
        
    def forward(self, x):
        stain = self.stain(x[:, :self.stain_size])
        morph = self.morph(x[:, self.stain_size:])
        return stain, morph