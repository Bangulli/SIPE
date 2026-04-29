import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import json
from operator import itemgetter
from src.utils.misc import make_name_from_list

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        #print(img.min(), img.max(), img.dtype, img.shape)
        if len(img.shape)==4: img = img.squeeze(0)
        denorm = (img * self.std.to(img.device)) + self.mean.to(img.device)
        denorm = (denorm.clamp(0, 1)*255).to(torch.uint8)
        #print(denorm.min(), denorm.max(), denorm.dtype, denorm.shape)
        return denorm
    
    
class UnNormalizeFloats(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        
        denorm = (img * self.std.to(img.device)) + self.mean.to(img.device)
        denorm = denorm.clamp(0, 1)
        #print(denorm.min(), denorm.max(), denorm.dtype, denorm.shape)
        return denorm

class SobelTransform: 
    def __init__(self, standardize=False):
        self.standardize = standardize
        self.denormer = UnNormalizeFloats([
            0.707223,
            0.578729,
            0.703617
        ], [
            0.211883,
            0.230117,
            0.177517
        ])
        self.sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)
            
    def _binarize(self, img): ## to be changed to a canny filter tuned based on compound used
        return self.denormer(img).mean(dim=1).unsqueeze(1)
    
    def _sobel(self, img):
        gx = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), self.sobel_x.to(img.device))
        gy = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), self.sobel_y.to(img.device))
        magnitude = torch.sqrt(gx ** 2 + gy ** 2)
        magnitude = magnitude / magnitude.max().clamp(min=1e-8)
        magnitude = torch.abs(magnitude-1)
        return magnitude
    
    def _apply_std(self, msk):
        mean = 0.9214555621147156
        std = 0.1259652078151703
        return (msk-mean)/std
    
    def sobel_unnorm(self, msk):
        mean = 0.9214555621147156
        std = 0.1259652078151703
        return (msk*std)+mean
    
    def __call__(self, batch):
        sobel = self._sobel(self._binarize(batch['image']))
        if self.standardize: sobel = self._apply_std(sobel)
        return sobel.squeeze(0)