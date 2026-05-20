import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import json
from operator import itemgetter
from sipe.utils.misc import make_name_from_list

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
    def __init__(self, normalize=False):
        self.normalize = normalize
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
        denormed = self.denormer(img)
        return denormed.mean(dim=1).unsqueeze(1)
    
    def _sobel(self, img):
        gx = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), self.sobel_x)
        gy = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), self.sobel_y)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2)
        magnitude = magnitude / magnitude.max().clamp(min=1e-8)
        magnitude = torch.abs(magnitude-1)
        return magnitude
    
    def _normalize(self, sobel, label):
        for i in range(len(label)):
            mean, std = itemgetter('mean', 'std')(self.cfg[label[i]])
            sobel[i] = (sobel[i]-float(mean))/float(std)
        return sobel
    
    def __call__(self, batch):
        sobel = self._sobel(self._binarize(batch['image']))
        #if self.normalize: sobel = self._normalize(sobel, [make_name_from_list(s['staining']) for s in batch['metadata']])
        return sobel.squeeze(0)