import torch

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
        
        denorm = (img * self.std.to(img.device)) + self.mean.to(img.device)
        denorm = (denorm.clamp(0, 1)*255).to(torch.uint8)
        #print(denorm.min(), denorm.max(), denorm.dtype, denorm.shape)
        return denorm