from torchvision import transforms

class ContrastiveTransformations:
    def __init__(self, n_views=2):
        self.cont_transforms = get_contrastive_transforms()
        self.n_views = n_views

    def __call__(self, x):
        return [self.cont_transforms(x) for i in range(self.n_views)]
    
def get_contrastive_transforms():
    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return contrast_transforms