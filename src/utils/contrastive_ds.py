from BPTorch.datasets import BigPictureRepository

class ContrastiveDataset(BigPictureRepository):
    def __init__(self, *args, contrastive_transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_transforms = contrastive_transforms
        
    def __getitem__(self, idx):
        dct = super().__getitem__(idx)
        if self.contrastive_transforms is not None: dct['contrastive'] = self.contrastive_transforms(dct['image'])
        return dct
    
    
import torch
def contrastive_collate(batch):
    images = []
    coordinates = []
    meta = []
    contrastive = []
    for item in batch:
       images.append(item['image'])
       coordinates.append(item['coordinates'])
       meta.append(item['metadata']) 
       contrastive.append(item['contrastive'])
    res = {
        'image': torch.stack(images),
        'coordinates': torch.stack(coordinates),
        'metadata': meta,
        'contrastive': torch.stack([torch.stack(k) for k in contrastive])
    }
    return res