import torch


def sipe_collate(batch):
    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }
