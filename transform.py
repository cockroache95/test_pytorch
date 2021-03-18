import torch
import numpy as np
import cv2

dev = torch.device("cuda:0")

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sam):
        for t in self.transforms:
            sam = t(sam)
        return sam
class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, self.size)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': image,
                'label': label}    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.astype(np.float32)
        image /= 255.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).to(dev),
                'label': torch.from_numpy(np.array([label], dtype=np.float32)).to(dev)}