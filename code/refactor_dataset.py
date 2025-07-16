from pathlib import Path
from PIL import Image as PILImage
import torch
import numpy as np
from fastai.vision.all import *
from skimage.morphology import skeletonize, binary_erosion, binary_dilation, remove_small_objects
from skimage.measure import label, regionprops
from skimage.draw import polygon

from config import *

mapping = dict()
mapping[(1, 1, 1)] = 1 #black (easy)
mapping[(255, 1, 255)] = 2 #pink (hard)
mapping[(255, 255, 255)] = 0 #white (background)

def get_segmentation(x):
    folder = x.parent.name
    return Path(MASKS_PATH) / folder / x.name.replace('.jpg', '.png')

def random_crop_tfm(size=(512, 800), randx=(0.0, 1.0), randy=(0.0, 1.0)):
    class RandomCrop(Transform):
        def encodes(self, tup: tuple):
            img, mask = tup
            h, w = size
            y = int((img.shape[1] - h) * random.uniform(*randy))
            x = int((img.shape[2] - w) * random.uniform(*randx))
            return img[:, y:y+h, x:x+w], mask[:, y:y+h, x:x+w]
    return RandomCrop()

def pad_tensor(t, multiple=8):
    def divround_up(val, step): return (val + step - 1) // step * step
    padded = torch.zeros(t.shape[0], divround_up(t.shape[1], multiple), divround_up(t.shape[2], multiple))
    padded[:, :t.shape[1], :t.shape[2]] = t
    return padded

class PadToMultiple(Transform):
    def __init__(self, multiple=8): self.multiple = multiple
    def encodes(self, x: tuple):
        img, mask = x
        return pad_tensor(img, self.multiple), pad_tensor(mask, self.multiple)

class CutInHalfTransform(Transform):
    def __init__(self, enabled=True): self.enabled = enabled
    def encodes(self, x: tuple):
        if not self.enabled: return x
        img, mask = x
        if random.random() < 0.5:
            img, mask = img[:, :, :img.shape[2]//2], mask[:, :, :mask.shape[2]//2]
        else:
            img, mask = img[:, :, img.shape[2]//2:], mask[:, :, mask.shape[2]//2:]
        return img.contiguous(), mask.contiguous()
    
def load_mask_with_ignore(fn, ignore=True, areaThreshold=3, cache_suffix='.cache.png'):
    """
    Loads a mask with ignore logic and small object removal, using a cache file if available.
    If the cache does not exist, processes the mask and saves the result as a cache.
    """
    from pathlib import Path
    from PIL import Image as PILImage
    import os

    fn = Path(fn)
    cache_fn = fn.with_name(fn.stem + cache_suffix)
    if cache_fn.exists():
        # Load cached mask
        return PILMask.create(cache_fn)
    else:
        # Process and cache
        im = PILMask.create(fn)
        px = tensor(im)
        px = px.unsqueeze(0) if px.ndim == 2 else px

        palette = im.getpalette()
        if palette is None:
            raise ValueError(f"Image {fn} does not have a palette.")
        new_px = px.clone()
        for x in range(int(px.max().item()) + 1):
            color = tuple(palette[x * 3:x * 3 + 3])
            if color in mapping:
                new_px[px == x] = mapping[color]
        px = new_px

        if ignore:
            skeleton = torch.zeros(px.shape).bool()
            for val in px[0].unique():
                if val == 0: continue
                sk = tensor(skeletonize((px[0] == val).numpy()))
                skeleton |= sk.bool()

            text_mask = (px[0] != 0).numpy()
            eroded, dilated = text_mask.copy(), text_mask.copy()
            for _ in range(3):
                eroded = binary_erosion(eroded)
                dilated = binary_dilation(dilated)
            eroded = tensor(np.expand_dims(eroded, 0))
            dilated = tensor(np.expand_dims(dilated, 0))
            px[((dilated != 0) * ((eroded == 0) & (skeleton == 0)))] += len(mapping)

        if areaThreshold is not None:
            labels = label(px[0].cpu() == 0, connectivity=2)
            for region in regionprops(labels):
                if region.area <= areaThreshold:
                    val = px[0, tensor(binary_dilation(labels == region.label))].max()
                    px[0, region.slice[0], region.slice[1]][px[0, region.slice[0], region.slice[1]] == 0] = val

            labels = label(px[0].cpu(), connectivity=2)
            remove_small_objects(labels, areaThreshold + 1)
            px[0][labels == 0] = 0

        mask = PILMask.create(px.squeeze().byte())
        mask.save(cache_fn)
        return mask