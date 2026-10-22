
import numpy as np
import nibabel as nib
from collections import defaultdict 
from pathlib import Path

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res

def preprocess_data(pair):
    mask, image = pair
    mask = mask.get_fdata()
    image = image.get_fdata()
    """ NORMALIZE IMAGE AND ONEHOT ENCODE THE MASK """
    image = (image - image.mean()) / (image.std() + 1e-8)
    mask = to_channels(mask)
    return (mask, image)


def load(folder, split=0): 
    folders = ["test", "train", "validate"]
    patients = defaultdict(list)
    for dir in folders: 
        seg = Path(f"{folder}/keras_slices_data/keras_slices_seg_{dir}")
        img = Path(f"{folder}/keras_slices_data/keras_slices_{dir}")
        masks = sorted([f for f in seg.iterdir() if f.is_file()])
        images = sorted([f for f in img.iterdir() if f.is_file()])
        c = 0
        for mask, image in zip(masks, images):
            mparts = list(filter(lambda x: x.isdigit(), mask.name.split("_")))
            imparts = list(filter(lambda x: x.isdigit(), image.name.split("_")))
            if not mparts == imparts:
                print(f"{mparts}, {imparts}")
                exit("INVALID DATASET, FILES ARENT IN SEQUENCE")
            patient = mparts[0]
            mask = nib.load(f"{seg}/{mask.name}")
            image = nib.load(f"{img}/{image.name}")
            if image.shape != mask.shape:
                exit("DIMENSIONS DONT MATCH")

            if image.shape != (256, 128):
                """ DIMENSIONS ARE NOT STANDARD, SKIPPING IMAGE """
                continue

            c+=1
            patients[patient].append((mask, image))

                
        print(f"counted {c} files in {dir}, {len(patients)} unique patients")
    return map(preprocess_data(), patients.values()), len(patients)