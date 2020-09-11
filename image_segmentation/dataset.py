import cv2
import os
from torch.utils.data import Dataset
import numpy as np

def preprocess_mask(mask):
    #segmentate "object" vs background
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


class ImageSegmentationDataset(Dataset):
    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist', 'unlabelled']

    CLASSES = ["pet"]

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.transform = augmentation

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        mask = preprocess_mask(mask)

        # # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.ids)