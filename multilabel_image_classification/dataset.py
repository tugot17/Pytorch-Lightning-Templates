import cv2
import torch
from torch.utils.data import Dataset
import json


class MultiLabelImageClassificationDataset(Dataset):
    def __init__(self, df, transform=None, classes=[]):
        self.paths = df["Path"].tolist()

        self.labels = df["Label"].tolist()
        self.labels = list(map(lambda row: json.loads(row), self.labels))
        self.labels = torch.tensor(self.labels, dtype=torch.float)

        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label = self.labels[idx]

        image_path = self.paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label