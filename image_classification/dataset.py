import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageClassificationDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, classes=[]):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label


if __name__ == "__main__":
    albumentations_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    albumentations_dataset = ImageClassificationDataset(
        file_paths=[
            "./images/image_1.jpg",
            "./images/image_2.jpg",
            "./images/image_3.jpg",
        ],
        labels=[1, 2, 3],
        transform=albumentations_transform,
    )
