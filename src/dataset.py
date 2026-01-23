from pathlib import Path
from PIL import Image # type: ignore
from torch.utils.data import Dataset # type: ignore
import torchvision.transforms as T # type: ignore
from lightning import LightningDataModule # type: ignore
from torch.utils.data import DataLoader # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import torch # type: ignore

class MultiplyBy255:
    def __call__(self, x):
        return x * 255.0

class VGGFaceDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        """
        samples: list of (image_path, class_name)
        """
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.class_to_idx[class_name]
        label = torch.tensor(label, dtype=torch.long)

        return img, label

class VGGFaceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=64,
        num_workers=4,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # VGG-Face mean (BGR order later)
        self.mean = [131.4538, 103.9875, 91.4623]

    def prepare_data(self):
        # Nothing to download
        pass

    def setup(self, stage=None):
        # 1. Collect all samples
        samples = []
        class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        for cls in class_names:
            for img_path in (self.data_dir / cls).glob("*"):
                if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    samples.append((img_path, cls))

        # 2. Split
        train_val, test = train_test_split(
            samples,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=[s[1] for s in samples],
        )

        train, val = train_test_split(
            train_val,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=[s[1] for s in train_val],
        )

        # 3. Transforms
        self.train_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),                      # RGB, [0,1]
            MultiplyBy255(),     # back to [0,255]
            T.Normalize(
                mean=[m / 255.0 for m in self.mean],
                std=[1/255.0, 1/255.0, 1/255.0],
            ),
        ])

        self.eval_transform = self.train_transform

        # 4. Datasets
        self.train_dataset = VGGFaceDataset(train, self.class_to_idx, self.train_transform)
        self.val_dataset = VGGFaceDataset(val, self.class_to_idx, self.eval_transform)
        self.test_dataset = VGGFaceDataset(test, self.class_to_idx, self.eval_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )