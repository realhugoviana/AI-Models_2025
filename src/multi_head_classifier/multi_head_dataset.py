from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
import numpy as np

class MultiplyBy255:
    def __call__(self, x):
        return x * 255.0

class MultiHeadVGGFaceDataset(Dataset):
    def __init__(self, samples, name_to_idx, sex_to_idx, transform=None):
        """
        samples: list of (image_path, celebrity_name, sex)
        """
        self.samples = samples
        self.name_to_idx = name_to_idx
        self.sex_to_idx = sex_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, celebrity_name, sex = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_name = self.name_to_idx[celebrity_name]
        label_sex = self.sex_to_idx[sex]
        
        label_name = torch.tensor(label_name, dtype=torch.long)
        label_sex = torch.tensor(label_sex, dtype=torch.long)

        return img, label_name, label_sex

def stratified_split_per_class(samples, n_train_per_class, n_val_per_class, random_state=42):
    """
    Split samples ensuring exact counts per class.
    
    Args:
        samples: list of (image_path, celebrity_name, sex)
        n_train_per_class: number of training samples per celebrity
        n_val_per_class: number of validation samples per celebrity
        random_state: random seed
        
    Returns:
        train, val, test splits
    """
    np.random.seed(random_state)
    
    # Group samples by celebrity
    celeb_samples = {}
    for sample in samples:
        _, celeb_name, _ = sample
        if celeb_name not in celeb_samples:
            celeb_samples[celeb_name] = []
        celeb_samples[celeb_name].append(sample)
    
    train = []
    val = []
    test = []
    
    skipped_celebs = []
    
    for celeb_name, celeb_imgs in celeb_samples.items():
        n_total = len(celeb_imgs)
        n_test = n_total - n_train_per_class - n_val_per_class
        
        # Check if we have enough samples
        if n_total < (n_train_per_class + n_val_per_class + 1):
            skipped_celebs.append((celeb_name, n_total))
            print(f"  ⚠ Skipping {celeb_name}: only {n_total} samples " +
                  f"(need at least {n_train_per_class + n_val_per_class + 1})")
            continue
        
        # Shuffle the samples for this celebrity
        shuffled = celeb_imgs.copy()
        np.random.shuffle(shuffled)
        
        # Split
        train.extend(shuffled[:n_train_per_class])
        val.extend(shuffled[n_train_per_class:n_train_per_class + n_val_per_class])
        test.extend(shuffled[n_train_per_class + n_val_per_class:])
    
    if skipped_celebs:
        print(f"\n{'='*70}")
        print(f"WARNING: Skipped {len(skipped_celebs)} celebrities with insufficient samples:")
        print(f"{'='*70}")
        for name, count in skipped_celebs:
            print(f"  {name}: {count} samples")
        print(f"\nMinimum required: {n_train_per_class + n_val_per_class + 1} samples per celebrity")
    
    return train, val, test

class MultiHeadVGGFaceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=64,
        num_workers=16,
        n_train_per_class=12,
        n_val_per_class=2,
        random_state=42
    ):
        """
        Args:
            data_dir: Path to data directory
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            n_train_per_class: Number of training samples per celebrity (default: 12)
            n_val_per_class: Number of validation samples per celebrity (default: 2)
            random_state: Random seed for reproducibility
        
        Note: Test samples = total_samples - n_train_per_class - n_val_per_class
              For Jeff Bezos with 17 samples: 12 train, 2 val, 3 test
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_train_per_class = n_train_per_class
        self.n_val_per_class = n_val_per_class
        self.random_state = random_state

        # VGG-Face mean (BGR order later)
        self.mean = [131.4538, 103.9875, 91.4623]

    def prepare_data(self):
        # Nothing to download
        pass

    def setup(self, stage=None):
        # 1. Collect all samples and extract celebrity names and sex
        samples = []
        celebrity_names = set()
        sexes = set()
        celeb_counts = {}

        # Iterate through folders with format: FirstName_MiddleName_LastName_Sex
        for folder in self.data_dir.iterdir():
            if not folder.is_dir():
                continue
            
            folder_name = folder.name
            
            # Parse folder name: split by underscore and extract sex (last part)
            parts = folder_name.split('_')
            if len(parts) < 2:
                print(f"Warning: Skipping folder with invalid format: {folder_name}")
                continue
            
            sex = parts[-1]  # Last part is sex (M/F/Unknown)
            celebrity_name = '_'.join(parts[:-1])  # Everything before sex is the name
            
            # Skip folders with Unknown sex
            if sex not in ['M', 'F']:
                print(f"Warning: Skipping folder with unknown sex: {folder_name}")
                continue
            
            celebrity_names.add(celebrity_name)
            sexes.add(sex)
            
            # Count images for this celebrity
            img_count = 0
            # Add all images from this folder
            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    samples.append((img_path, celebrity_name, sex))
                    img_count += 1
            
            celeb_counts[celebrity_name] = img_count

        print(f"\n{'='*70}")
        print("DATASET OVERVIEW")
        print(f"{'='*70}")
        print(f"Total samples collected: {len(samples)}")
        print(f"Total celebrities: {len(celebrity_names)}")
        print(f"Sex classes: {sorted(sexes)}")
        
        # Show sample distribution
        print(f"\nSample distribution per celebrity:")
        print(f"  Min samples: {min(celeb_counts.values())}")
        print(f"  Max samples: {max(celeb_counts.values())}")
        print(f"  Average samples: {sum(celeb_counts.values()) / len(celeb_counts):.1f}")
        
        # Show celebrities with few samples
        few_samples = [(name, count) for name, count in celeb_counts.items() 
                       if count < (self.n_train_per_class + self.n_val_per_class + 1)]
        if few_samples:
            print(f"\nCelebrities with < {self.n_train_per_class + self.n_val_per_class + 1} samples:")
            for name, count in sorted(few_samples, key=lambda x: x[1])[:10]:
                print(f"  {name}: {count} samples")
            if len(few_samples) > 10:
                print(f"  ... and {len(few_samples) - 10} more")

        # 2. Split data using per-class stratification
        print(f"\n{'='*70}")
        print("SPLITTING DATA")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Train samples per celebrity: {self.n_train_per_class}")
        print(f"  Val samples per celebrity: {self.n_val_per_class}")
        print(f"  Test samples per celebrity: remaining (e.g., 17 total → 3 test)")
        
        train, val, test = stratified_split_per_class(
            samples,
            n_train_per_class=self.n_train_per_class,
            n_val_per_class=self.n_val_per_class,
            random_state=self.random_state
        )

        # Recalculate celebrity names based on what's actually in the splits
        # (some might have been filtered out)
        train_celebs = set([name for _, name, _ in train])
        
        # 3. Create mappings (only for celebrities that made it through filtering)
        self.name_to_idx = {name: i for i, name in enumerate(sorted(train_celebs))}
        self.sex_to_idx = {sex: i for i, sex in enumerate(sorted(sexes))}
        
        # Create reverse mappings for inference
        self.idx_to_name = {i: name for name, i in self.name_to_idx.items()}
        self.idx_to_sex = {i: sex for sex, i in self.sex_to_idx.items()}
        
        print(f"\n{'='*70}")
        print("FINAL SPLIT STATISTICS")
        print(f"{'='*70}")
        print(f"Train samples: {len(train)}")
        print(f"Val samples: {len(val)}")
        print(f"Test samples: {len(test)}")
        print(f"Total samples (after filtering): {len(train) + len(val) + len(test)}")
        print(f"Celebrities included: {len(self.name_to_idx)}")
        print(f"Sex classes: {self.sex_to_idx}")
        
        # Verify per-class counts
        train_per_celeb = {}
        for _, name, _ in train:
            train_per_celeb[name] = train_per_celeb.get(name, 0) + 1
        
        val_per_celeb = {}
        for _, name, _ in val:
            val_per_celeb[name] = val_per_celeb.get(name, 0) + 1
        
        test_per_celeb = {}
        for _, name, _ in test:
            test_per_celeb[name] = test_per_celeb.get(name, 0) + 1
        
        print(f"\nVerification (showing first 5 celebrities):")
        for i, name in enumerate(sorted(self.name_to_idx.keys())[:5]):
            train_count = train_per_celeb.get(name, 0)
            val_count = val_per_celeb.get(name, 0)
            test_count = test_per_celeb.get(name, 0)
            total = train_count + val_count + test_count
            print(f"  {name}: {train_count} train, {val_count} val, {test_count} test (total: {total})")

        # 4. Transforms
        self.train_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),                      # RGB, [0,1]
            MultiplyBy255(),                   # back to [0,255]
            T.Normalize(
                mean=[m / 255.0 for m in self.mean],
                std=[1/255.0, 1/255.0, 1/255.0],
            ),
        ])

        self.eval_transform = self.train_transform

        # 5. Datasets
        self.train_dataset = MultiHeadVGGFaceDataset(
            train, self.name_to_idx, self.sex_to_idx, self.train_transform
        )
        self.val_dataset = MultiHeadVGGFaceDataset(
            val, self.name_to_idx, self.sex_to_idx, self.eval_transform
        )
        self.test_dataset = MultiHeadVGGFaceDataset(
            test, self.name_to_idx, self.sex_to_idx, self.eval_transform
        )

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