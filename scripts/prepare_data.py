import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict


def prepare_imagenet_subset(imagenet_path, output_path, num_classes, seed=42):
    """
    Create a subset of ImageNet with specified number of classes.
    
    Args:
        imagenet_path: Path to full ImageNet dataset with train/val folders
        output_path: Path to save the subset
        num_classes: Number of classes to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    imagenet_path = Path(imagenet_path)
    output_path = Path(output_path)
    
    for split in ['train', 'val']:
        split_path = imagenet_path / split
        if not split_path.exists():
            raise ValueError(f"ImageNet {split} directory not found at {split_path}")
        
        all_classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
        
        if split == 'train':
            selected_classes = random.sample(all_classes, num_classes)
            selected_classes.sort()
        
        output_split_path = output_path / split
        output_split_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split} split...")
        for class_name in selected_classes:
            src_class_dir = split_path / class_name
            dst_class_dir = output_split_path / class_name
            
            if not src_class_dir.exists():
                print(f"Warning: Class {class_name} not found in {split} split, skipping...")
                continue
            
            if dst_class_dir.exists():
                print(f"  {class_name} already exists, skipping...")
                continue
            
            print(f"  Copying {class_name}...")
            shutil.copytree(src_class_dir, dst_class_dir)
    
    with open(output_path / 'selected_classes.txt', 'w') as f:
        for class_name in selected_classes:
            f.write(f"{class_name}\n")
    
    print(f"\nDataset subset created at {output_path}")
    print(f"Classes: {num_classes}")
    
    train_images = sum(1 for _ in (output_path / 'train').rglob('*.JPEG'))
    val_images = sum(1 for _ in (output_path / 'val').rglob('*.JPEG'))
    print(f"Train images: {train_images}")
    print(f"Val images: {val_images}")


def verify_dataset(data_path):
    """Verify dataset structure and count images."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return False
    
    for split in ['train', 'val']:
        split_path = data_path / split
        if not split_path.exists():
            print(f"Error: {split} directory not found")
            return False
        
        classes = [d for d in split_path.iterdir() if d.is_dir()]
        total_images = sum(len(list(c.glob('*.JPEG'))) + len(list(c.glob('*.jpg'))) 
                          for c in classes)
        
        print(f"{split}: {len(classes)} classes, {total_images} images")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare ImageNet subset')
    parser.add_argument('--imagenet-path', type=str, required=True,
                       help='Path to full ImageNet dataset')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for subset')
    parser.add_argument('--num-classes', type=int, default=100,
                       help='Number of classes to include (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing dataset')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.output)
    else:
        prepare_imagenet_subset(
            args.imagenet_path,
            args.output,
            args.num_classes,
            args.seed
        )
        verify_dataset(args.output)


if __name__ == '__main__':
    main()
