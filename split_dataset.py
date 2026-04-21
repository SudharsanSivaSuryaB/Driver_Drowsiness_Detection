import os
import shutil
import random

def split_data(src, dest, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    for cls in os.listdir(src):
        cls_path = os.path.join(src, cls)

        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_files in splits.items():
            split_dir = os.path.join(dest, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)

            for file in split_files:
                src_file = os.path.join(cls_path, file)
                dst_file = os.path.join(split_dir, file)
                shutil.copy(src_file, dst_file)

    print(f"✅ Done splitting: {src} → {dest}")


# Run splitting
split_data("raw/eyes", "dataset/eyes")
split_data("raw/mouth", "dataset/mouth")