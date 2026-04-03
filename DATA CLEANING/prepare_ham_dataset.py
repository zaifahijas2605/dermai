

import argparse
import csv
import hashlib
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(fn: str) -> bool:
    return os.path.splitext(fn.lower())[1] in IMG_EXTS


def md5_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_copy(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def strip_split_prefix(filename: str) -> str:
   
    if "__" in filename:
        return filename.split("__", 1)[1]
    return filename


def base_id(filename: str) -> str:
    
    fn = strip_split_prefix(filename)
    return fn.split("_aug_")[0]


@dataclass
class Item:
    cls: str
    src_path: str
    filename: str
    md5: str
    base: str
    is_aug: bool


def collect_items(dataset_root: str) -> List[Item]:
    items: List[Item] = []
    classes = sorted(
        d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))
    )
    if not classes:
        raise FileNotFoundError(f"No class folders found under: {dataset_root}")

    for cls in classes:
        for split in ("train", "test"):
            split_dir = os.path.join(dataset_root, cls, split)
            if not os.path.isdir(split_dir):
                
                continue

            for fn in os.listdir(split_dir):
                if not is_image(fn):
                    continue
                src = os.path.join(split_dir, fn)
                if not os.path.isfile(src):
                    continue
                h = md5_file(src)
                b = base_id(fn)
                aug = "_aug_" in strip_split_prefix(fn)
                items.append(Item(cls=cls, src_path=src, filename=fn, md5=h, base=b, is_aug=aug))

    return items


def deduplicate(items: List[Item]) -> Tuple[List[Item], List[Tuple[Item, Item]], List[Tuple[Item, Item]]]:
    
    first_by_md5: Dict[str, Item] = {}
    kept: List[Item] = []
    removed: List[Tuple[Item, Item]] = []
    conflicts: List[Tuple[Item, Item]] = []

    for it in items:
        if it.md5 not in first_by_md5:
            first_by_md5[it.md5] = it
            kept.append(it)
        else:
            first = first_by_md5[it.md5]
            removed.append((it, first))
            if it.cls != first.cls:
                conflicts.append((it, first))

    return kept, removed, conflicts


def split_dataset(
    kept: List[Item],
    out_dir: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    random.seed(seed)

    has_aug: Dict[Tuple[str, str], bool] = {}
    by_class_base: Dict[Tuple[str, str], List[Item]] = {}

    for it in kept:
        key = (it.cls, it.base)
        by_class_base.setdefault(key, []).append(it)
        if it.is_aug:
            has_aug[key] = True

    for split in ("train", "val", "test"):
        for cls in sorted(set(it.cls for it in kept)):
            ensure_dir(os.path.join(out_dir, split, cls))

    
    summary = {"train": 0, "val": 0, "test": 0}
    summary_by_class = {}

    for cls in sorted(set(it.cls for it in kept)):
       
        bases = sorted({b for (c, b) in by_class_base.keys() if c == cls})
        bases_with_aug = [b for b in bases if has_aug.get((cls, b), False)]
        bases_no_aug = [b for b in bases if not has_aug.get((cls, b), False)]

        for b in bases_with_aug:
            for it in by_class_base[(cls, b)]:
                dst = os.path.join(out_dir, "train", cls, it.filename)
                safe_copy(it.src_path, dst)
                summary["train"] += 1
                summary_by_class.setdefault(cls, {"train": 0, "val": 0, "test": 0})
                summary_by_class[cls]["train"] += 1

        random.shuffle(bases_no_aug)

        n = len(bases_no_aug)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_bases = bases_no_aug[:n_train]
        val_bases = bases_no_aug[n_train : n_train + n_val]
        test_bases = bases_no_aug[n_train + n_val :]

        def copy_originals(bases_list: List[str], split_name: str):
            for b in bases_list:
                for it in by_class_base[(cls, b)]:
                    
                    if split_name in ("val", "test") and it.is_aug:
                        continue
                    dst = os.path.join(out_dir, split_name, cls, it.filename)
                    safe_copy(it.src_path, dst)
                    summary[split_name] += 1
                    summary_by_class.setdefault(cls, {"train": 0, "val": 0, "test": 0})
                    summary_by_class[cls][split_name] += 1

        copy_originals(train_bases, "train")
        copy_originals(val_bases, "val")
        copy_originals(test_bases, "test")

  
    print("\n=== Split complete ===")
    print("Total images written:")
    for k in ("train", "val", "test"):
        print(f"  {k}: {summary[k]}")

    print("\nPer-class counts:")
    for cls in sorted(summary_by_class.keys()):
        d = summary_by_class[cls]
        print(f"  {cls:6s} | train {d['train']:4d} | val {d['val']:4d} | test {d['test']:4d}")

    
    def count_aug_in(split_name: str) -> int:
        cnt = 0
        for cls in os.listdir(os.path.join(out_dir, split_name)):
            cls_dir = os.path.join(out_dir, split_name, cls)
            for fn in os.listdir(cls_dir):
                if "_aug_" in fn:
                    cnt += 1
        return cnt

    aug_val = count_aug_in("val")
    aug_test = count_aug_in("test")
    print("\nAugmented files in VAL:", aug_val)
    print("Augmented files in TEST:", aug_test)
    if aug_val != 0 or aug_test != 0:
        print("WARNING: Found augmented files in val/test. Check filename patterns.")


def write_reports(
    out_dir: str,
    kept: List[Item],
    removed: List[Tuple[Item, Item]],
    conflicts: List[Tuple[Item, Item]],
) -> None:
    ensure_dir(out_dir)

    
    dup_csv = os.path.join(out_dir, "duplicates_removed.csv")
    with open(dup_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["removed_class", "removed_path", "kept_class", "kept_path", "md5"])
        for rem, kept_it in removed:
            w.writerow([rem.cls, rem.src_path, kept_it.cls, kept_it.src_path, rem.md5])

    conf_csv = os.path.join(out_dir, "label_conflicts.csv")
    with open(conf_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_class", "new_path", "first_class", "first_path", "md5"])
        for new_it, first_it in conflicts:
            w.writerow([new_it.cls, new_it.src_path, first_it.cls, first_it.src_path, new_it.md5])

    
    kept_csv = os.path.join(out_dir, "kept_items.csv")
    with open(kept_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "path", "filename", "md5", "base_id", "is_aug"])
        for it in kept:
            w.writerow([it.cls, it.src_path, it.filename, it.md5, it.base, int(it.is_aug)])

    print("\nReports written:")
    print("  ", dup_csv)
    print("  ", conf_csv)
    print("  ", kept_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="Path to dataset root that contains class folders")
    ap.add_argument("--out_dir", required=True, help="Output directory for split dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--reports_dir", default=None, help="Where to write CSV reports (default: out_dir)")
    args = ap.parse_args()

    print("Collecting files...")
    items = collect_items(args.dataset_root)
    print(f"Found {len(items)} image files before dedup.")

    print("Deduplicating (MD5 exact duplicates)...")
    kept, removed, conflicts = deduplicate(items)
    print(f"Kept: {len(kept)}")
    print(f"Removed exact duplicates: {len(removed)}")
    print(f"Label conflicts (same image in different classes): {len(conflicts)}")

    reports_dir = args.reports_dir or args.out_dir
    write_reports(reports_dir, kept, removed, conflicts)

    print("Splitting dataset (aug-safe: val/test originals only)...")
    # Start fresh
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    ensure_dir(args.out_dir)

    split_dataset(
        kept=kept,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print("\nDone ")
    print("Next training tip:")
    print('  When applying on-the-fly augmentation, only augment files that do NOT contain "_aug_".')


if __name__ == "__main__":
    main()