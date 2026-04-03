import os
ROOT = r"C:\Users\LENOVO\OneDrive\Desktop\Top Up-FINAL SEM\Disertation\FINAL\ZAIFA FINAL PROJECT\HAM_2WAY_SPLIT_OUT\train"

total_aug = 0
total_images = 0

for cls in os.listdir(ROOT):
    cls_dir = os.path.join(ROOT, cls)
    if not os.path.isdir(cls_dir):
        continue
    
    aug_count = 0
    total_count = 0
    
    for fn in os.listdir(cls_dir):
        if "_aug_" in fn:
            aug_count += 1
        total_count += 1
    
    print(f"{cls:6s} | Augmented: {aug_count:4d} | Total Train: {total_count:4d}")
    
    total_aug += aug_count
    total_images += total_count

print("\nTOTAL TRAIN IMAGES:", total_images)
print("TOTAL AUGMENTED IMAGES:", total_aug)
print("TOTAL ORIGINAL IMAGES IN TRAIN:", total_images - total_aug)