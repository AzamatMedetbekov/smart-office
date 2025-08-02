import zipfile, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

TARGET_CLASSES = ["person", "chair", "monitor", "keyboard", "laptop", "phone"]
CLASS_ALIAS_MAP = {
    "persona": "person",
    "cellphone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "laptop computer": "laptop",
    "pc monitor": "monitor",
    "screen": "monitor",
    "desk chair": "chair",
    "key board": "keyboard",
}
DATASET_ZIP_DIR = Path("datasets/raw_zips")
OUTPUT_DIR = Path("datasets/smart_office")
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

(OUTPUT_DIR / "images/train").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images/val").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels/train").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels/val").mkdir(parents=True, exist_ok=True)


def extract_all_zips():
    extracted_dirs = []
    for zip_path in DATASET_ZIP_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            name = zip_path.stem
            out_dir = DATASET_ZIP_DIR / name
            zip_ref.extractall(out_dir)
            extracted_dirs.append(out_dir)
    return extracted_dirs


def load_class_names(yaml_path):
    with open(yaml_path, "r") as f:
        for line in f:
            if line.strip().startswith("names:"):
                return eval(line.strip().split(":", 1)[1])
    return []


def normalize_class_name(name):
    return CLASS_ALIAS_MAP.get(name.lower(), name.lower())


def copy_and_filter_labels_and_images(src_img_dir, src_lbl_dir, class_map):
    for lbl_file in src_lbl_dir.glob("*.txt"):
        with open(lbl_file, "r") as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            cid, *coords = line.strip().split()
            cid = int(cid)
            class_name = class_map.get(cid)
            if not class_name:
                continue
            norm = normalize_class_name(class_name)
            if norm in TARGET_CLASSES:
                new_id = TARGET_CLASSES.index(norm)
                filtered_lines.append(f"{new_id} {' '.join(coords)}\n")

        if filtered_lines:
            img_path = next(
                (
                    src_img_dir / (lbl_file.stem + ext)
                    for ext in IMAGE_EXTS
                    if (src_img_dir / (lbl_file.stem + ext)).exists()
                ),
                None,
            )
            if img_path:
                shutil.copy(img_path, OUTPUT_DIR / "images/train" / img_path.name)
                with open(OUTPUT_DIR / "labels/train" / lbl_file.name, "w") as out_f:
                    out_f.writelines(filtered_lines)


def split_train_val():
    all_imgs = list((OUTPUT_DIR / "images/train").glob("*"))
    train_imgs, val_imgs = train_test_split(all_imgs, test_size=0.2, random_state=42)
    for img in val_imgs:
        lbl = OUTPUT_DIR / "labels/train" / (img.stem + ".txt")
        shutil.move(img, OUTPUT_DIR / "images/val" / img.name)
        shutil.move(lbl, OUTPUT_DIR / "labels/val" / lbl.name)


def write_data_yaml():
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(f"train: {OUTPUT_DIR / 'images/train'}\n")
        f.write(f"val: {OUTPUT_DIR / 'images/val'}\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write(f"names: {TARGET_CLASSES}\n")


print("📦 Extracting ZIPs...")
dirs = extract_all_zips()

print("🔍 Filtering and copying relevant labels...")
for dataset_dir in dirs:
    yaml_path = next(dataset_dir.glob("*.yaml"), None)
    if not yaml_path:
        continue

    class_names = load_class_names(yaml_path)
    class_map = {i: name for i, name in enumerate(class_names)}

    img_dir = dataset_dir / "train" / "images"
    lbl_dir = dataset_dir / "train" / "labels"

    if img_dir.exists() and lbl_dir.exists():
        copy_and_filter_labels_and_images(img_dir, lbl_dir, class_map)

print("🧪 Splitting into train/val...")
split_train_val()

print("📝 Writing data.yaml...")
write_data_yaml()

print("✅ Dataset prepared successfully at:", OUTPUT_DIR)
