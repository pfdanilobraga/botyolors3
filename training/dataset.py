"""
Operações de dataset YOLO, sem interface gráfica.

Layout (um dataset por item, dentro de dataset/):
    dataset/<item>/images/        imagens coletadas (.jpg/.png)
    dataset/<item>/labels/        um .txt por imagem anotada (formato YOLO: classe cx cy w h, normalizado)
    dataset/<item>/classes.txt    nomes das classes, um por linha (o índice é o número da linha)
    dataset/<item>/descartadas/   imagens descartadas na anotação (nada é apagado)
Gerados por prepare_split():
    dataset/<item>/train|valid/{images,labels}  e  dataset/<item>/data.yaml
"""
import os
import random
import re
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

DATASET_ROOT = "dataset"
IMG_EXT = (".jpg", ".jpeg", ".png")

# (classe, cx, cy, w, h) com cx/cy/w/h normalizados (0-1)
Box = Tuple[int, float, float, float, float]


def sanitize_name(name: str) -> str:
    """Nome seguro para pasta/arquivo: letras, números, '-' e '_'."""
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return clean or "item"


def dataset_dir(item: str) -> str:
    return os.path.join(DATASET_ROOT, sanitize_name(item))


def images_dir(item: str) -> str:
    return os.path.join(dataset_dir(item), "images")


def labels_dir(item: str) -> str:
    return os.path.join(dataset_dir(item), "labels")


def create_dataset(item: str) -> str:
    """Cria a estrutura do dataset (se ainda não existir) e devolve o nome saneado."""
    item = sanitize_name(item)
    os.makedirs(images_dir(item), exist_ok=True)
    os.makedirs(labels_dir(item), exist_ok=True)
    return item


def list_items() -> List[str]:
    """Datasets existentes (pastas de dataset/ que têm 'images')."""
    if not os.path.isdir(DATASET_ROOT):
        return []
    return sorted(
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d, "images"))
    )


def list_images(item: str) -> List[str]:
    folder = images_dir(item)
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(IMG_EXT))


def label_path(item: str, image_name: str) -> str:
    return os.path.join(labels_dir(item), os.path.splitext(image_name)[0] + ".txt")


# ------------------------------------------------------------------ classes

def load_classes(item: str) -> List[str]:
    path = os.path.join(dataset_dir(item), "classes.txt")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_classes(item: str, classes: List[str]) -> None:
    os.makedirs(dataset_dir(item), exist_ok=True)
    with open(os.path.join(dataset_dir(item), "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + ("\n" if classes else ""))


# ------------------------------------------------------------------ rótulos

def read_labels(item: str, image_name: str) -> Optional[List[Box]]:
    """Caixas da imagem, [] se marcada sem objetos, None se ainda não anotada."""
    path = label_path(item, image_name)
    if not os.path.isfile(path):
        return None
    boxes: List[Box] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                boxes.append((int(parts[0]), *[float(v) for v in parts[1:]]))
            except ValueError:
                continue
    return boxes


def write_labels(item: str, image_name: str, boxes: List[Box]) -> None:
    """Grava as caixas. Lista vazia grava arquivo vazio (imagem 'sem objetos', usada como negativa)."""
    os.makedirs(labels_dir(item), exist_ok=True)
    with open(label_path(item, image_name), "w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def remove_labels(item: str, image_name: str) -> None:
    path = label_path(item, image_name)
    if os.path.isfile(path):
        os.remove(path)


def discard_image(item: str, image_name: str) -> None:
    """Move a imagem (e o rótulo) para 'descartadas/'. Nada é apagado."""
    target = os.path.join(dataset_dir(item), "descartadas")
    os.makedirs(target, exist_ok=True)
    shutil.move(os.path.join(images_dir(item), image_name), os.path.join(target, image_name))
    lbl = label_path(item, image_name)
    if os.path.isfile(lbl):
        shutil.move(lbl, os.path.join(target, os.path.basename(lbl)))


def stats(item: str) -> Dict:
    """Resumo: total, anotadas, negativas (sem objetos), caixas totais e por classe."""
    images = list_images(item)
    classes = load_classes(item)
    per_class = [0] * len(classes)
    annotated = negatives = boxes_total = 0
    for name in images:
        boxes = read_labels(item, name)
        if boxes is None:
            continue
        annotated += 1
        if not boxes:
            negatives += 1
        for cls, *_ in boxes:
            boxes_total += 1
            if 0 <= cls < len(per_class):
                per_class[cls] += 1
    return {
        "total": len(images), "annotated": annotated, "negatives": negatives,
        "boxes": boxes_total, "per_class": per_class, "classes": classes,
    }


# ------------------------------------------------------------------ variedade (o que anotar a seguir)

def _thumbnail(item: str, name: str) -> Optional[np.ndarray]:
    """Miniatura 48x27 colorida da imagem (em cache na pasta thumbs/), para comparar imagens."""
    src = os.path.join(images_dir(item), name)
    folder = os.path.join(dataset_dir(item), "thumbs")
    path = os.path.join(folder, os.path.splitext(name)[0] + ".png")
    if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(src):
        thumb = cv2.imread(path)
    else:
        image = cv2.imread(src)
        if image is None:
            return None
        thumb = cv2.resize(image, (48, 27), interpolation=cv2.INTER_AREA)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(path, thumb)
    return None if thumb is None else thumb.astype(np.float32)


def most_different_unlabeled(item: str) -> Optional[str]:
    """
    Imagem ainda sem rótulo que mais se parece COM MENOS das já anotadas (a mais "diferente"): anotá-la
    ensina mais ao modelo do que mais uma imagem parecida com as que já foram marcadas.
    """
    unlabeled, annotated = [], []
    for name in list_images(item):
        (annotated if read_labels(item, name) is not None else unlabeled).append(name)
    if not unlabeled:
        return None
    ref = [t for t in (_thumbnail(item, n) for n in annotated) if t is not None]
    if not ref:
        return unlabeled[0]
    best_name, best_dist = unlabeled[0], -1.0
    for name in unlabeled:
        thumb = _thumbnail(item, name)
        if thumb is None:
            continue
        dist = min(float(np.abs(thumb - r).mean()) for r in ref)
        if dist > best_dist:
            best_name, best_dist = name, dist
    return best_name


# ------------------------------------------------------------------ divisão treino/validação

def prepare_split(item: str, val_ratio: float = 0.2, seed: int = 42) -> Dict:
    """
    Gera train/ e valid/ (cópias) e o data.yaml a partir das imagens anotadas.
    Imagens sem arquivo de rótulo são ignoradas. Refaz as pastas train/valid a cada chamada.
    """
    item = sanitize_name(item)
    classes = load_classes(item)
    if not classes:
        raise ValueError("Defina ao menos uma classe antes de preparar o dataset.")

    annotated = [n for n in list_images(item) if read_labels(item, n) is not None]
    if len(annotated) < 2:
        raise ValueError("São necessárias ao menos 2 imagens anotadas.")

    rng = random.Random(seed)
    rng.shuffle(annotated)
    n_val = max(1, int(round(len(annotated) * val_ratio)))
    splits = {"valid": annotated[:n_val], "train": annotated[n_val:]}

    root = os.path.abspath(dataset_dir(item))
    for split, names in splits.items():
        split_dir = os.path.join(root, split)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)  # pasta gerada por esta função
        os.makedirs(os.path.join(split_dir, "images"))
        os.makedirs(os.path.join(split_dir, "labels"))
        for name in names:
            shutil.copy2(os.path.join(images_dir(item), name), os.path.join(split_dir, "images", name))
            shutil.copy2(label_path(item, name), os.path.join(split_dir, "labels", os.path.basename(label_path(item, name))))

    data_yaml = os.path.join(root, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "path": root, "train": "train/images", "val": "valid/images",
            "nc": len(classes), "names": classes,
        }, f, allow_unicode=True, sort_keys=False)

    return {"yaml": data_yaml, "n_train": len(splits["train"]), "n_val": len(splits["valid"])}


def make_colab_zip(item: str) -> str:
    """Empacota train/, valid/ e um data.yaml com caminhos relativos para subir no Colab."""
    item = sanitize_name(item)
    root = os.path.abspath(dataset_dir(item))
    classes = load_classes(item)
    zip_path = os.path.join(root, f"{item}_colab.zip")
    colab_yaml = yaml.safe_dump(
        {"train": "train/images", "val": "valid/images", "nc": len(classes), "names": classes},
        allow_unicode=True, sort_keys=False,
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for split in ("train", "valid"):
            for sub in ("images", "labels"):
                folder = os.path.join(root, split, sub)
                for name in sorted(os.listdir(folder)):
                    z.write(os.path.join(folder, name), f"{split}/{sub}/{name}")
        z.writestr("data.yaml", colab_yaml)
    return zip_path
