"""
Treino YOLOv8 sem interface gráfica: usado pela tela do Treinador e pelo tools/train_model.py.
"""
import logging
import os
import shutil
import time
from typing import Callable, Dict, Optional

import yaml


class TrainingStopped(Exception):
    """Levantada dentro do treino quando o usuário pede para parar."""


def normalize_data_yaml(data_path: str) -> str:
    """
    Gera um data_local.yaml ao lado do original com 'path' absoluto e os prefixos '../' corrigidos
    (exports do Roboflow apontam para '../train/images' mesmo com as pastas ao lado do yaml).
    """
    data_path = os.path.abspath(data_path)
    root = os.path.dirname(data_path)
    with open(data_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = root
    for key in ("train", "val", "test"):
        value = cfg.get(key)
        if not value:
            continue
        if not os.path.exists(os.path.join(root, value)) and value.startswith("../"):
            value = value[3:]
        cfg[key] = value

    # Alguns exports chamam a validação de 'valid'
    if not cfg.get("val") or not os.path.exists(os.path.join(root, cfg["val"])):
        for alt in ("valid/images", "val/images"):
            if os.path.exists(os.path.join(root, alt)):
                cfg["val"] = alt
                break

    out_path = os.path.join(root, "data_local.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out_path


def count_images(root: str, rel: str) -> int:
    folder = os.path.join(root, rel)
    if not os.path.isdir(folder):
        return 0
    return sum(1 for n in os.listdir(folder) if n.lower().endswith((".jpg", ".jpeg", ".png")))


def default_device() -> str:
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


class _LogForwarder(logging.Handler):
    """Encaminha o log do ultralytics para um callback."""

    def __init__(self, callback: Callable[[str], None]):
        super().__init__(logging.INFO)
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            for line in self.format(record).splitlines():
                if line.strip():
                    self.callback(line)
        except Exception:
            pass


def run_training(
    data_yaml: str,
    out_path: str,
    model: str = "yolov8n.pt",
    imgsz: int = 512,
    epochs: int = 100,
    batch: int = 16,
    patience: int = 30,
    device: str = "auto",
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Treina e instala o melhor modelo em out_path.

    on_progress(fração 0-1, texto) é chamado durante o treino; should_stop() é consultado a cada lote.
    Retorna {"status": "done"|"stopped", "path": out_path|None, "names": {...}|None, "checkpoint": caminho|None}.
    Se o treino for interrompido NADA é instalado em out_path (não sobrescreve um modelo bom);
    o último checkpoint fica em "checkpoint".
    """
    log = on_log or (lambda msg: None)
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"data.yaml não encontrado: {data_yaml}")

    import torch  # noqa: F401  (import tardio: deixa a interface abrir rápido)
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    if device == "auto":
        device = default_device()
    if device == "cpu":
        log("[AVISO] Sem GPU: na CPU, 100 épocas a 512 px levam VÁRIAS HORAS. "
            "Para um teste rápido use 5 épocas e imgsz 320, ou treine no Google Colab.")

    local_yaml = normalize_data_yaml(data_yaml)
    with open(local_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    n_train = count_images(cfg["path"], cfg.get("train", ""))
    n_val = count_images(cfg["path"], cfg.get("val", ""))
    log(f"[DATASET] classes: {cfg.get('names')} | treino: {n_train} | validação: {n_val}")
    if n_train == 0:
        raise ValueError("Nenhuma imagem de treino encontrada. Confira o caminho 'train' do data.yaml.")
    if n_train < 50:
        log("[AVISO] Menos de 50 imagens de treino: o modelo tende a ficar fraco. Ideal: 100 a 300.")

    yolo = YOLO(model)
    started = time.time()
    state = {"batches": 0, "batch_in_epoch": 0}

    def report(epoch_float: float, extra: str = ""):
        if not on_progress:
            return
        fraction = min(max(epoch_float / epochs, 0.0), 1.0)
        elapsed = time.time() - started
        eta = (elapsed / fraction - elapsed) if fraction > 0.01 else None
        eta_txt = f" | faltam ~{int(eta // 60)} min" if eta is not None else ""
        on_progress(fraction, f"{extra}{eta_txt}".strip(" |"))

    def on_epoch_start(trainer):
        state["batch_in_epoch"] = 0

    def on_batch_end(trainer):
        if should_stop and should_stop():
            raise TrainingStopped()
        state["batch_in_epoch"] += 1
        nb = max(len(trainer.train_loader), 1)
        report(trainer.epoch + state["batch_in_epoch"] / nb, f"Época {trainer.epoch + 1}/{trainer.epochs}")

    def on_fit_epoch_end(trainer):
        metrics = getattr(trainer, "metrics", {}) or {}
        map50 = metrics.get("metrics/mAP50(B)")
        extra = f"Época {trainer.epoch + 1}/{trainer.epochs}" + (f" | mAP50 {map50:.2f}" if map50 is not None else "")
        report(trainer.epoch + 1, extra)

    yolo.add_callback("on_train_epoch_start", on_epoch_start)
    yolo.add_callback("on_train_batch_end", on_batch_end)
    yolo.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    handler = _LogForwarder(log)
    LOGGER.addHandler(handler)
    run_name = os.path.splitext(os.path.basename(out_path))[0]
    status = "done"
    try:
        yolo.train(
            data=local_yaml, imgsz=imgsz, epochs=epochs, batch=batch, patience=patience, device=device,
            project=os.path.abspath(os.path.join("runs", "train")), name=run_name, exist_ok=True,
            plots=False,
        )
    except TrainingStopped:
        status = "stopped"
        log("[TREINO] Interrompido pelo usuário.")
    finally:
        LOGGER.removeHandler(handler)

    save_dir = str(yolo.trainer.save_dir)
    best = os.path.join(save_dir, "weights", "best.pt")
    last = os.path.join(save_dir, "weights", "last.pt")
    source = best if os.path.isfile(best) else (last if os.path.isfile(last) else None)
    if source is None:
        log("[TREINO] Nenhum checkpoint foi gerado (parou antes do fim da primeira época).")
        return {"status": status, "path": None, "names": None, "checkpoint": None}
    if status == "stopped":
        log(f"[TREINO] Nada foi instalado. Último checkpoint salvo em: {source}")
        return {"status": status, "path": None, "names": None, "checkpoint": source}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shutil.copy2(source, out_path)
    names = YOLO(out_path).names
    log(f"[TREINO] Modelo instalado em {out_path} | classes: {names}")
    return {"status": status, "path": out_path, "names": names, "checkpoint": source}
