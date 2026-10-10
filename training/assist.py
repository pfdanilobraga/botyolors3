"""
Assistente de anotação: um modelo rápido que aprende com as imagens que o usuário já anotou e sugere
caixas nas imagens novas. A cada rodada de treino ele continua do que já aprendeu (parte dos pesos do
assistente anterior), então fica mais preciso conforme o usuário corrige e anota mais imagens.

Sem interface gráfica; a aba "Anotar" do Treinador usa este módulo.
"""
import json
import math
import os
import shutil
import threading
from typing import Callable, Dict, List, Optional, Tuple

from training import dataset as ds
from training.trainer import run_training

# (classe, x1, y1, x2, y2, confiança) em pixels da imagem
Suggestion = Tuple[int, float, float, float, float, float]

DEFAULT_IMGSZ = 416
BATCH = 4
FIRST_STEPS = 200          # passos de treino na 1ª rodada (poucos passos = confiança baixa mesmo acertando a posição)
CONTINUE_STEPS = 100       # passos nas rodadas seguintes: já parte do que foi aprendido
MIN_AUTO_EPOCHS, MAX_AUTO_EPOCHS = 3, 60
MIN_ANNOTATED = 8          # menos que isso o assistente quase não aprende
RETRAIN_HINT_EVERY = 20    # sugere atualizar a cada N imagens anotadas novas


def model_path(item: str) -> str:
    return os.path.join(ds.dataset_dir(item), "assist_model.pt")


def _info_path(item: str) -> str:
    return os.path.join(ds.dataset_dir(item), "assist_info.json")


def load_info(item: str) -> Dict:
    try:
        with open(_info_path(item), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_info(item: str, info: Dict) -> None:
    with open(_info_path(item), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def trained_count(item: str) -> int:
    """Quantas imagens anotadas existiam quando o assistente foi treinado pela última vez."""
    return int(load_info(item).get("annotated", 0))


def new_since_training(item: str) -> int:
    """Imagens anotadas desde o último treino do assistente (todas, se ele ainda não existe)."""
    annotated = ds.stats(item)["annotated"]
    return annotated - (trained_count(item) if os.path.isfile(model_path(item)) else 0)


def needs_update(item: str) -> bool:
    """True quando há bastante imagem anotada nova desde o último treino rápido."""
    annotated = ds.stats(item)["annotated"]
    if annotated < MIN_ANNOTATED:
        return False
    if not os.path.isfile(model_path(item)):
        return True
    return annotated - trained_count(item) >= RETRAIN_HINT_EVERY


def auto_epochs(n_train: int, continuing: bool = False) -> int:
    """
    Épocas para chegar a um número FIXO de passos de treino (e não de épocas): assim cada rodada leva
    aproximadamente o mesmo tempo, qualquer que seja o tamanho do dataset (~0,5 s por imagem do lote na CPU).
    Rodadas que continuam um assistente já treinado precisam de menos passos.
    """
    steps = CONTINUE_STEPS if continuing else FIRST_STEPS
    steps_per_epoch = max(math.ceil(n_train / BATCH), 1)
    return min(max(math.ceil(steps / steps_per_epoch), MIN_AUTO_EPOCHS), MAX_AUTO_EPOCHS)


def train_assistant(
    item: str,
    epochs: Optional[int] = None,
    imgsz: int = DEFAULT_IMGSZ,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Treina (ou continua treinando) o assistente com as imagens anotadas do item.
    Parte dos pesos do assistente anterior, se existirem; senão do yolov8n.pt.
    epochs=None calcula automaticamente (ver auto_epochs). O início da rede fica congelado e não há
    aquecimento: é um ajuste fino rápido, que na CPU leva alguns minutos.
    """
    stats = ds.stats(item)
    if stats["annotated"] < MIN_ANNOTATED:
        raise ValueError(f"Anote ao menos {MIN_ANNOTATED} imagens antes de treinar o assistente "
                         f"(hoje: {stats['annotated']}).")

    split = ds.prepare_split(item, val_ratio=0.2)
    try:  # deixa 2 núcleos livres: a tela de anotação continua responsiva durante o treino
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    except Exception:
        pass
    out = model_path(item)
    base = out if os.path.isfile(out) else "yolov8n.pt"
    epochs = epochs or auto_epochs(split["n_train"], continuing=(base == out))
    if on_log:
        on_log(f"[ASSISTENTE] Treinando com {stats['annotated']} imagens anotadas, {epochs} épocas "
               f"(base: {'assistente anterior' if base == out else 'yolov8n.pt'}).")

    result = run_training(
        split["yaml"], out, model=base, imgsz=imgsz, epochs=epochs, batch=BATCH, patience=100,
        device="auto", on_progress=on_progress, on_log=on_log, should_stop=should_stop,
        train_kwargs={"freeze": 10, "warmup_epochs": 0, "cache": "ram"},
    )
    if result["path"]:
        _save_info(item, {"annotated": stats["annotated"], "imgsz": imgsz, "classes": stats["classes"]})
    return result


def install_external(item: str, source_pt: str) -> Dict:
    """
    Usa um modelo .pt já treinado (ex.: o best.pt do Colab, que treina na GPU) como assistente do item.
    Retorna {"matched": [...], "missing": [...]} com as classes do item que o modelo conhece ou não.
    As próximas rodadas locais continuam a partir dele.
    """
    from ultralytics import YOLO

    yolo = YOLO(source_pt)
    known = {str(n).lower() for n in yolo.names.values()}
    classes = ds.load_classes(item)
    matched = [c for c in classes if c.lower() in known]
    missing = [c for c in classes if c.lower() not in known]
    if not matched:
        raise ValueError(f"O modelo não conhece nenhuma das classes do item ({', '.join(classes) or 'nenhuma'}). "
                         f"Classes do modelo: {', '.join(str(n) for n in yolo.names.values())}.")
    os.makedirs(ds.dataset_dir(item), exist_ok=True)
    shutil.copy2(source_pt, model_path(item))
    train_args = (getattr(yolo, "ckpt", None) or {}).get("train_args", {}) or {}
    _save_info(item, {"annotated": ds.stats(item)["annotated"], "classes": classes, "imported": True,
                      "imgsz": int(train_args.get("imgsz") or DEFAULT_IMGSZ)})
    return {"matched": matched, "missing": missing}


class Assistant:
    """Carrega o modelo do assistente (sob demanda, em segundo plano) e sugere caixas."""

    def __init__(self):
        self._yolo = None
        self._loaded_path: Optional[str] = None
        self._loaded_mtime: Optional[float] = None
        self._imgsz = DEFAULT_IMGSZ
        self._loading = False
        self._lock = threading.Lock()

    def exists(self, item: str) -> bool:
        return bool(item) and os.path.isfile(model_path(item))

    def is_ready(self, item: str) -> bool:
        """Modelo carregado e atualizado em relação ao arquivo em disco."""
        path = model_path(item)
        return (self._yolo is not None and self._loaded_path == path and os.path.isfile(path)
                and self._loaded_mtime == os.path.getmtime(path))

    def load(self, item: str) -> bool:
        """Carrega (ou recarrega, se o arquivo mudou). Bloqueante: chame de uma thread."""
        path = model_path(item)
        if not os.path.isfile(path):
            return False
        with self._lock:
            if self.is_ready(item):
                return True
            from ultralytics import YOLO
            yolo = YOLO(path)
            train_args = (getattr(yolo, "ckpt", None) or {}).get("train_args", {}) or {}
            self._imgsz = int(train_args.get("imgsz") or load_info(item).get("imgsz") or DEFAULT_IMGSZ)
            self._yolo, self._loaded_path, self._loaded_mtime = yolo, path, os.path.getmtime(path)
        return True

    def load_async(self, item: str, on_done: Optional[Callable[[bool], None]] = None) -> None:
        """Carrega em segundo plano (importar o ultralytics na primeira vez leva dezenas de segundos)."""
        if self._loading:
            return
        self._loading = True

        def work():
            ok = False
            try:
                ok = self.load(item)
            except Exception:
                ok = False
            finally:
                self._loading = False
            if on_done:
                on_done(ok)

        threading.Thread(target=work, daemon=True).start()

    @property
    def loading(self) -> bool:
        return self._loading

    def suggest(self, item: str, image, classes: List[str], conf: float = 0.35) -> List[Suggestion]:
        """Caixas sugeridas para a imagem (BGR). Só inclui classes que existem na lista do item."""
        if not self.is_ready(item) or image is None:
            return []
        by_name = {c.lower(): i for i, c in enumerate(classes)}
        result = self._yolo(image, verbose=False, conf=conf, imgsz=self._imgsz)[0]
        found: List[Suggestion] = []
        for box in result.boxes:
            name = str(self._yolo.names[int(box.cls[0])]).lower()
            if name in by_name:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                found.append((by_name[name], x1, y1, x2, y2, float(box.conf[0])))
        return found
