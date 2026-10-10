"""
Treina um modelo YOLOv8 pela linha de comando e instala o .pt em models/ (a tela do Treinador faz o mesmo).

    python tools/train_model.py --data dataset/iron/data.yaml --out models/mineracao/mini_iron.pt

Receita usada nos modelos atuais: yolov8n, imgsz 512, 100 épocas, batch 16.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina um YOLOv8 e instala o .pt em models/.")
    parser.add_argument("--data", required=True, help="data.yaml do dataset")
    parser.add_argument("--out", required=True, help="destino do modelo, ex.: models/mineracao/mini_iron.pt")
    parser.add_argument("--model", default="yolov8n.pt", help="pesos base (padrão: yolov8n.pt)")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30, help="para cedo se não melhorar por N épocas")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' ou índice da GPU, ex.: 0")
    args = parser.parse_args()

    try:
        result = run_training(
            args.data, args.out, model=args.model, imgsz=args.imgsz, epochs=args.epochs,
            batch=args.batch, patience=args.patience, device=args.device, on_log=print,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERRO] {e}")
        return 1
    return 0 if result["path"] else 1


if __name__ == "__main__":
    sys.exit(main())
