"""
Apoio ao treino no Google Colab: gera o notebook pronto e espera o modelo baixado aparecer na pasta Downloads.

O Colab não tem API para ser controlado por programa, então o fluxo é semi-automático: o Treinador gera o
.zip do dataset e um notebook que faz tudo (instalar, receber o zip, treinar e baixar o .pt); o usuário só
envia o notebook/zip e clica em "Executar tudo". Depois o Treinador detecta o .pt baixado e o instala.
"""
import json
import os
import time
from typing import Callable, Optional

COLAB_URL = "https://colab.research.google.com/"


def _code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": source.strip("\n").splitlines(keepends=True)}


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}


def build_notebook(item: str, imgsz: int = 512, epochs: int = 100, batch: int = 16) -> str:
    """Notebook (.ipynb em JSON) que treina o dataset do Treinador na GPU do Colab e baixa o best.pt."""
    cells = [
        _markdown(f"""
# Treinar `{item}` no Colab (feito pelo Treinador)
1. **Ambiente de execução → Alterar tipo de ambiente → GPU (T4) → Salvar.**
2. **Ambiente de execução → Executar tudo.**
3. Quando aparecer o botão **Escolher arquivos**, selecione o `.zip` gerado pelo Treinador.
4. No fim, o arquivo `{item}_best.pt` é baixado sozinho. Volte ao Treinador.
"""),
        _code("""
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NENHUMA - troque o tipo de ambiente para GPU e execute de novo")
!pip -q install ultralytics
"""),
        _code("""
import glob, os, shutil, zipfile
from google.colab import files

os.chdir("/content")
if not glob.glob("/content/*.zip"):
    files.upload()   # abre o seletor: escolha o .zip do Treinador
zips = sorted(glob.glob("/content/*.zip"), key=os.path.getmtime)
assert zips, "Nenhum .zip foi enviado."
print("Usando:", zips[-1])
shutil.rmtree("/content/ds", ignore_errors=True)
with zipfile.ZipFile(zips[-1]) as z:
    z.extractall("/content/ds")
print(open("/content/ds/data.yaml").read())
"""),
        _code(f"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="/content/ds/data.yaml", imgsz={imgsz}, epochs={epochs}, batch={batch},
                      project="/content/runs", name="treino", exist_ok=True, plots=False)
print("mAP50:", results.results_dict.get("metrics/mAP50(B)"))
"""),
        _code(f"""
import shutil
from google.colab import files

shutil.copy("/content/runs/treino/weights/best.pt", "/content/{item}_best.pt")
files.download("/content/{item}_best.pt")
"""),
    ]
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "cells": cells,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=1)


def downloads_dir() -> str:
    """Pasta Downloads do usuário (padrão do Windows)."""
    return os.path.join(os.path.expanduser("~"), "Downloads")


def wait_for_new_model(since: float, folder: str, should_stop: Callable[[], bool],
                       timeout: float = 3600.0, poll: float = 2.0) -> Optional[str]:
    """
    Espera surgir um .pt novo (criado ou modificado depois de `since`) em `folder`, com tamanho estável, e devolve o
    caminho. Arquivos em download (.crdownload/.part) são ignorados porque só o .pt final é considerado.
    Retorna None se `should_stop()` ficar verdadeiro ou se o tempo esgotar.
    """
    deadline = time.time() + timeout
    while time.time() < deadline and not should_stop():
        try:
            candidates = [os.path.join(folder, n) for n in os.listdir(folder) if n.lower().endswith(".pt")]
            # no Windows getctime é a data de CRIAÇÃO do arquivo: pega também cópias que preservam a data antiga
            fresh = [p for p in candidates if max(os.path.getmtime(p), os.path.getctime(p)) >= since]
        except OSError:
            fresh = []
        if fresh:
            path = max(fresh, key=os.path.getmtime)
            try:
                size = os.path.getsize(path)
                time.sleep(1.5)  # só aceita quando o tamanho parou de mudar (download terminou)
                if size > 0 and size == os.path.getsize(path):
                    return path
            except OSError:
                pass
        time.sleep(poll)
    return None
