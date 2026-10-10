"""Atalho: o coletor de imagens agora é a aba "Coletar" do Treinador (o mesmo que `python trainer_main.py`)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from gui.trainer_window import TAB_COLLECT, main

if __name__ == "__main__":
    main(start_tab=TAB_COLLECT)
