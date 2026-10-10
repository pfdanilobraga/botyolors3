"""Abre o Treinador (coletar imagens, anotar, treinar e registrar um novo alvo no bot)."""
import os
import sys

# Garante que caminhos relativos (assets/, models/, dataset/) funcionem de qualquer pasta
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from gui.trainer_window import main

if __name__ == "__main__":
    main()
