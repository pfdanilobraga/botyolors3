"""
Registro das atividades criadas pelo Treinador em config/custom_activities.json.
O config/activities.py mescla esse arquivo, então o bot passa a listar o novo alvo sem editar código.
"""
import json
import os
from typing import Dict, List, Optional

CUSTOM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "custom_activities.json")


def load_custom() -> Dict:
    try:
        with open(CUSTOM_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def register_activity(skill: str, item: str, model: str, target: str,
                      xp_validator: str = "", drop_item: str = "",
                      priority_targets: Optional[List[str]] = None) -> None:
    """
    Cria/atualiza skill -> item. Caminhos relativos à raiz do projeto (ex.: models/mineracao/mini_iron.pt);
    xp_validator e drop_item são nomes de arquivos em assets/. priority_targets: classes que o bot clica
    antes do alvo normal (a primeira da lista tem mais prioridade), ex.: ["Mithril_buff"].
    """
    data = load_custom()
    entry = {"model": model.replace("\\", "/"), "target": target, "xp_validator": xp_validator}
    if drop_item:
        entry["drop_item"] = drop_item
    if priority_targets:
        entry["priority_targets"] = list(priority_targets)
    data.setdefault(skill, {})[item] = entry

    tmp = CUSTOM_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CUSTOM_PATH)
