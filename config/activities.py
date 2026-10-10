# config/activities.py

ACTIVITIES = {
    "Mineração": {
        "Copper": {
            "model": "models/mineracao/mini_copper.pt",
            "target": "Copper_rock",
            "xp_validator": "indicadores/xp_mineracao.png",
            "drop_item": "copper_ore.png",  # template em assets/ do item dropado ao encher a mochila
        },
        "Iron": {
            "model": "models/mineracao/mini_iron.pt",
            "target": "Iron_rock",
            "xp_validator": "indicadores/xp_mineracao.png",
        },
    },
    "Pesca": {
        "Lobster": {
            "model": "models/pesca/fishing_model.pt",
            "target": "Fishing_spot",
            "xp_validator": "indicador_xp_pesca.png",
        }
    },
}


import copy as _copy
import json as _json
import os as _os

# Arquivo escrito pelo Treinador com os alvos que ele registrou
CUSTOM_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "custom_activities.json")

# Cópia dos alvos fixos deste arquivo: reload_activities() recomeça dela e mescla o JSON do Treinador de novo
_BASE_ACTIVITIES = _copy.deepcopy(ACTIVITIES)


def _merge_custom_activities() -> None:
    """Acrescenta/atualiza as atividades criadas pelo Treinador (config/custom_activities.json)."""
    try:
        with open(CUSTOM_PATH, encoding="utf-8") as f:
            custom = _json.load(f)
    except (OSError, ValueError):
        return
    for skill, items in custom.items():
        if isinstance(items, dict):
            ACTIVITIES.setdefault(skill, {}).update(items)


def custom_mtime() -> float:
    """Data de modificação do arquivo do Treinador (0 se ele não existe): serve para detectar alvos novos."""
    try:
        return _os.path.getmtime(CUSTOM_PATH)
    except OSError:
        return 0.0


def reload_activities() -> dict:
    """Relê os alvos (fixos + os do Treinador) SEM trocar o objeto ACTIVITIES, que outros módulos já importaram."""
    if _os.path.exists(CUSTOM_PATH):
        try:  # arquivo incompleto (sendo gravado agora)? mantém a lista atual em vez de perder os alvos
            with open(CUSTOM_PATH, encoding="utf-8") as f:
                _json.load(f)
        except (OSError, ValueError):
            return ACTIVITIES
    ACTIVITIES.clear()
    ACTIVITIES.update(_copy.deepcopy(_BASE_ACTIVITIES))
    _merge_custom_activities()
    return ACTIVITIES


_merge_custom_activities()
