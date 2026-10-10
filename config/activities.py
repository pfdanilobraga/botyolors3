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


def _merge_custom_activities() -> None:
    """Acrescenta/atualiza as atividades criadas pelo Treinador (config/custom_activities.json)."""
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_activities.json")
    try:
        with open(path, encoding="utf-8") as f:
            custom = json.load(f)
    except (OSError, ValueError):
        return
    for skill, items in custom.items():
        if isinstance(items, dict):
            ACTIVITIES.setdefault(skill, {}).update(items)


_merge_custom_activities()
