# config/activities.py

ACTIVITIES = {
    "Mineração": {
        "Copper": {
            "model": "models/mineracao/mini_copper.pt",
            "target": "Copper_rock",
            "xp_validator": "indicadores/xp_mineracao.png",
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
