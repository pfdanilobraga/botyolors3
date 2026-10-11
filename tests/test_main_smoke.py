"""Fumaça do orquestrador: o main.py importa (com o ultralytics, que muda o cv2.imread) e tem todas as peças."""
import os

import pytest

REQUIRED_METHODS = [
    "start", "stop", "run", "_bot_loop", "_configure_session", "_configure_goals", "_configure_safety",
    "_handle_session_events", "_on_level_up_message", "_check_xp", "_safe_check", "_finish_session",
    "read_xp_now", "_read_xp_idle", "sync_progress", "_update_state_log",
]


def test_main_imports_and_has_every_controller_method():
    main = pytest.importorskip("main")
    missing = [name for name in REQUIRED_METHODS if not hasattr(main.BotController, name)]
    assert not missing, f"métodos que sumiram do BotController: {missing}"


def test_optional_assets_that_do_not_exist_yet_do_not_break_startup():
    """assets/logout_botao.png ainda pode não existir: o controlador sobe sem ele (com o ultralytics importado)."""
    pytest.importorskip("main")
    from engine.logout import LogoutAction
    action = LogoutAction(None, None, None, print, is_ingame=lambda frame: True,
                          template_path=os.path.join("assets", "arquivo_que_nao_existe.png"))
    assert not action.ready
