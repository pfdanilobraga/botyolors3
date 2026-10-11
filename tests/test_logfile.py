import datetime

from gui.logfile import append_log


def test_appends_timestamped_lines_to_the_daily_file(tmp_path):
    now = datetime.datetime(2026, 10, 10, 13, 59, 8)
    append_log("[XP] Mouse no ícone de Mineração: (1089, 191)", str(tmp_path), now)
    append_log("\n[ORQUESTRADOR] Parada solicitada.", str(tmp_path), now)
    text = (tmp_path / "bot_2026-10-10.log").read_text(encoding="utf-8")
    assert text == "[13:59:08] [XP] Mouse no ícone de Mineração: (1089, 191)\n[13:59:08] [ORQUESTRADOR] Parada solicitada.\n"


def test_never_raises_if_the_folder_cannot_be_written(tmp_path):
    blocker = tmp_path / "arquivo"
    blocker.write_text("x")
    append_log("qualquer coisa", str(blocker / "logs"))      # o 'diretório' é um arquivo: falha em silêncio
