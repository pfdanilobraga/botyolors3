"""Log do bot em arquivo (logs/bot_AAAA-MM-DD.log): a caixa de texto da janela some ao fechar o bot."""
import datetime
import os
import threading

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_lock = threading.Lock()  # o log é escrito pela thread da tela e pela do bot


def append_log(message: str, log_dir: str = LOG_DIR, now: datetime.datetime = None) -> None:
    """Acrescenta '[HH:MM:SS] mensagem' ao arquivo do dia. Nunca levanta erro: log não pode derrubar o bot."""
    now = now or datetime.datetime.now()
    line = f"[{now:%H:%M:%S}] {message.strip()}\n"
    try:
        with _lock:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"bot_{now:%Y-%m-%d}.log"), "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass
