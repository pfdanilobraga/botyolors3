"""
Guarda anti-ban: nenhuma ação do bot pode esperar um tempo FIXO. Qualquer `sleep(0.5)` com número literal nos módulos
que agem no jogo (engine, actuators, main) reprova o teste: o tempo tem que vir de antiban/humanize.py (ou de um sorteio).
"""
import glob
import re

SLEEP_WITH_LITERAL = re.compile(r"\bsleep\(\s*[0-9]")
FILES = sorted(glob.glob("engine/*.py") + glob.glob("actuators/*.py") + ["main.py"])


def test_no_action_waits_a_fixed_literal_time():
    offenders = []
    for path in FILES:
        for number, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
            code = line.split("#")[0]
            if SLEEP_WITH_LITERAL.search(code):
                offenders.append(f"{path}:{number}: {line.strip()}")
    assert not offenders, "esperas fixas encontradas (use antiban.humanize):\n" + "\n".join(offenders)


def test_no_module_uses_the_global_random_directly_for_actions():
    """Os sorteios passam por humanize.rng (semente controlável nos testes e ritmo da sessão)."""
    offenders = []
    for path in FILES:
        text = open(path, encoding="utf-8").read()
        for number, line in enumerate(text.splitlines(), 1):
            code = line.split("#")[0]
            if re.search(r"\brandom\.(uniform|randint|choice|random|gauss)\(", code):
                offenders.append(f"{path}:{number}: {line.strip()}")
    assert not offenders, "sorteios fora do humanize:\n" + "\n".join(offenders)
