# Treinar um alvo novo (ex.: ferro) — pelo Treinador

Abra pelo botão **TREINADOR** na janela do bot, ou com `python trainer_main.py`.
Tudo acontece em 3 abas, sempre sobre um **item** (ex.: `iron`) escolhido no topo; cada item tem sua pasta `dataset/<item>/`.

## 1. Coletar
1. Digite o nome do item no topo (ex.: `iron`) e clique **Usar / Criar**.
2. Abra o RuneScape perto do objeto (rochas de ferro). O status da aba precisa dizer **"RuneScape visível"** — a janela do Treinador não pode cobrir o jogo (deixe-a ao lado ou atrás).
3. Capture com **F9** (uma imagem) ou **F10** (liga/desliga a auto-captura, 1 imagem a cada ~2 s). Funciona com o jogo em foco. A auto-captura ignora cenas repetidas.
4. Meta: **100 a 300 imagens**, variando ângulo, câmera, rocha cheia e esgotada, com e sem outras coisas na tela.
5. Para o **Auto-Drop**: com o minério na mochila (aberta), use **Recortar ícone do item…** e salve (ex.: `iron_ore.png`).

## 2. Anotar
1. Crie as classes (ex.: `Iron_rock`). Se quiser que o bot ignore rochas esgotadas, crie outra classe **cujo nome NÃO contenha o nome do alvo** (ex.: `Iron_depleted`; **não** `Iron_rockU`). O bot compara por trecho do nome, então `Iron_rockU` seria clicada como alvo.
2. Desenhe uma caixa em volta de **cada** objeto (arrastar com o mouse). Botão direito remove uma caixa. Teclas: `A/D` anterior/próxima, `1-9` classe, `Z` desfazer, `X` sem objetos.
3. Imagens sem nenhum objeto: marque **SEM objetos** (ajudam o modelo a não ver rochas onde não há).
4. Dica para ganhar tempo: depois de ~40 imagens, treine um modelo rápido e use **Sugerir caixas** para pré-anotar o resto; só corrija.

## 3. Treinar
- **Colab (recomendado, GPU grátis):** **Gerar pacote para o Colab (.zip)**, siga o roteiro que aparece, e ao terminar use **Instalar .pt treinado no Colab…**.
- **Neste PC:** **Treinar agora**. Sem GPU leva horas (100 épocas a 512 px). Para testar o fluxo use 5 épocas e imgsz 320.
- Em ambos os casos o modelo vai para `models/<habilidade>/mini_<nome>.pt` e o alvo é registrado em `config/custom_activities.json`. **Reinicie o bot** e ele aparece na lista de alvos.
- Parar o treino **não** instala nada (não sobrescreve um modelo bom); o último checkpoint fica em `runs/train/`.
