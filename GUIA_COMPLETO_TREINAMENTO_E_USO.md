# Guia Completo: RS3 VisionBot Engine v2.0
**Do Dataset à Execução Autônoma**

Este documento detalha o ciclo de vida completo de configuração, treinamento da inteligência artificial e execução do bot. Siga as fases em ordem cronológica para garantir o sucesso operacional.

---

## FASE 1: Preparação Inicial

### 1.1. Pré-requisitos
Certifique-se de que o Python 3.10 ou superior está instalado.
1. Abra o terminal na pasta do projeto (`c:/Users/danil/Documents/BOTRUNESCAPE`).
2. Instale todas as bibliotecas necessárias rodando:
   ```bash
   pip install -r requirements.txt
   ```
   *(Nota: O projeto utiliza `ultralytics` para o YOLO, `opencv-python` para visão computacional, `mss` para captura de tela rápida, `transitions` para a FSM e `pyautogui/pydirectinput` para atuação).*

### 1.2. Configuração do RuneScape 3
* **Resolução e Layout:** A visão computacional Clássica (UI) depende da consistência de escala. Mantenha o layout do jogo o mais "limpo" possível. A resolução recomendada é jogar em Modo Janela (preferencialmente fixo) para garantir que a janela se chame `"RuneScape"`.
* **Interface:** Deixe a mochila (Inventory) em uma posição fixa e visível.

---

## FASE 2: Configuração de Interface (OpenCV 2D)

O bot precisa de imagens estáticas (Templates) para entender o estado da Interface de Usuário (Ex: Saber se a mochila está aberta ou se está cheia).

### Passo a passo para criar os templates:
1. Deixe o RuneScape 3 aberto no monitor principal.
2. No terminal, rode a ferramenta de setup:
   ```bash
   python tools/ferramenta_setup.py
   ```
3. O script aguardará 3 segundos para você posicionar o jogo, tirará um print da tela e abrirá uma janela.
4. **Clique e arraste** para selecionar as seguintes regiões (você terá que rodar o script e renomear o arquivo salvo para cada necessidade):
   * **Âncora da Mochila:** O ícone do topo da mochila ou aba do inventário. Salve como `assets/mochila_aberta_ancora.png`.
   * **Slot Vazio:** Selecione exatamente o contorno interno de um espaço livre da mochila. Salve como `assets/slot_vazio_mochila.png`.
   * **Indicador de XP / Mineração:** (Opcional, usado pelo validador) A barra de carregamento ou bolinha de XP sobre a cabeça do personagem.

---

## FASE 3: Captura de Dados para a IA (Mundo 3D)

O modelo YOLOv8 precisa aprender como as rochas (ex: Copper rock) se parecem no jogo.

1. **Grave um Vídeo ou Tire Prints:** Caminhe pelas áreas de mineração no jogo e tire vários prints (`PrintScreen` ou via script de captura) de diversos ângulos, iluminações e situações (pedra cheia, pedra esgotada, etc).
2. **Quantidade Mínima:** Tente juntar pelo menos de 50 a 100 imagens variadas. Quanto mais, melhor a IA fica.
3. Salve tudo em uma pasta chamada `dataset_raw` no seu computador.

---

## FASE 4: Anotação e Treinamento no Roboflow

O Roboflow é a plataforma que usaremos para dizer à IA o que é cada coisa nas imagens capturadas.

1. **Crie o Projeto:**
   * Acesse [Roboflow.com](https://roboflow.com/) e crie uma conta gratuita.
   * Crie um novo "Workspace" e clique em **Create New Project**.
   * Escolha **Object Detection**. Dê um nome ao projeto (ex: `RS3-Mining`) e defina o *Annotation Group* (ex: `Copper_rock`).

2. **Faça o Upload:**
   * Arraste as imagens da sua pasta `dataset_raw` para o Roboflow.
   * Clique em **Save and Continue**.

3. **Anotação (Bounding Boxes):**
   * Agora vem o trabalho manual: Vá em cada imagem, selecione a ferramenta de caixa (Bounding Box) e desenhe um quadrado envolta de todas as rochas de cobre na tela.
   * Nomeie a classe estritamente como **`Copper_rock`** (Atenção às maiúsculas e minúsculas, pois o código busca exatamente este nome).

4. **Gerar Versão do Dataset:**
   * Após anotar tudo, adicione as imagens ao *Dataset*.
   * Na tela de "Generate", você pode aplicar *Augmentations* (variações de brilho, leve desfoque, flips horizontais). Isso ajuda o modelo a não ficar "viciado".
   * Clique em **Generate**.

5. **Treinamento:**
   * Você pode treinar direto pelo site do Roboflow utilizando a infraestrutura deles (clique em **Train with Roboflow**) e escolhendo o modelo "YOLOv8 Fast".
   * *Alternativa Local:* Baixe o dataset (Export -> YOLOv8 -> Show Download Code) e treine localmente usando a CLI da ultralytics.

---

## FASE 5: Integração do Modelo Treinado

Assim que o treinamento finalizar no Roboflow:
1. Vá na aba de **Deploy / Export** do seu projeto no Roboflow.
2. Selecione a opção para fazer o download dos pesos do modelo, procurando pelo arquivo `.pt` (PyTorch).
3. Salve o arquivo baixado como **`best_rs3.pt`**.
4. Mova este arquivo para a pasta `models/` do projeto:
   ```
   c:/Users/danil/Documents/BOTRUNESCAPE/models/best_rs3.pt
   ```

---

## FASE 6: Execução do Bot

Com os templates de UI (`assets/`) criados e o "cérebro" de visão 3D (`models/best_rs3.pt`) no lugar, o bot está pronto para operar de forma autônoma.

1. Deixe o personagem perto das rochas.
2. Abra o terminal no projeto e execute:
   ```bash
   python main.py
   ```
3. O bot fará o seguinte ciclo:
   * **Checará a Interface:** Verificará se a mochila está aberta (abrirá apertando 'B' se não estiver).
   * **Checará o Inventário:** Se a mochila estiver com os slots cheios, ele ativará a pausa do motor de mineração (Estado: `INVENTARIO_CHEIO`).
   * **Procurar Alvo:** Invocará a IA YOLOv8 para escanear a tela em busca da classe `Copper_rock`.
   * **Movimentação:** Usará o atuador de mouse para deslizar suavemente de forma humana até a pedra e clicará.
   * **Validação:** Checará se você ainda está minerando. Assim que parar, ele buscará a próxima pedra.

4. **Para interromper o bot**, volte ao terminal e pressione `Ctrl + C`.

---

## FASE 7: Entendendo a Arquitetura Lógica (Para futuras edições)

O cérebro do bot é governado pelo arquivo `engine/state_mining.py`. Ele é uma FSM (Finite State Machine).
Caso queira adicionar novas funcionalidades (ex: Ir até o banco depositar os itens quando o inventário encher), siga esta lógica:

1. Adicione um novo estado na lista `STATES` no `state_mining.py` (Ex: `'INDO_PARA_BANCO'`).
2. Adicione as transições em `self.machine.add_transition(...)`.
3. Crie a função reativa `def on_enter_INDO_PARA_BANCO(self):` para ditar o que o mouse ou teclado fará neste estado.
4. Ajuste o fluxo de `update(self, frame)` para saber qual gatilho visual tira o bot de `INVENTARIO_CHEIO` e inicia o trajeto pro banco.
