# Documento de Contexto Mestre - v2.0

**Nome do Projeto:** RS3 VisionBot Engine  
**Objetivo:** Desenvolver um bot engine autônomo, 100% visual e indetectável para RuneScape 3, utilizando Machine Learning para percepção de mundo e OpenCV Clássico para percepção de interface, orquestrados por uma Máquina de Estados Finita (FSM).  
**Linguagem Principal:** Python 3.10+

---

## 1. Princípios Arquiteturais Inegociáveis

*   **Zero Injeção:** O bot NÃO pode ler, escrever ou interagir com a memória do cliente (Zero Client Hooking). Todo input é via hardware simulado.
*   **Modularidade Estrita:** Percepção (Visão), Tomada de Decisão (Logic/FSM) e Ações (Actuators) são camadas isoladas. A visão retorna dados brutos, a lógica decide o que fazer, o atuador executa movimentos.
*   **Anti-Ban First:** Comportamento humano simulado via Curvas de Bézier, ruído paramétrico em coordenadas e entropia em tempos de espera.
*   **Independência de Resolução:** O uso de âncoras visuais permite que o bot funcione mesmo se a janela for movida ou a resolução levemente alterada.

## 2. Stack Tecnológico Definido

*   **Percepção (Mundo 3D):** YOLO (via `ultralytics`) EXCLUSIVAMENTE para objetos dinâmicos no mundo de jogo (Minérios, NPCs, Árvores, Loot no chão).
*   **Percepção (Interface 2D):** OpenCV Clássico (`cv2`) com **Template Matching**, detecção de bordas (Canny) e análise de cores para elementos estáticos de UI (Mochila, Banco, Barras de Vida, Botões).
*   **Captura:** `mss` (Captura de tela ultrarrápida).
*   **Ação (Atuadores):** `PyAutoGUI` / `PyDirectInput` envelopados em classes customizadas com jitter humano.
*   **Cérebro (Lógica):** Máquina de Estados Finita (FSM) usando a biblioteca `transitions`.

## 3. Estrutura de Diretórios e Módulos

*   `/vision/`: 
    *   `detector.py`: Inferência YOLO para o mundo 3D.
    *   `inventory.py`: Lógica de UI que utiliza templates de `/assets/` para validar estado da mochila.
    *   `capture.py`: Gerenciamento de screenshots e regiões de interesse (ROI).
*   `/actuators/`: 
    *   `mouse.py`: Movimentação Bézier.
    *   `keyboard.py`: Pressionamento de teclas com hold-time humano.
*   `/engine/`: 
    *   `state_mining.py`: Loop de mineração.
    *   `state_combat.py`: Loop de combate.
*   `/assets/`: Coleção de imagens âncora (PNG) para Template Matching (ex: ícones, slots vazios, cabeçalhos de janelas).
*   `/antiban/`: Gerador de entropia e simulador de fadiga.
*   `main.py`: Orquestrador central.

## 4. Fluxo de Dados e Garantia de Estado

1.  **Captura:** O módulo `vision` tira o print da área ativa.
2.  **Garantia de Estado (UI):** Antes de qualquer leitura de interface (ex: Checar se a mochila está cheia), o Engine verifica se a interface está aberta via âncora visual.
    *   *Regra:* Se `is_open()` for False, o Engine comanda o `actuator` para abrir a interface (via atalho ou clique) antes de prosseguir.
3.  **Análise de Mundo:** O YOLO identifica coordenadas do alvo 3D.
4.  **Decisão:** O Engine (FSM) avalia o cenário (Ex: "Alvo encontrado E Inventário com espaço?").
5.  **Ação:** Comanda o `actuator` para interagir com o alvo.
6.  **Looping & Dormir:** O `antiban` define micropausas baseadas em frames para evitar detecção e fadiga de CPU.

## 5. Instrução Operacional para a IA

*   **Tipagem Estrita:** Sempre use Type Hints.
*   **Documentação:** Funções complexas exigem Docstrings detalhadas.
*   **Foco Modular:** Se o ajuste for em como o bot "vê" a mochila, altere apenas `vision/inventory.py`. Se for em como ele "reage" ao inventário cheio, altere `engine/state_mining.py`.
*   **Segurança de Thread:** Toda captura e inferência deve ser eficiente o suficiente para manter a responsividade da FSM.
