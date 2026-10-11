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

### 6.1. Sessão & Metas (painel à direita da janela do bot)

* **Tempo de bot:** `h` e `min`. Ao acabar, o bot para sozinho e avisa. `0 h 0 min` = sem limite.
* **Meta (nível):** o nível que você quer atingir na habilidade escolhida. O bot mostra o nível, o XP atual, o XP ganho, quanto falta, o XP/hora, a previsão de chegada na meta e uma barra de progresso.
* **O XP é lido do jogo, não estimado.** Com **Ler o XP real na aba Skills** ligado, o bot abre a aba Skills (tecla **H**, configurável), passa o mouse no ícone da habilidade, lê o tooltip (*Current XP / Next level / Remainder*), fecha a aba com a mesma tecla (só se foi ele quem abriu; **nunca clica no X**, porque um clique perdido no mundo faria o personagem andar) e devolve o mouse para onde estava. Isso acontece no início da sessão e depois a cada N minutos (padrão 3), sempre entre as ações do bot (nunca no meio de um clique ou de um drop). Perto da meta ele releia antes do intervalo para não passar do ponto.
* **Como lê os números:** a fonte do tooltip é de pixels fixos, então cada dígito é reconhecido por comparação exata com `assets/skills/fonte_xp.json`. Os três números se conferem (*Current + Remainder = Next level*, e *Next level* tem que existir na tabela de XP do RuneScape); leitura que não fecha a conta é descartada. Dígitos que a biblioteca ainda não conhece são deduzidos por essa mesma conta e aprendidos sozinhos (ficam em `config/fonte_xp_aprendida.json`), então nas primeiras leituras alguns tooltips podem demorar a ser decifrados.
* **Mensagem de nível no chat:** quando aparece *"You've just advanced a ... level!"* no chat (canto inferior esquerdo da janela do jogo), o bot lê o XP na aba Skills na hora, sem esperar o intervalo, e então avisa a subida de nível/meta com o valor exato. A mensagem é reconhecida pela frase fixa (vale para qualquer habilidade); o número do nível não é lido do chat. Mensagens que já estavam no chat ao iniciar são ignoradas. Sem leitura do XP ligada, a mensagem só toca o som de subida de nível.
* **Posição da janela Skills:** ela pode estar em qualquer lugar da tela e com larguras diferentes. O bot a procura pelo título (logo + "SKILLS") e mede a largura pelo botão X a cada leitura, então as colunas de ícones se ajustam. Se a aba não for reconhecida, o log mostra a nota do casamento (mínimo 0,85) e uma foto da tela é salva em `debug/skills_nao_reconhecida_*.jpg`.
* **Chat em qualquer lugar:** a mensagem de nível é procurada na tela inteira (uma varredura por segundo), não só no canto inferior esquerdo.
* **Ler XP agora:** testa a leitura na hora (com o bot parado ou rodando) e preenche *Nível atual / XP atual* com o que leu. Use para conferir se a tecla e a posição da aba Skills estão certas no seu cliente.
* **Se a leitura falhar** (tecla errada, aba em outro lugar, escala de interface diferente), o bot continua minerando, tenta de novo e, depois de algumas falhas seguidas, desliga a leitura automática desta sessão (o painel avisa o motivo; **Ler XP agora** religa). *Nível atual / XP atual* à mão servem de ponto de partida nesse caso, e **Corrigir nível/XP à mão** ajusta o valor com o bot rodando.
* **Habilidades:** a posição de cada habilidade na aba Skills vem da ordem padrão do jogo (inglês ou português: Mineração, Pesca, Lenhador...). Habilidades "elite" (Invention) usam outra curva de XP e ficam sem leitura automática.
* **Ao atingir a meta (ou o tempo):** som, o botão do bot pisca na barra de tarefas e abre uma janela de aviso. Com **Parar o bot ao atingir a meta** marcado (padrão) o bot para; desmarcado, ele só avisa e continua. Cada subida de nível também toca um som curto e vai para o log.
* **Continuidade:** ao parar (manualmente, por meta ou por tempo), o último nível/XP lido e a meta ficam em `config/progress.json` e a tela já abre com eles na próxima vez. Depois de atingir a meta, defina uma nova meta maior antes de iniciar de novo.
* **Requisito do cliente:** interface na escala padrão (100%) e em inglês, com a aba Skills no layout padrão (3 colunas), como nas imagens em `assets/skills/`.

### 6.2. Câmera e diagnóstico

* **Girar a câmera (setas) se não vir rochas por N s** (padrão 10 s): se o YOLO não enxergar nenhuma rocha do alvo por esse tempo, o bot gira a câmera com as setas ← → em passos curtos (esperando o YOLO olhar a cena de novo entre um passo e outro) e, a cada volta inteira, inclina a câmera (↑, depois ↓) e gira de novo. Para assim que uma rocha aparece e só recomeça depois de outros N s sem ver nenhuma. Isso resolve rochas escondidas atrás de relevo ou fora do ângulo; **não** leva o personagem de volta se ele tiver andado para longe das rochas (nesse caso o log mostra "Ainda sem rochas há ...").
* **Log em arquivo:** tudo que aparece na caixa de log da janela também é gravado em `logs/bot_AAAA-MM-DD.log`, com horário. Se o personagem fizer algo inesperado (ex.: andar para longe), os horários de `[AÇÃO] Clicando em`, `[XP] Mouse ...` e `[CÂMERA] ...` mostram o que o bot fez em cada instante.

* **Barra de stamina e banners:** sobre o personagem aparecem duas barras (fio ciano em cima, laranja/azul embaixo) e, em golpes críticos, banners ciano grandes ("Critical swing!", "x4 damage!"). O leitor da barra descarta manchas ciano altas (mais de 6 px), então um banner não é mais lido como "barra em 88%". A leitura da barra some por alguns segundos enquanto o banner cobre a barra; nesse intervalo o bot segue sem leitura (a regra de reclique por tempo sem barra continua valendo).
* **Mouse depois do clique:** o bot afasta o cursor do personagem para não deixar o tooltip da rocha em cima da barra. A direção (qualquer ângulo, sempre diferente da anterior em pelo menos 50 graus) e a distância (180 a 420 px) são sorteadas a cada clique.

### 6.3. Segurança: bot perdido, jogo travado e logout

Se o personagem morrer (ou for teleportado, ou as rochas sumirem), o bot não tem o que minerar e ficaria procurando para sempre. O bloco **Parar se ficar perdido ou o jogo travar** (painel Sessão & Metas, ligado por padrão) evita isso:

* **Perdido:** se ficar **N min** (padrão 5) em `PROCURANDO_ALVO` sem ver nenhuma rocha, o bot **para**, toca um alarme, pisca na barra de tarefas e abre um aviso. A contagem começa quando as rochas somem (mesmo que o bot ainda estivesse minerando e a barra pedindo reclique) e continua enquanto a câmera gira; achar uma rocha zera. Se o seu alvo demora a reaparecer (rochas de respawn lento), aumente o tempo.
* **Jogo travado:** se a imagem do jogo ficar **idêntica por 1 minuto** (o mundo do jogo nunca fica 100% parado: o relógio do minimapa, a água e os personagens sempre se mexem), o bot para, toca um alarme rápido, abre o aviso e guarda a imagem em `debug/jogo_travado_*.jpg`. Não tenta deslogar, porque um cliente travado não responderia. O mesmo alarme vale se a janela do jogo for minimizada ou coberta por outra janela. A comparação custa ~10 ms por segundo.
* **Deslogar:** com **e deslogar do jogo** marcado, depois de parar o bot ele aperta **Esc** (abre a janela "OPTIONS MENU"), acha o botão **Logout** por template (`assets/logout_botao.png`, recortado do seu cliente) e clica nele. O menu pode abrir em qualquer lugar da tela; o botão é reconhecido pelo texto, então os vizinhos (Hop Worlds, Exit to Lobby) não são clicados. O jogo não pede confirmação. Se ele recusar (por exemplo, logo depois de um combate), o bot tenta de novo até 3 vezes e depois pede para você deslogar à mão. Se o seu menu tiver outra aparência (outro idioma ou escala), recorte o botão de novo com `python tools/ferramenta_setup.py` e salve como `assets/logout_botao.png`.

### 6.4. Anti-ban: nada de tempo fixo nem de movimento previsível

Regra do projeto: **nenhuma ação do bot espera um tempo fixo, nem se move do mesmo jeito duas vezes**. Tudo passa por `antiban/humanize.py` (e um teste, `tests/test_no_fixed_waits.py`, reprova qualquer `sleep` com número literal nos módulos que agem no jogo).

* **Ritmo da sessão:** cada Start sorteia um fator de ritmo (0,88 a 1,22) que acelera ou atrasa todas as esperas. Raramente (1,5% dos cliques) o bot faz uma pausa de 3 a 11 s, como alguém que se distraiu.
* **Tempos:** reação antes de cada clique (mediana ~0,3 s, com cauda), espera entre toques de drop, entre doses de poção, hesitações ocasionais; todos seguem distribuições assimétricas, não valores fixos.
* **Leitura do XP:** o "a cada ~3 min" é só a mediana: cada intervalo é sorteado entre ~0,55x e ~1,8x (1,6 a 5,4 min), a primeira leitura acontece entre 1,5 e 9 s depois do Start, e quando o chat avisa de um nível o bot leva alguns segundos para ler. Perto da meta ele ainda relê antes, para não passar do ponto.
* **Reclique:** o limite da barra varia ±8 pontos a cada reclique, o intervalo mínimo é sorteado (6 a 12 s), e há tempo de reação antes de clicar. Os limiares de "o XP parou" também são sorteados a cada rocha.
* **Cliques:** caem num ponto gaussiano **dentro da caixa** da rocha (não no centro exato), com ruído extra de poucos pixels. Entre rochas com confiança parecida, a escolhida é sorteada.
* **Mouse:** trajetória em curva cúbica diferente a cada vez, velocidade em sino (devagar no início e no fim), tremor suave e, em movimentos longos, uma ultrapassagem seguida de correção (cerca de 1 em cada 5). A duração cresce com a distância e varia. Depois de ler o XP o mouse volta para **perto** de onde estava (nunca ao pixel exato); ao passar sobre o ícone da habilidade, cai num ponto sorteado da célula; o estacionamento depois do clique sorteia direção e distância.
* **Teclado e câmera:** tempo de tecla pressionada variável; a busca com a câmera tem esperas, sentido, tamanho das voltas e inclinações sorteados (e às vezes inverte o sentido no meio).
* **Máscara de cristal:** renova entre 4:26 e 4:52 (e não sempre aos 4:50).

---

## FASE 7: Entendendo a Arquitetura Lógica (Para futuras edições)

O cérebro do bot é governado pelo arquivo `engine/state_mining.py`. Ele é uma FSM (Finite State Machine).
Caso queira adicionar novas funcionalidades (ex: Ir até o banco depositar os itens quando o inventário encher), siga esta lógica:

1. Adicione um novo estado na lista `STATES` no `state_mining.py` (Ex: `'INDO_PARA_BANCO'`).
2. Adicione as transições em `self.machine.add_transition(...)`.
3. Crie a função reativa `def on_enter_INDO_PARA_BANCO(self):` para ditar o que o mouse ou teclado fará neste estado.
4. Ajuste o fluxo de `update(self, frame)` para saber qual gatilho visual tira o bot de `INVENTARIO_CHEIO` e inicia o trajeto pro banco.
