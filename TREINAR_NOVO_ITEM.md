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
4. **Você não precisa marcar todas à mão — use o Assistente** (painel à direita):
   1. Marque ~10 a 30 imagens à mão.
   2. Clique em **Aprender com o que marquei**. É um treino rápido em segundo plano (alguns minutos na CPU; dá para continuar anotando enquanto roda).
   3. Daí em diante, cada imagem nova já vem com **caixas sugeridas (tracejadas, com “?” e a confiança)**. **Clique nas que estão corretas** para confirmá-las (ficam sólidas); botão direito remove uma caixa; arrastar desenha as que faltaram. Com **“Só valem as que eu confirmar”** ligado (padrão), as sugestões não confirmadas são **descartadas ao avançar** e, como a imagem passa a contar como anotada, viram “fundo” no treino: ensinam ao modelo o que **não** é o alvo. `C` confirma todas as sugestões da imagem (útil quando o assistente já acerta quase tudo); `R` remove as sugestões; se nenhuma estiver certa, marque **SEM objetos** (`X`).
   4. Com poucas imagens o assistente aponta os lugares certos mas com **confiança baixa** (por isso a “Confiança mínima” automática começa em 0,05 e sobe para 0,25 quando ele já aprendeu com 30+ imagens). Ele **não** é outra tecnologia: é um YOLO rápido, e o que falta para ficar seguro é **mais imagens variadas**.
   5. Use **“Ir para a mais DIFERENTE das anotadas”** para anotar primeiro as imagens de outras cenas/ângulos: ensinam mais do que mais uma imagem parecida com as que você já marcou.
   6. **Você não precisa clicar em treinar a cada rodada:** com **“Atualizar sozinho a cada [10] novas”** ligado (padrão), o assistente treina em segundo plano sempre que houver 10 imagens anotadas novas (e pela primeira vez ao chegar em 8). Cada rodada continua do que já foi aprendido e deixa 2 núcleos livres para a tela. O botão **Aprender com o que marquei** serve só para forçar uma rodada; **Parar** interrompe e o automático só volta quando houver mais imagens anotadas.
   7. **Tempo de cada rodada:** é calculado por *passos de treino*, não por épocas, então fica em torno de 3 min (1ª rodada ~6 min) em CPU comum, mesmo com o dataset crescendo (até algumas centenas de imagens).
   8. **Quer mais rápido e mais forte? Use a GPU do Colab como assistente:** aba **Treinar → Gerar pacote para o Colab** (veja o passo a passo na seção “Treinar” abaixo) e, na janela que abre, clique em **“Esperar o .pt → usar como ASSISTENTE”**. Quando o `.pt` for baixado, ele vira o assistente do item sozinho (as classes são casadas pelo nome) e as rodadas locais seguintes continuam a partir dele. Também dá para usar **“Usar modelo (.pt) do Colab como assistente…”** aqui no painel e escolher o arquivo à mão. Compensa a cada ~40–50 imagens novas.
   9. Alternativa: anotar no **Roboflow** (Label Assist) e treinar no Colab; o Treinador continua servindo para instalar o `.pt` e registrar o alvo.

### Várias classes e prioridade (ex.: `Mithril_rock` e `Mithril_buff`)
- O treino aprende **todas** as classes da lista do item; não é preciso treinar separado.
- Na aba **Treinar**, em **Classes prioritárias**, escreva as que o bot deve clicar primeiro (ex.: `Mithril_buff`; vírgula separa, a primeira vale mais). A **classe-alvo** continua sendo a rocha normal.
- No bot: ao escolher a próxima rocha, a prioritária vence mesmo com confiança menor; e, **minerando uma rocha comum, se uma prioritária aparecer na tela, o bot troca para ela** (a checagem roda a cada ~2,5 s). Já numa prioritária, ele não troca.

## 3. Treinar
- **Colab (recomendado, GPU grátis):** clique em **Gerar pacote para o Colab**. O Treinador cria, na pasta `dataset/<item>/`, o `.zip` do dataset e um **notebook pronto** (`treinar_<item>_colab.ipynb`) e abre uma janela com o passo a passo:
  1. **Abrir o Colab** (navegador, conta Google) → *Arquivo → Fazer upload de notebook* → escolha o `.ipynb` (o botão **Mostrar os arquivos** abre o Explorer já com o zip selecionado).
  2. *Ambiente de execução → Alterar tipo de ambiente → GPU (T4) → Salvar*, depois *Ambiente de execução → Executar tudo*.
  3. Na 3ª célula aparece o botão **Escolher arquivos**: selecione o `.zip`. (Se preferir não usar o notebook, a janela traz um roteiro manual com as células para colar.)
  4. Ao fim, o `<item>_best.pt` é baixado sozinho. Na janela do Treinador, use **Esperar o .pt → instalar como MODELO FINAL** (ou **→ usar como ASSISTENTE**) *antes ou durante* o treino: ele observa a pasta Downloads, detecta o `.pt` novo e instala/registra. A espera pode ser cancelada.
  - Não há como automatizar o Colab por completo (o Google não oferece API para isso): você faz o upload e o “Executar tudo”, o resto é detectado pelo Treinador.
  - Alternativa sem a espera: **Instalar .pt treinado no Colab…** e escolher o arquivo à mão (a janela abre na pasta Downloads).
- **Neste PC:** **Treinar agora**. Sem GPU leva horas (100 épocas a 512 px). Para testar o fluxo use 5 épocas e imgsz 320.
- Em ambos os casos o modelo vai para `models/<habilidade>/mini_<nome>.pt` e o alvo é registrado em `config/custom_activities.json`. Na janela do bot, o alvo novo aparece **sozinho ao voltar para ela** (ou clique em **↻ Atualizar alvos**): não precisa reiniciar. Modelo, ícone de drop e classes prioritárias são lidos quando você aperta **Start**.
- Parar o treino **não** instala nada (não sobrescreve um modelo bom); o último checkpoint fica em `runs/train/`.
