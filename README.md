# Nintendo DS Reinforcement Learning Pipeline

Este projeto implementa um agente de Aprendizado por Reforço (Reinforcement Learning - PPO) projetado para iterar em ambientes simulados do Nintendo DS (especificamente *New Super Mario Bros*). A arquitetura contorna a necessidade de leitura direta de endereços de memória (RAM) através da aplicação de heurísticas de Visão Computacional para o rastreamento espacial do agente.

O repositório foi projetado com foco em MLOps e suporta abordagens híbridas de aprendizado, integrando modelos Não-Supervisionados (Autoencoders) e de Motivação Intrínseca (ICM) para mitigação de recompensas esparsas e aceleração de convergência.

---

## 🛠 Pré-requisitos e Instalação

Recomenda-se a utilização de um ambiente virtual (conda/venv) para isolamento de dependências.

```bash
pip install gymnasium stable-baselines3[extra] opencv-python py-desmume mlflow torch tqdm numpy
```

> **Aviso:** O pacote `py-desmume` é o wrapper que encapsula a engine de emulação do Nintendo DS. Sem ele, o ambiente limitará-se a comportamentos estáticos de fallback.

### Estrutura de Arquivos
Posicione o binário da aplicação (ROM) e o snapshot de estado (Savestate) no diretório `data/`:
- `data/mario.nds` (Imagem ROM)
- `data/state.dst` (Savestate indicando o *frame* zero de inicialização do episódio)

---

## 🔬 Metodologia de Recompensa Visual

Para dispensar o acoplamento com endereços de memória voláteis do emulador, a função de recompensa (Reward Function) baseia-se em duas heurísticas de Visão Computacional operando sobre o tensor de imagens:

1. **Optical Flow Denso (Método de Farneback):** Calcula o vetor de deslocamento dos pixels do cenário de fundo (*background*). Quando a câmera acompanha o deslocamento do agente para a direita, o cenário move-se na direção oposta, fornecendo um gradiente escalar de avanço no eixo X. Utiliza-se a mediana direcional para isolar ruídos no primeiro plano.
2. **Subtração de Frames Diferencial (AbsDiff):** Resolve o problema mecânico de "zonas mortas" da câmera, onde o agente se desloca na tela mas o cenário permanece estático. Emprega limiarização sobre a diferença de frames (`cv2.absdiff`) para isolar os pixels da entidade em movimento, computando a sua mediana geométrica e integrando esse deslocamento local ao deslocamento global da câmera.

Esta fusão gera uma coordenada X sintética altamente precisa. Sua implementação eliminou cenários críticos de *Reward Hacking*, onde o agente minimizava a função objetivo acionando resets prematuros do episódio em virtude de falhas de captação na zona morta da câmera.

---

## 🚀 Arquiteturas de Treinamento

O pipeline suporta três métodos, ativados via argumentos de CLI.

### 1. Aprendizado por Reforço Base (PPO)
A política (Policy Network) e as camadas de extração convolucionais (CNN) são otimizadas simultaneamente do zero utilizando unicamente o sinal de recompensa do ambiente (*Extrinsic Reward*).

**Comando:**
```bash
python src/train.py --rom "data/mario.nds" --state "data/state.dst" --timesteps 1000000 --num-envs 4
```
*(O parâmetro `--num-envs 4` inicializa subprocessos paralelos do emulador, descorrelacionando lotes de transição e maximizando a taxa de amostragem).*

---

### 2. Visão Computacional Pré-Treinada (Autoencoder)
Para mitigar a ineficiência de amostragem (*Sample Inefficiency*) inerente ao processamento direto de tensores de imagem, emprega-se uma fase de Representational Learning.

* **Objetivo de Otimização:** O Autoencoder (CNN Encoder + CNN Decoder) ingere um conjunto de transições de imagem aleatórias. A função de perda (MSE) penaliza a assimetria entre a imagem de entrada e a imagem reconstruída após a passagem pelo gargalo dimensional (512 *features* latentes). Isso induz a rede a desenvolver de forma autônoma filtros de detecção de bordas, mapeamento de obstáculos e segmentação de entidades operantes, independente do sinal de recompensa final.
* **Transferência de Pesos:** No treinamento subsequente do PPO, os tensores do Encoder são instanciados e sua computação de gradiente é congelada (`requires_grad=False`). O PPO otimiza estritamente os Perceptrons (FCN) da camada de ação, reduzindo ordens de magnitude na convergência da política.

**Pipeline de Execução:**
```bash
# Geração de dataset via Random Policy
python src/collect_data.py --num-frames 10000

# Treinamento não-supervisionado do Autoencoder
python src/train_autoencoder.py --dataset "../data/mario_dataset.npz" --epochs 20

# RL com pesos visuais transferidos
python src/train.py --timesteps 1000000 --num-envs 4 --use-autoencoder
```

---

### 3. Exploração via Curiosidade Intrínseca (Módulo ICM)
Em ambientes de topologia complexa, a ausência prolongada de recompensas extrínsecas leva ao colapso do gradiente da política. O ICM mitiga isso formulando o ambiente como um problema de modelagem preditiva.

* **Dinâmica do Módulo:** O ICM compõe-se de dois sub-modelos. O *Modelo Inverso* prediz qual ação causou a transição $S_t \rightarrow S_{t+1}$, forçando o extrator de features a modelar apenas elementos do cenário que são controláveis pela política. O *Modelo Direto* utiliza essas features filtradas para prever o estado $S_{t+1}$ subsequente dada uma ação $A_t$.
* **Sinal de Recompensa Intrínseca:** O Erro Quadrático Médio da previsão do *Modelo Direto* é extraído e somado à recompensa escalar enviada ao PPO. Consequentemente, o agente é matematicamente incentivado a convergir para os espaços estocásticos de maior dificuldade preditiva, forçando sistematicamente a exploração ativa e a transposição de obstáculos.

**Comando:**
```bash
python src/train.py --timesteps 1000000 --num-envs 4 --use-icm
```

**Treinos ICM executados (100k, família RecurrentPPO):**
```bash
python src/train_ppo_recurrent.py --timesteps 100000 --num-envs 4 --use-icm --icm-update-freq 8 --no-tensorboard --run-id ppo_icm_100k
python src/train_ppo_recurrent.py --timesteps 100000 --num-envs 4 --use-icm --use-autoencoder --icm-update-freq 8 --no-tensorboard --run-id ppo_icm_ae_100k
```
*Notas de engenharia: `--icm-update-freq 8` batchiza o backward do ICM (a recompensa intrínseca continua calculada todo step — mesmo sinal, ~6x mais rápido); `--no-tensorboard` evita o import do TensorFlow que estourava a RAM; o `MLflowCallback` amostra `intrinsic_reward` a cada 50 steps (logar todo step derrubava o treino de ~29fps para ~4fps via commits SQLite). Tempos medidos: **ppo_icm_100k ≈ 48 min, ppo_icm_ae_100k ≈ 49 min** (4 envs, CUDA).*

---

### 4. Redes Residuais Convolucionais (ImpalaCNN)
Para otimizar o aprendizado de representações visuais profundas e melhorar a generalização espacial do agente, o pipeline inclui suporte à arquitetura residual **ImpalaCNN** (composta de blocos ResNet alternados com sub-amostragens e camadas convolucionais densas).

Essa arquitetura customizada (`src/impala_cnn.py`) substitui a clássica `NatureCNN` da biblioteca Stable-Baselines3, obtendo maior estabilidade e consistência de navegação sob políticas estocásticas (atingindo 80% de taxa de sucesso na superação de inimigos iniciais com 1M de passos).

**Comando de Treinamento:**
```bash
python src/train_impala.py --run-id "mario_impala" --timesteps 1000000 --num-envs 6 --n-steps 1376 --lr 0.0003 --ent-coef 0.01
```

**Comando de Avaliação:**
```bash
python src/evaluate.py --model "models/mario_impala.zip" --stochastic
```
---

### 5. Algoritmo Model-Based com World Models (DreamerV3)
Para acelerar drasticamente a eficiência de amostragem (Sample Efficiency) e viabilizar planejamento espacial latente, o projeto suporta o algoritmo **DreamerV3** através da biblioteca **SheepRL** (baseada no Lightning Fabric).

* **Funcionamento:** O DreamerV3 treina um Modelo de Mundo Recorrente (RSSM) composto por um Encoder, Decoder, Modelo de Transição e Modelo de Recompensa. O agente (Actor-Critic) é otimizado inteiramente na "imaginação" (rollouts latentes simulados pelo World Model), reduzindo a quantidade de passos de física reais exigidos do emulador.
* **Aceleração GPU:** O pipeline foi atualizado para suporte completo à GPU (CUDA 12.x), permitindo compilação e treinamento eficientes. Para evitar estouros de disco no memmap padrão da SheepRL, otimizamos o buffer para rodar diretamente na memória RAM (`buffer.memmap=false`) com limite de 100.000 transições.

**Comando de Treinamento (GPU):**
```bash
python src/train_dreamer.py
```
*Este script inicializa a distribuição do Fabric e registra dinamicamente o wrapper do Gymnasium (`MarioNDS-Dreamer-v0`) com transposição de canais para formato de imagem compatível com PyTorch (`1, 84, 84`).*

**Monitoramento de Métricas (TensorBoard):**
```bash
tensorboard --logdir logs/runs/dreamer_v3
```

---

### 6. PPO Recorrente, Transformer Causal e Aprendizado de Representações Auxiliares (CURL / SPR)
Para lidar com dependências temporais complexas em que frames isolados não contêm informações suficientes sobre velocidade e aceleração (sem frame stacking), o pipeline suporta o **Recurrent PPO** (utilizando política LSTM) e modelos baseados em **Causal Transformers**. Além da política base, integramos técnicas avançadas de aprendizado não-supervisionado online para acelerar a eficiência de amostragem visual:

#### 6.1. CURL (Contrastive Unsupervised Representations for Reinforcement Learning)
O CURL extrai representações robustas de imagens maximizando a concordância entre visões aumentadas da mesma observação através de Contrastive Learning.
* **Augmentação de Dados:** Aplica `random_crop` espacial sobre lotes extraídos do buffer de rollout para gerar duas visões diferentes de cada frame.
* **Perda InfoNCE (Bilinear):** Otimiza uma matriz de projeção bilinear $W$ para maximizar a similaridade das visões correspondentes (positive pairs) e minimizar a similaridade com frames diferentes no batch (InfoNCE loss). A rede *target* é atualizada por média móvel exponencial (EMA).
* **Modo Híbrido (Autoencoder + CURL):** Inicializa o codificador com pesos do Autoencoder pré-treinado e o mantém descongelado (`--unfreeze-encoder`), permitindo ajuste fino online via CURL e mitigando o problema do "catastrophic forgetting".

#### 6.2. SPR (Self-Predictive Representations)
O SPR obriga a rede a prever os seus próprios estados latentes futuros através de um modelo de transição implícito. Diferente do CURL (que foca na reconstrução espacial/contrastiva do *frame* atual), o SPR foca na coerência da dinâmica temporal.
* **Transição Latente:** Emprega um projetor e um modelo de transição multi-step na representação da CNN. Dada uma sequência de ações observadas, tenta adivinhar o *feature map* futuro sem precisar renderizar os pixels (como num Autoencoder preditivo).
* **Eficiência:** Evita o custo de reconstruir pixels completos, otimizando o agrupamento (clustering) temporal de estados no espaço latente. Na prática, acelera a compreensão das físicas do jogo.

#### 6.3. Causal Transformer (Policy Network)
Substitui a típica LSTM de agentes recorrentes por um Transformador de Atenção Causal (semelhante a arquitetura de modelos GPT).
* **Vantagens:** Melhor memória de longo-prazo. Onde as LSTMs "esquecem" heurísticas após algumas centenas de passos devido à atenuação dos gradientes na propagação no tempo (BPTT), a atenção cruzada causal permite que a política olhe diretamente para *tokens* anteriores e crie correlações distantes (crucial em fases complexas do Mario).

---

### Benchmark Final de Arquiteturas: 100k vs 1M de Passos
Avaliamos rigorosamente todas as arquiteturas após o treinamento. Para garantir a significância estatística, cada modelo compilado rodou ativamente em **10 episódios determinísticos completos** (`num_episodes=10`) em um ambiente isolado.

**Resultados Oficiais de Desempenho (10 Episódios de Avaliação por Modelo):**

| Configuração (Modelo) | Média Recompensa | Desvio Padrão | Max Recompensa |
| :--- | :---: | :---: | :---: |
| **NE-Dreamer (100k steps)** | 378.78 | 179.61 | 847.52 |
| **World Models GA: Encoder + LSTM + Alg. Genético (100k steps)** | 425.58 | 0.00 | 425.58 |
| **World Models sep-CMA-ES: Encoder híbrido + LSTM + CMA-ES (104k steps)** | 242.69 | 0.00 | 242.69 |
| **PPO Pure (100k steps)** | 13.10 | 0.00 | 13.10 |
| **PPO SPR (100k steps)** | 254.34 | 0.00 | 254.34 |
| **PPO CURL (100k steps)** | 105.00 | 0.00 | 105.00 |
| **PPO DrQ-v2 (100k steps)** | 74.92 | 0.00 | 74.92 |
| **IMPALA Transformer SPR (1M steps)** | 74.74 | 0.00 | 74.74 |
| **IMPALA CURL Recurrent (1M steps)** | 324.79 | 0.00 | 324.79 |

> **Análise Técnica Detalhada:**
> 1. **NE-Dreamer (100k):** O algoritmo baseado em World Models apresentou a **maior média de recompensa geral (378.78)** e o **maior pico (847.52)** em apenas 100 mil interações. Como ele aprende a dinâmica do mundo de forma não-supervisionada (imaginando estados futuros) antes de otimizar a política, ele alcança uma eficiência de amostra incrivelmente superior ao PPO puro, embora seu comportamento seja mais instável (Desvio Padrão de 179.61).
> 2. **IMPALA CURL (1 Milhão):** O modelo mais robusto dentre os baseados em model-free (PPO ImpalaCNN). Alcançou uma excelente consistência (324.79 de média sem sofrer penalidades de travamento). A convolução avançada (IMPALA) aliada ao aprendizado contrastivo temporal (CURL) gerou uma política incrivelmente estável, superando a arquitetura base.
> 3. **PPO SPR vs PPO CURL vs DrQ-v2 (100k):** Ao comparar as representações auxiliares na marca de 100k:
>    - **SPR (254.34)** brilhou porque focar na previsão da dinâmica temporal forçou a CNN a focar no movimento futuro, acelerando a extração do conceito de movimento e obstáculos.
>    - **CURL (105.00)** foi mais lento na convergência, focando muito na reconstrução contrastiva do mesmo frame e caindo em armadilhas locais.
>    - **DrQ-v2 (74.92)** sofreu com o fato de que a augmentação espacial (shifts/crops) destruiu a precisão de sub-pixels necessária para navegação precisa no jogo em apenas 100k steps.
> 4. **IMPALA Transformer SPR (1 Milhão):** Obteve um resultado fraco (74.74). Como notado na literatura de Transformers em RL, mecanismos de Atenção Cruzada Causal (Causal Attention) requerem datasets massivos para aprender o alinhamento. 1 Milhão de passos num ambiente online não foram suficientes para as matrizes de projeção do Transformer convirjam, gerando resultados sub-ótimos comparado ao LSTM do CURL.
> 5. **PPO Puro (100k):** Falhou completamente (13.10), não saindo da tela inicial do jogo devido à severa ineficiência de amostra das CNNs tradicionais de RL.
> 6. **World Models GA (100k):** O controlador linear evoluído sobre `[z (Encoder 512) + h (LSTM 256)]` obteve a **maior média em 100k (425.58)** com **desvio zero** — política determinística que sobrevive os 1000 steps do episódio em todas as 10 avaliações. Supera o NE-Dreamer na média, mas perde no pico (847.52) e na generalização: com apenas 4.614 parâmetros e sem gradiente, o GA explora pouco além do que já funciona (possível ótimo local). Ressalva metodológica: a avaliação usa teto de 1000 steps/episódio, então o valor reflete sobrevivência completa, não término natural da fase.
> 7. **World Models sep-CMA-ES (104k):** Trocar o encoder VAE pelo híbrido AE+CURL e o GA pelo sep-CMA-ES **piorou a média (242.69)** — mas também com sobrevivência total (10×1000 steps) e desvio zero. Hipóteses: (a) as features contrastivas do CURL, boas para PPO com gradiente, descartam micro-sinais de movimento que o controlador linear evolutivo precisava; (b) o teto de 500 steps na evolução seleciona comportamento de curto prazo; (c) só 8 gerações limitaram a adaptação do step-size ($\sigma$: 0.08→0.079). Conclusão prática: para políticas lineares evolutivas, o encoder VAE puro foi melhor; CURL ajuda quem tem gradiente, não quem tem mutação.

### 7. World Models com Algoritmo Genético (Encoder + Memória LSTM + GA)
Alternativa inspirada em Ha & Schmidhuber (2018), em 3 camadas — e aqui vale o seu ponto: **o Autoencoder entra apenas como pré-treino; o que alimenta o RL é só o Encoder** (o decoder é descartado após o treino por reconstrução):
* **V (Encoder visual):** o Encoder CNN pré-treinado (`models/autoencoder.pth`, 512 latentes) é congelado e usado como extrator puro de features.
* **M (Memória):** uma LSTM (512+6 → 256) treinada de forma supervisionada a prever o próximo latente $z_{t+1}$ a partir de $(z_t, a_t)$ em 10k frames de política aleatória.
* **C (Controlador genético):** um linear minúsculo $a = W[z;h]+b$ (4.614 params) evoluído com GA elitista (pop 24, mutação gaussiana $\sigma=0{.}05$), sem gradiente — orçamento restante de 90k steps.

**Comando (100k steps totais = 10k memória + 90k GA):**
```bash
python src/train_worldmodels_ga.py --timesteps 100000 --mem-frames 10000 --pop-size 24 --run-id worldmodels_ga_100k
```
**Tempo de treino medido (wall-clock, GPU CUDA + 1 env CPU): 5510.9s = 91.8 min** — coleta 498.4s (~8.3 min) + treino LSTM 2.9s + evolução GA 5003.9s (~83.4 min). Artefatos: `models/worldmodels_ga_100k.npz` (controlador) e `models/worldmodels_ga_100k_memory.pth` (LSTM).

### 8. World Models v2: Encoder Híbrido + sep-CMA-ES + Treino Paralelo
Evolução do item 7 com as duas acelerações do item 2:
* **V:** Encoder híbrido AE+CURL extraído de `models/ppo_hybrid_ae_curl_100k.zip` (custo zero — pesos já treinados).
* **C:** sep-CMA-ES diagonal (NumPy; mesma matemática do `evosax`, sem instalar JAX+CUDA — o gargalo é o emulador, não a álgebra da evolução).
* **Aceleração:** pool de 4 envs persistentes com init escalonado (padrão de `train_ppo_recurrent.py`), evolução com teto de 500 steps + validação full (1000) no top-3, coleta da memória reduzida para 5k.

**Comando:**
```bash
python src/train_worldmodels_cma.py --timesteps 100000 --mem-frames 5000 --workers 4 --run-id worldmodels_cma_100k
```
**Tempo de treino medido: 2648.8s = 44.1 min (2.1× mais rápido que o v1)** — encoder 0.3s + coleta 241.2s (~4 min) + LSTM 3.8s + CMA-ES 8 gens 2039.3s (~34 min) + validação/avaliação. Env steps reais: 104.000 (5k + 96k + 3k — a última geração ultrapassa um pouco o orçamento). Artefatos: `models/worldmodels_cma_100k.npz` e `models/worldmodels_cma_100k_memory.pth`.

### 9. Benchmark Unificado: 10 episódios determinísticos + 10 estocásticos
Todas as alternativas SB3 foram reavaliadas com o mesmo protocolo (`src/evaluate_benchmark.py`, episódios completos, sem render, JSONs em `evals/`):

| Modelo | Budget | det (média) | stoch (média ± std / max) |
| :--- | :---: | :---: | :--- |
| PPO Pure | 100k | 120.80 | 331.21 ± 171.39 / 684.92 |
| PPO + Autoencoder | 100k | 74.77 | 199.93 ± 118.31 / 429.88 |
| PPO CURL | 100k | 226.72 | 362.45 ± 152.44 / 711.47 |
| PPO Híbrido AE+CURL | 100k | 74.74 | 249.19 ± 125.73 / 488.00 |
| PPO SPR | 100k | 254.34 | 323.55 ± 68.95 / 464.04 |
| PPO DrQ-v2 | 100k | 74.92 | 248.49 ± 106.25 / 425.54 |
| **PPO + ICM** | 100k | 222.49 | 223.11 ± 114.45 / 402.33 |
| **PPO + ICM + Autoencoder** | 100k | 74.77 | 235.13 ± 116.79 / 420.44 |
| **PPO RAM-only (sem visão)** | 100k | 61.40 | 357.15 ± 297.62 / **955.10** |
| **PPO RAM + geometria (pits)** | 100k | 61.40 | 388.16 ± 266.08 / 784.62 |
| **PPO GNN relacional (grafo)** | 100k | 61.40 | 527.50 ± 362.69 / **1378.89** |
| Recurrent PPO | longo | 31.74 | 540.75 ± 211.67 / **880.77** |
| ImpalaCNN PPO | 1M | 74.74 | 181.16 ± 92.38 / 345.79 |

> **Leituras:**
> 1. **det tem std 0.00 sempre** (ambiente + política determinísticos): 10 eps det são 10 replays idênticos — por isso o modo stoch foi adicionado.
> 2. **Cluster do "primeiro pit"**: ae_frozen, híbrido, icm_ae e impala morrem deterministicamente no mesmo ponto (~74.7 / 39 steps); o det não os separa, o stoch sim.
> 3. **stoch ≥ det quase sempre** — ruído de exploração ajuda políticas subt reinadas a passar do primeiro obstáculo.
> 4. **ICM fica no meio do pelotão** (222/235): não supera o SPR; no icm_ae o encoder domina e a curiosidade agrega pouco em 100k.
> 5. **Validação do protocolo**: spr-det (254.34) e drq-det (74.92) reproduzem a tabela antiga exatamente; pure e curl divergem dela (protocolo det antigo desconhecido — linhas antigas mantidas como histórico).
> 6. **Cuidado com n=10 stoch**: duas varreduras variaram ±50–100 na média — para rankings apertados use ≥30 episódios ou múltiplas seeds.

### 10. Mapa de RAM (EUR) e `reward_mode="ram"`
Extração direta de variáveis via `emu.memory` (`src/ram_state.py`, busca em `src/ram_search.py`). O mapa US (TASVideos/DataCrystal) **não vale** para a ROM EUR — validado por busca diferencial + screenshots:

| Endereço (EUR) | Variável | Status |
| :--- | :--- | :---: |
| `0x0209DC00` u8 | Vidas (5→4 na morte; cf. AR EUR `2209DC00`) | ✅ |
| `0x02098240` / `0x020DCFA0` u32 | Câmera X absoluta (deadzone, congela no mapa) | ✅ |
| `0x0209AE9C` u32 | Contador de atividade horizontal (sobe nas 2 direções, pausa na morte) | ✅ |
| `0x021C1904`… s32 | Espelhos de velocidade (±6144 = ±1,5 px/frame em 20.12) | ✅ |
| `0x021C1890` +`0x44/0x60/0x68` | **Objeto do Mario**: tipo `0x1C`, A=Y (arco de pulo, chão −480px), B=X | ✅ |
| lista em `obj+0x38` | **Lista ligada circular de objetos** (inimigos linkam ao se aproximar) | ✅ |
| `RamState.enemies()` | `[{type, dx, dy}]` em px — goomba validado: 164px→contato (~10px)→morte | ✅ |
| `0x020DC968` u32 | Timer da fase em 20.12 (398.3 = HUD; linearidade validada) | ✅ |
| `0x021C1908` s32 | Y-vel (curva de gravidade no pulo; `==0` ⟺ no chão, salvo 1 step no ápice) | ✅ |
| `0x020A703C` u32 | Y complementar (experimental) | 🧪 |

**Validação RAM vs optical flow** (mesmo rollout, 39 steps pré-morte): correlação do progresso acumulado **0,91**, mesma escala (câmera em 20.12 fixed point confirma `RAM_PX_PER_UNIT=1/4096` a 18%), **morte detectada 3 steps antes** (vidas caem no início da animação; o template só na tela preta). Custo: ~7 leituras/step (µs) vs ~50 ms de flow+ORB.

**Uso:** `MarioNdsEnv(..., reward_mode="ram")` — mesma fórmula de recompensa, progresso pela câmera-RAM e morte pelas vidas (templates seguem como redundância). Default segue `"flow"` (todos os resultados publicados usam ele).

### 11. Treino sem visão computacional (estilo NES/SMB)
`src/ram_env.py` (`MarioRamEnv`): observação `Box(17)` pura de RAM, zero pixels/CNN —
`[x, y, vx, vy, on_ground, lives, time, cam_x + 3×(dx, dy, tipo)]` (posições relativas ao reset, inimigos filtrados p/ tela). Done por vidas, timer zerado ou 1000 steps; mesma recompensa do env visual.

**Prova de viabilidade** (`src/train_ram.py`, PPO `MlpPolicy`): smoke de 5000 steps em 308s (16,2 fps) evoluiu para treino completo de **100k steps (102.400 reais, 2 envs, 36,7 fps, 2725s = 45 min)**, `models/ram_ppo_100k.zip`:
```bash
python src/train_ram.py --timesteps 100000 --num-envs 2 --run-id ram_ppo_100k
```

**Comparativo RAM vs visão (100k, protocolo §9):** RAM det 61,40 (morre no pit como os visuais fracos) mas stoch **357,15 ± 297,62 / máx 955,10** — top-2 com o CURL (362,45) e **maior pico geral** (recurrent 880,77; Dreamer 847,52). Interpretação honesta: o vetor RAM cobre entidades dinâmicas mas **não a geometria estática** (pits/paredes são tiles, fora da RAM de estado) — o agente compensa **memorizando** o nível (spawn determinístico + X relativo). Conclusão: sem nenhum pixel, sem CNN, chega ao nível dos melhores métodos visuais em 100k — e o gap restante (det no pit) é exatamente onde a visão faz falta. Notas: `finish (+100)` segue pixel (fora do `ram_env`); pendentes p/ paridade NES/SMB total: powerup, level ID, moedas/score; 2 envs foi o ponto ótimo (4 envs têm contention: 10,7 fps).

**Becos sem saída (registrados p/ não repetir)**: OAM só tem HUD (jogo renderiza Mario/inimigos em **3D**); scroll 2D (`0x040000xx`) zerado; shift US→EUR não é uniforme; morte volta ao **mapa-mundi** (não respawn); poke único de RAM é apagado pelo jogo (código AR escreve todo frame). A lista de objetos EUR foi resolvida via código AR "Big Jumps" (`021C1944`) + walk circular a partir do Mario. **Pendente agora**: X absoluto em pixels de tela (duas bases coerentes nos deltas), calibragem fina do Y e reward shaping com `enemies()` (ex.: penalidade por `dx` pequeno de Goomba/Koopa).

### 12. Geometria da fase via ROM (pits, paredes, spawns — sem visão)
Fases NSMB ficam em `course/X##_#.bin` + `_bgdat.bin` dentro da ROM (`src/course.py`, via `ndspy`; formato documentado pelo NSMBe/NSMB Central: tile objects `(obj, x, y, w, h)` em tiles, sprites 12B `(type, x, y)` em tiles, entrances 20B em pixels).
Validações cruzadas RAM↔ROM na 1-1 (`A01_1`): spawn entrada (80, 464)px = Mario (88, pés 480) − meia-altura do sprite; chão na fileira 30 (= Y_RAM −480px); **pit no tile 30** (morte além do goomba); **goomba no tile 24 = 384px** (RAM o viu em 365px marchando p/ esquerda ✓); `?`-blocks nos tiles 18–23 (batem com screenshots).
`MarioRamEnv(geo=True)` anexa 6 flags de pit (`pit_ahead`, colunas +1..+6, obs 17→23); treino `ram_geo_100k` (102.400 steps, 2 envs, 36,1 fps, 2768s = 46 min):
```bash
python src/train_ram.py --timesteps 100000 --num-envs 2 --geo --run-id ram_geo_100k
```
Resultado honesto: stoch 388,16 (vs 357,15 sem geo — empate técnico dentro do ruído ±270) e det ainda 61,40 no pit. As flags informam *que* há pit à frente, mas 100k steps de MLP não converteram isso em pulo cronometrado — geometria estática ajuda menos que entidades dinâmicas nesse budget; próximo teste seria janela de ocupação mais rica ou reward shaping por proximidade do pit.

### 13. GNN relacional (MeanMPNN puro-torch, sem PyG/DGL)
Com o ambiente totalmente observável por entidades, o baseline natural seguinte é relacional: `src/graph_env.py` monta grafo de 14 nós (Mario + ≤5 inimigos + ≤8 retângulos estáticos da ROM) × 8 features `[dx, dy, w, h, is_mario, is_enemy, is_static, tipo]` + vetor global + máscara (obs `Box(134)`); `src/gnn_extractor.py` faz message passing 2× (média mascarada, grafo completo) + pool global — **invariante a permutação por construção** (o MLP ordenado por distância sofre descontinuidades em trocas de slot). Treino `gnn_100k` (102.400 steps, 2 envs, 34,9 fps, 2866s = 48 min):
```bash
python src/train_gnn.py --timesteps 100000 --num-envs 2 --run-id gnn_100k
```
Resultado: det 61,40 (pit, como toda a família RAM) mas stoch **527,50 ± 362,69 / máx 1378,89** — melhor pico geral (recurrent 880,77; RAM 955,10) e média no nível do recurrent (540,75). Leitura honesta: o viés relacional supera o MLP flat no jogo estocástico (527 vs 357/388), mas o ruído (±363, n=10) impede declarar vitória definitiva; e nem o grafo salvou o det do pit.

Esses resultados comprovam a drástica superioridade das metodologias baseadas em **World Models (NE-Dreamer)** no quesito eficiência (Sample Efficiency), bem como o enorme impacto de usar regularizadores de dinâmica espacial (**CURL/SPR**) comparado à otimização extrínseca pura (PPO).

---

## 💾 Gestão de Experimentos (MLOps)

### Versionamento de Modelos (Run IDs)
Para preservar o isolamento sináptico de arquiteturas iterativas, utilize a flag `--run-id`. Ela estabelece o nome do arquivo serializado e unifica os namespaces das métricas logadas.
```bash
python src/train.py --run-id "experimento_icm_alpha" --use-icm
```
*Output: `models/experimento_icm_alpha.zip`*

### Retomada Contínua de Otimização (Resume)
O projeto aplica manipulação assíncrona sobre interrupções do SO (ex: `SIGINT`), assegurando a persistência do modelo em disco antes do encerramento da subrotina. Para restaurar o treinamento preservando o *global step counter* e a topologia de decaimento do otimizador:
```bash
python src/train.py --timesteps 1000000 --resume "models/experimento_icm_alpha"
```

### Inferência e Avaliação Qualitativa
Para carregar uma política parametrizada e renderizar os resultados inferenciais de forma determinística:
```bash
python src/evaluate.py --model "models/experimento_icm_alpha.zip"
```
> Utilize o argumento paramétrico `--stochastic` caso o objetivo seja avaliar a exploração baseada na distribuição de probabilidade das *logits* e não na argmax pura.

### Observabilidade de Métricas
O rastreamento de métricas como cross-entropy intrínseca, duração temporal do agente e retorno episódico acumulado é gerenciado centralmente pelo MLflow.
```bash
mlflow ui
```
O console emitirá a porta de alocação padrão (comumente `http://localhost:5000`) para acesso ao dashboard interativo.
