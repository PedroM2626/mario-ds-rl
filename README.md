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
