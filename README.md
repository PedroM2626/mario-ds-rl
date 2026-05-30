# Mario NDS Reinforcement Learning 🍄🤖

Este projeto treina um agente de Inteligência Artificial usando **Reinforcement Learning (PPO)** para jogar *New Super Mario Bros* de Nintendo DS. Diferente de projetos tradicionais de emuladores que leem diretamente a RAM do jogo, este projeto utiliza **Visão Computacional (Optical Flow e Subtração de Frames)** para deduzir a posição do Mario e calcular recompensas dinamicamente!

Além disso, a arquitetura foi expandida para suportar pipelines de MLOps de estado da arte, incluindo **Aprendizado Não-Supervisionado (Autoencoders)** e **Curiosidade Intrínseca (ICM)**.

---

## 🛠 Pré-requisitos e Instalação

Certifique-se de ter as seguintes bibliotecas instaladas (recomendável usar um ambiente virtual conda/venv):

```bash
pip install gymnasium stable-baselines3[extra] opencv-python py-desmume mlflow torch tqdm numpy
```

> **Aviso:** O pacote `py-desmume` é essencial para inicializar o emulador do Nintendo DS internamente no Python.

### Estrutura de Pastas (Exigida)
Certifique-se de que a ROM do jogo e o seu Savestate (logo no início da primeira fase) estão na pasta `data/`.
- `data/mario.nds` (A ROM do Nintendo DS)
- `data/state.dst` (O arquivo de Savestate para a IA sempre renascer no mesmo lugar)

---

## 🚀 As 3 "Rotas" de Treinamento

O projeto suporta 3 metodologias distintas de treinamento de IA. Você pode escolher qual ativar usando flags de comando.

### 1. Treinamento Clássico (RL Puro)
Usa o algoritmo PPO padrão do Stable-Baselines3. A IA aprende a ver e jogar simultaneamente do zero.

**Comando:**
```bash
python src/train.py --rom "data/mario.nds" --state "data/state.dst" --timesteps 1000000 --num-envs 4
```
*(O parâmetro `--num-envs 4` abre 4 emuladores em paralelo para acelerar a coleta de dados).*

---

### 2. Visão Pré-treinada (Módulo Autoencoder)
Nesta rota, nós primeiro ensinamos a IA a "enxergar" a física do jogo de forma não-supervisionada, antes de ensiná-la a apertar botões. O aprendizado fica muito mais rápido!

**Passo A: Coletar os Dados**
```bash
python src/collect_data.py --rom "data/mario.nds" --state "data/state.dst" --num-frames 10000
```
*(Isso gerará milhares de fotos aleatórias e salvará em um `.npz`)*

**Passo B: Treinar a Visão (Unsupervised)**
```bash
python src/train_autoencoder.py --dataset "../data/mario_dataset.npz" --epochs 20
```
*(O PyTorch criará o "cérebro visual" do Mario e o salvará em `models/autoencoder.pth`)*

**Passo C: Treinar o Mario usando o Cérebro Visual**
```bash
python src/train.py --rom "data/mario.nds" --state "data/state.dst" --timesteps 1000000 --num-envs 4 --use-autoencoder
```

---

### 3. Explorador Curioso (Módulo ICM)
Nesta rota, o Mário recebe uma dose extra de **Dopamina e Curiosidade**. O Módulo ICM (Intrinsic Curiosity Module) avalia a surpresa da IA diante de cada imagem nova e dá pontos bônus para forçá-la a explorar buracos, canos e inimigos desconhecidos.

**Comando:**
```bash
python src/train.py --rom "data/mario.nds" --state "data/state.dst" --timesteps 1000000 --num-envs 4 --use-icm
```

---

## 💾 Comandos Úteis e MLOps

### Nomear o Treinamento (Evitar Sobrescrita)
Por padrão, o projeto salva o modelo como `ppo_mario.zip`. Se você for testar as 3 rotas, é essencial dar um nome para o seu treino para não apagar os outros. Use a flag `--run-id`:
```bash
python src/train.py --run-id "mario_icm_v1" --use-icm
```
Isso salvará o modelo como `models/mario_icm_v1.zip` e colocará o mesmo nome nos gráficos do MLflow!

### Continuar um Treino Parado (Resume)
O projeto conta com salvamento gracioso. Se você apertar `Ctrl+C`, ele salva o modelo sem corromper. Para continuar o treino de onde parou (mantendo gráficos contínuos e taxas de aprendizado precisas):
```bash
python src/train.py --rom "data/mario.nds" --state "data/state.dst" --timesteps 1000000 --resume "models/ppo_mario"
```

### Avaliar o Modelo Treinado (Ver a IA Jogando)
Para abrir uma janela visual e assistir ao Mario jogando sem que os pesos neurais mudem:
```bash
python src/evaluate.py --rom "data/mario.nds" --state "data/state.dst" --model "models/ppo_mario.zip"
```
> Dica: Se quiser ver a IA jogando de forma um pouco mais imprevisível e exploratória, adicione a flag `--stochastic` no final do comando.

### Acompanhar Gráficos de Aprendizado (MLflow)
Este projeto usa MLOps. Para ver os gráficos de recompensa, morte, duração do episódio e curiosidade em tempo real, abra um terminal e rode:
```bash
mlflow ui
```
Em seguida, acesse `http://localhost:5000` no seu navegador!

---

## 📊 Comparativo de Performance das 3 Rotas

Qual rota é a "melhor"? Depende do que você quer priorizar: velocidade do processador ou velocidade de aprendizado.

| Rota | Velocidade do PC (FPS) | Eficiência de Aprendizado (Sample Efficiency) | Descrição |
|---|---|---|---|
| **1. RL Puro** | 🚀🚀🚀 (Muito Rápido) | 🐢 (Lento) | O PC roda rápido, mas a IA demora para aprender o que é um inimigo. |
| **2. Autoencoder** | 🚀🚀 (Rápido) | 🧠🧠🧠 (Mestre Rápido) | O melhor custo-benefício. O custo no PC é quase igual à Rota 1, mas o Mario aprende incrivelmente rápido pois sua "visão" já está calibrada. |
| **3. Módulo ICM** | 🐢 (Muito Lento) | 🧠🧠 (Muito Bom) | A curiosidade resolve quebra-cabeças complexos rapidamente, mas rodar 3 redes neurais ao mesmo tempo sacrifica o FPS do seu processador. |

---

## 🧠 Curiosidades de IA (Reward Hacking)
Um dos maiores desafios resolvidos neste projeto foi o **Reward Hacking**. Como o emulador de DS possui uma "Zona Morta" de câmera (a câmera não acompanha o Mario nos primeiros passos da fase), o Optical Flow clássico punia a IA por andar para frente. 

A IA aprendeu a hackear o sistema: ela pulava no frame 0 (para causar um solavanco de pontuação falsa na câmera) e cometia suicídio no Goomba o mais rápido possível para resetar a fase e farmar pontos infinitamente!

**A Solução:** O projeto agora utiliza um rastreador de centro de massa `cv2.absdiff` em conjunto com o Optical Flow `np.median` para calcular um **X Universal**. O suicídio não é mais a opção matematicamente perfeita.
