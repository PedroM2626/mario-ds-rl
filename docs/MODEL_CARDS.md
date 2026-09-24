# Model Cards: *New Super Mario Bros. DS* (EUR) RL Architecture Suite

This repository documents the model cards for the 14 reinforcement learning, representation learning, and world model architectures evaluated on *New Super Mario Bros.* (EUR release NTR-A2DP-EUR).

---

## 1. Taxonomic Overview

The 14 models span four distinct conceptual paradigms:
1. **Pixel-Based Representation Learning (Online & Auxiliary Losses):** `pure`, `drq`, `ae_frozen`, `hybrid`, `spr`, `curl`, `icm`, `icm_ae`.
2. **Deep & Recurrent Vision Policies:** `impala`, `recurrent`.
3. **Egocentric RAM & Geometric State Policies:** `ram100k`, `ramgeo100k`, `gnn100k`.
4. **Physics-Informed World Models & Model-Based RL:** `pinn_wm`, `dyna_ppo` (and `worldmodels_cma`/`worldmodels_ga`).

---

## 2. Shared Baseline Hyperparameters (PPO Foundations)

Unless specified otherwise in individual cards, all PPO-based agents share the following Stable-Baselines3 / PyTorch training configuration:
* **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-5}$)
* **Learning Rate:** $\alpha = 3.0 \times 10^{-4}$ (linear decay)
* **Discount Factor ($\gamma$):** $0.99$
* **GAE Parameter ($\lambda$):** $0.95$
* **PPO Clipping Parameter ($\epsilon_{\text{clip}}$):** $0.20$
* **Value Loss Coefficient ($c_1$):** $0.50$
* **Entropy Bonus Coefficient ($c_2$):** $0.01$
* **Rollout Horizon ($T_{\text{rollout}}$):** $512$ steps per environment
* **Mini-batch Size:** $256$
* **PPO Epochs per Update ($K$):** $10$
* **Frame Skip:** $8$ ($7.5\text{ Hz}$ decision rate)

---

## 3. Individual Model Cards

### 3.1 `pure` (Vanilla Nature CNN PPO)
* **Checkpoint:** [`models/ppo_pure_100k.zip`](file:///d:/mario-ds/models/ppo_pure_100k.zip) (39.2 MB)
* **Input Space:** $4 \times 84 \times 84$ Grayscale frame stack $\in [0, 1]^{4 \times 84 \times 84}$.
* **Action Space:** Discrete(7).
* **Architecture:** Nature CNN (Mnih et al., 2015):
  - Conv1: $8 \times 8$, stride 4, 32 filters, ReLU.
  - Conv2: $4 \times 4$, stride 2, 64 filters, ReLU.
  - Conv3: $3 \times 3$, stride 1, 64 filters, ReLU.
  - Linear: 512 units, ReLU.
  - Separate policy $\pi(a|s)$ and value $V(s)$ linear heads.
* **Loss Function:** Standard clipped surrogate PPO objective:
  $$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right] + c_1 \mathcal{L}_{\text{VF}}(\theta) - c_2 \mathcal{H}(\pi_\theta)$$
* **Training Budget:** 100,000 environment steps.
* **Benchmark Performance:** Det Mean: 120.80 | Stoch Mean: $295.51 \pm 80.93$.

---

### 3.2 `drq` (Data-Regularized Q / PPO with Random Shift)
* **Checkpoint:** [`models/ppo_drq_100k.zip`](file:///d:/mario-ds/models/ppo_drq_100k.zip) (39.2 MB)
* **Input Space:** $4 \times 84 \times 84$ Grayscale frame stack.
* **Architecture:** Nature CNN + DrQ data augmentation module.
* **Inductive Bias:** Pad $84 \times 84$ images with 4 pixels of reflection padding to $92 \times 92$, followed by random crop back to $84 \times 84$.
* **Loss Function:** PPO loss computed over augmented observation pairs $f(s)$ and $f'(s)$ to enforce transformation invariance:
  $$\mathcal{L}_{\text{DrQ}}(\theta) = \frac{1}{2}\left[\mathcal{L}_{\text{PPO}}(f(s), a) + \mathcal{L}_{\text{PPO}}(f'(s), a)\right]$$
* **Benchmark Performance:** Det Mean: 74.92 | Stoch Mean: $214.82 \pm 105.80$.

---

### 3.3 `ae_frozen` (Autoencoder Latent PPO)
* **Checkpoints:** [`models/ppo_autoencoder_frozen_100k.zip`](file:///d:/mario-ds/models/ppo_autoencoder_frozen_100k.zip) (26.3 MB), [`models/autoencoder.pth`](file:///d:/mario-ds/models/autoencoder.pth) (13.4 MB)
* **Input Space:** Pretrained latent vector $\mathbf{z} \in \mathbb{R}^{64}$ extracted from $4 \times 84 \times 84$ frames.
* **Architecture:**
  - Autoencoder: 4-layer convolutional encoder ($32 \to 64 \to 128 \to 256$) with bottleneck $\mathbf{z} \in \mathbb{R}^{64}$, transposed conv decoder.
  - Policy: Frozen encoder; 2-layer MLP ($[128, 64]$) acting on $\mathbf{z}$.
* **Objective:** Encoder pretrained via Mean Squared Error pixel reconstruction $\mathcal{L}_{\text{recon}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$; policy trained with frozen encoder weights.
* **Benchmark Performance:** Det Mean: 74.77 | Stoch Mean: $234.27 \pm 140.10$.

---

### 3.4 `hybrid` (AE Reconstruction + CURL Contrastive Latent)
* **Checkpoint:** [`models/ppo_hybrid_ae_curl_100k.zip`](file:///d:/mario-ds/models/ppo_hybrid_ae_curl_100k.zip) (39.7 MB)
* **Input Space:** $4 \times 84 \times 84$ Grayscale frame stack.
* **Architecture:** Shared convolutional backbone with dual heads: generative decoder + contrastive bilinear projection.
* **Loss Function:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + \alpha \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{CURL}}$$
* **Benchmark Performance:** Det Mean: 74.74 | Stoch Mean: $286.14 \pm 141.60$.

---

### 3.5 `spr` (Self-Predictive Representations)
* **Checkpoint:** [`models/ppo_spr_100k.zip`](file:///d:/mario-ds/models/ppo_spr_100k.zip) (39.2 MB)
* **Input Space:** $4 \times 84 \times 84$ Grayscale frame stack.
* **Architecture:** Online encoder $f_\theta$, target momentum encoder $f_\xi$, transition predictor $g_\phi$, projection head $h_\psi$.
* **Loss Formulation:** Multi-step forward latent predictive cosine similarity loss across $K=5$ future steps:
  $$\mathcal{L}_{\text{SPR}} = -\sum_{k=1}^K \frac{\langle h_\psi(\hat{\mathbf{z}}_{t+k}), \tilde{\mathbf{z}}_{t+k} \rangle}{\|h_\psi(\hat{\mathbf{z}}_{t+k})\|_2 \|\tilde{\mathbf{z}}_{t+k}\|_2}$$
* **Benchmark Performance:** Det Mean: 254.34 | Stoch Mean: $259.95 \pm 112.75$.

---

### 3.6 `curl` (Contrastive Unsupervised Representations)
* **Checkpoint:** [`models/ppo_curl_online_100k.zip`](file:///d:/mario-ds/models/ppo_curl_online_100k.zip) (39.2 MB)
* **Input Space:** $4 \times 84 \times 84$ Grayscale frame stack.
* **Architecture:** Query encoder $q = f_\theta(\text{crop}_1(s))$ and Momentum Key encoder $k = f_\xi(\text{crop}_2(s))$. Bilinear similarity matrix $W \in \mathbb{R}^{d \times d}$.
* **Loss Function:** InfoNCE contrastive loss over mini-batch negative samples:
  $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{q}^T W \mathbf{k}_+)}{\exp(\mathbf{q}^T W \mathbf{k}_+) + \sum_{j} \exp(\mathbf{q}^T W \mathbf{k}_j^-)}$$
* **Benchmark Performance:** Det Mean: 226.72 | Stoch Mean: $320.38 \pm 139.19$.

---

### 3.7 `icm` (Intrinsic Curiosity Module)
* **Checkpoint:** [`models/ppo_icm_100k.zip`](file:///d:/mario-ds/models/ppo_icm_100k.zip) (39.2 MB)
* **Input Space:** $4 \times 84 \times 84$ Grayscale frame stack.
* **Architecture:** Inverse dynamics model $g(\phi(s_t), \phi(s_{t+1})) \to \hat{a}_t$ and forward dynamics model $f(\phi(s_t), a_t) \to \hat{\phi}(s_{t+1})$.
* **Intrinsic Reward Formulation:**
  $$r_t^{\text{intr}} = \frac{\eta}{2} \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|_2^2$$
* **Benchmark Performance:** Det Mean: 222.49 | Stoch Mean: $231.37 \pm 94.60$.

---

### 3.8 `icm_ae` (ICM over Autoencoder Latents)
* **Checkpoint:** [`models/ppo_icm_ae_100k.zip`](file:///d:/mario-ds/models/ppo_icm_ae_100k.zip) (26.3 MB)
* **Input Space:** Latent representation $\mathbf{z}_t \in \mathbb{R}^{64}$.
* **Objective:** Evaluates inverse and forward dynamics directly inside compact Autoencoder latent manifold.
* **Benchmark Performance:** Det Mean: 74.77 | Stoch Mean: $230.09 \pm 99.38$.

---

### 3.9 `impala` (Deep Residual Convolutional Network)
* **Checkpoint:** [`models/mario_impala.zip`](file:///d:/mario-ds/models/mario_impala.zip) (26.3 MB)
* **Input Space:** $4 \times 84 \times 84$ Grayscale frame stack.
* **Architecture:** IMPALA CNN (Espeholt et al., 2018):
  - 3 Residual blocks with channel depths $[16, 32, 32]$.
  - Each block: Conv $3 \times 3$, MaxPool $3 \times 3$ stride 2, two residual sub-blocks.
  - Linear 256 units + ReLU.
* **Benchmark Performance:** Det Mean: 74.74 | Stoch Mean: $290.16 \pm 161.17$.

---

### 3.10 `recurrent` (Recurrent PPO with LSTM Memory)
* **Checkpoint:** [`models/recurrent_ppo_mario.zip`](file:///d:/mario-ds/models/recurrent_ppo_mario.zip) (39.2 MB)
* **Input Space:** $1 \times 84 \times 84$ Grayscale frame + recurrent hidden state $(h_t, c_t) \in \mathbb{R}^{256}$.
* **Architecture:** Nature CNN encoder feeding into an LSTM cell ($256$ hidden units) to resolve partial observability in horizontally scrolling stages.
* **Benchmark Performance:** Det Mean: 31.74 | Stoch Mean: $516.80 \pm 131.04$ (Highest stochastic mean among pure vision models).

---

### 3.11 `ram100k` (Egocentric RAM MLP Policy)
* **Checkpoint:** [`models/ram_ppo_100k.zip`](file:///d:/mario-ds/models/ram_ppo_100k.zip) (160 KB)
* **Input Space:** State vector $\mathbf{s}_t \in \mathbb{R}^{16}$ from DS Main RAM:
  $[v_x, v_y, \mathbf{1}_{\text{ground}}, \Delta X_{\text{cam}}, \text{odo}, \{dx_i, dy_i, \text{type}_i\}_{i=0}^2]$.
* **Architecture:** 2-layer MLP $[64, 64]$ with Tanh activations. Parameter count: ~4,500 weights ($<0.5\text{ ms}$ inference).
* **Benchmark Performance:** Det Mean: 61.40 | Stoch Mean: $392.39 \pm 223.91$.

---

### 3.12 `ramgeo100k` (Geometry-Augmented RAM Policy)
* **Checkpoint:** [`models/ram_geo_100k.zip`](file:///d:/mario-ds/models/ram_geo_100k.zip) (170 KB)
* **Input Space:** State vector $\mathbf{s}_t \in \mathbb{R}^{21}$ (16 RAM dimensions + 5 forward surface continuity flags).
* **Architecture:** 2-layer MLP $[64, 64]$ with Tanh activations.
* **Benchmark Performance:** Det Mean: 61.40 | Stoch Mean: $521.95 \pm 425.36$ (Highest stochastic mean overall).

---

### 3.13 `gnn100k` (Graph Neural Interaction Network)
* **Checkpoint:** [`models/gnn_100k.zip`](file:///d:/mario-ds/models/gnn_100k.zip) (1.87 MB)
* **Input Space:** Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with Mario and dynamic enemy sprite nodes.
* **Architecture:** 2-layer Message Passing Neural Network (MPNN):
  $$m_{ij} = \phi_e(v_i, v_j, e_{ij}), \quad v_i' = \phi_v\left(v_i, \sum_{j \in \mathcal{N}(i)} m_{ij}\right)$$
* **Benchmark Performance:** Det Mean: 61.40 | Stoch Mean: $401.11 \pm 312.21$.

---

### 3.14 `pinn_wm` & `dyna_ppo` (Physics-Informed World Model + Dyna-PPO)
* **Checkpoints:**
  - PINN World Model: [`models/pinn_nsmb.pt`](file:///d:/mario-ds/models/pinn_nsmb.pt) (180 KB)
  - Dyna-PPO Policy: [`models/pinn_policy.zip`](file:///d:/mario-ds/models/pinn_policy.zip) (173 KB)
  - Transition Pool: [`models/pinn_pool.npz`](file:///d:/mario-ds/models/pinn_pool.npz) (156 KB, 3,500 real emulator transitions)
* **World Model Architecture (Hard-Residual PINN):**
  - Parameter count: 43,440.
  - Kinematic Hard Residual Formulation:
    $$x_{t+1} = x_t + v_{x, t} \cdot \Delta t, \quad y_{t+1} = y_t + v_{y, t} \cdot \Delta t$$
  - Learned Neural Residual Head: Predicts nonlinear forces, friction, terrain collision normal constraints, and sprite interaction deltas.
  - Classification Head: Class-weighted binary cross-entropy ($w_{\text{death}} = 25.0$) predicting episode termination / lethal hazard contact.
* **Dyna-PPO Policy:**
  - 2-layer MLP $[64, 64]$ trained 100% inside learned imagination rollouts ($H=200$). Zero emulator interaction required during policy learning.
  - Training throughput: $>400\text{ imagination frames/second}$.
* **Control Modes:**
  - `mpc`: Closed-loop Cross-Entropy Method Model Predictive Control (CEM-MPC, $H=12, N=128$).
  - `ppo+reflex`: Preemptive hazard reflex (running leap over dynamic Goombas / pits) layered over Dyna-PPO horizontal momentum policy.
