import torch
import torch.nn as nn
import numpy as np
from impala_cnn import ImpalaCNNFeaturesExtractor
from gymnasium import spaces

class DecisionTransformer(nn.Module):
    """
    Decision Transformer (DT) architecture tailored for Super Mario Bros DS.
    Learns to predict actions conditioned on returns-to-go, states, and history.
    """
    def __init__(
        self,
        state_dim: int = 512,
        act_dim: int = 6,
        hidden_size: int = 256,
        max_ep_len: int = 8192,
        n_layer: int = 3,
        n_head: int = 8,
        n_inner: int = 1024,
        activation_function: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        
        # State encoder: Impala CNN
        # Constrói um espaço de observação dummy de 1 canal 84x84 para instanciar o extrator
        obs_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.state_encoder = ImpalaCNNFeaturesExtractor(obs_space, features_dim=state_dim)
        
        # Embeddings para projetar inputs à dimensão oculta do transformer
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_action = nn.Embedding(act_dim, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Causal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=n_inner,
            dropout=dropout,
            activation=activation_function,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Prediction heads
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, states, actions, rtgs, timesteps, attention_mask=None):
        # states: (B, T, 1, 84, 84)
        # actions: (B, T)
        # rtgs: (B, T, 1)
        # timesteps: (B, T)
        # attention_mask: (B, T) - 1 para válidos, 0 para padding
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # 1. Mapeamento de estados (passagem pela CNN)
        flat_states = states.reshape(batch_size * seq_len, 1, 84, 84)
        if flat_states.dtype == torch.uint8 or flat_states.max() > 1.0:
            flat_states = flat_states.float() / 255.0
            
        state_feats = self.state_encoder(flat_states) # (B * T, state_dim)
        state_embeddings = self.embed_state(state_feats).reshape(batch_size, seq_len, self.hidden_size)
        
        # 2. Embeddings das outras modalidades
        action_embeddings = self.embed_action(actions) # (B, T, hidden_size)
        rtg_embeddings = self.embed_rtg(rtgs) # (B, T, hidden_size)
        time_embeddings = self.embed_timestep(timesteps) # (B, T, hidden_size)
        
        # Somar timestep embeddings a cada token de seu respectivo timestep
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings
        
        # 3. Intercalação sequencial: [rtg_0, state_0, action_0, rtg_1, state_1, action_1, ...]
        # Saída esperada: (B, 3 * T, hidden_size)
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=2
        ) # (B, T, 3, hidden_size)
        stacked_inputs = stacked_inputs.reshape(batch_size, 3 * seq_len, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # 4. Construção das máscaras
        device = states.device
        causal_mask = nn.Transformer.generate_square_subsequent_mask(3 * seq_len, device=device)
        
        if attention_mask is not None:
            # Replicar máscara para os 3 tokens de cada timestep
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=2
            ).reshape(batch_size, 3 * seq_len)
            src_key_padding_mask = (stacked_attention_mask == 0)
        else:
            src_key_padding_mask = None
            
        # 5. Transformer forward
        transformer_outputs = self.transformer(
            stacked_inputs,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        ) # (B, 3 * T, hidden_size)
        
        # 6. Predição da ação
        # Usamos a representação do token de estado (posição 3t + 1) para predizer a ação t
        x = transformer_outputs[:, 1::3, :] # (B, T, hidden_size)
        action_logits = self.predict_action(x) # (B, T, act_dim)
        
        return action_logits
