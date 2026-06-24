import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import FlattenExtractor
from gymnasium import spaces
from typing import Optional, Union, List, Dict, Any, Type

class CausalTransformerWrapper(nn.Module):
    """
    A drop-in causal transformer wrapper that replicates the interface of nn.LSTM
    to process sequences for Stable Baselines 3 Recurrent PPO.
    """
    def __init__(self, input_size: int, hidden_size: int, context_len: int = 16, nhead: int = 8, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_len = context_len
        self.num_layers = num_layers
        
        # Project input features to hidden_size if they are different
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
        # Multi-head attention layer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='relu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor, states: tuple):
        # x shape: (seq_len, batch_size, input_size)
        # states: (h, c) where h is shape (context_len, batch_size, hidden_size)
        h, c = states
        seq_len, batch_size, _ = x.shape
        
        # Project inputs
        x_proj = self.input_proj(x) # (seq_len, batch_size, hidden_size)
        
        if seq_len > 1:
            # Training pass (processes full rollout sequence of length e.g. 256)
            # Create a causal sequence mask to prevent step t from attending to future steps
            device = x.device
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            
            out = self.transformer(x_proj, mask=mask)
            
            # Update history state with the last context_len observations
            if seq_len >= self.context_len:
                new_h = x_proj[-self.context_len:]
            else:
                pad_len = self.context_len - seq_len
                new_h = torch.cat([h[-pad_len:], x_proj], dim=0)
                
            return out, (new_h, c)
        else:
            # Collection pass (single step execution, seq_len = 1)
            # Slide history window: remove oldest step and append new step
            new_h = torch.cat([h[1:], x_proj], dim=0) # (context_len, batch_size, hidden_size)
            
            # Run self-attention over the current context window
            mask = nn.Transformer.generate_square_subsequent_mask(self.context_len, device=x.device)
            out_history = self.transformer(new_h, mask=mask)
            
            # Output only the representation at the current step (last element)
            out = out_history[-1:]
            
            return out, (new_h, c)

class RecurrentTransformerActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    Recurrent PPO Policy that replaces the LSTM layers with Causal Transformers.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
        context_len: int = 16,
        nhead: int = 8,
    ):
        # Initialize standard recurrent policy first
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )
        
        # Override the state shape and context length
        self.context_len = context_len
        self.lstm_hidden_state_shape = (context_len, 1, lstm_hidden_size)
        
        # Replace the actor and critic LSTM layers with causal transformer wrappers
        self.lstm_actor = CausalTransformerWrapper(
            input_size=self.features_dim,
            hidden_size=lstm_hidden_size,
            context_len=context_len,
            nhead=nhead,
            num_layers=n_lstm_layers
        ).to(self.device)
        
        if self.enable_critic_lstm:
            self.lstm_critic = CausalTransformerWrapper(
                input_size=self.features_dim,
                hidden_size=lstm_hidden_size,
                context_len=context_len,
                nhead=nhead,
                num_layers=n_lstm_layers
            ).to(self.device)
            
        # Re-initialize the optimizer with the new parameters
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
