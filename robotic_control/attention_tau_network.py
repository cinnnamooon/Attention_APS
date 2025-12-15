"""
Attention-based Tau Network for Adaptive Preference Scaling

This module implements a Transformer-based neural network to predict tau values
for preference pairs, replacing the Newton optimization method while preserving
the original implementation for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionTauNetwork(nn.Module):
    """
    Transformer-based network for predicting tau values
    
    Key idea: Different preference pairs may be correlated.
    The network uses self-attention to capture these relationships
    and predict better tau values than independent Newton optimization.
    
    Architecture:
    1. Feature encoder: Extract features from each preference pair
    2. Transformer layers: Capture inter-pair relationships
    3. Tau predictor: Predict tau for each pair
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        tau_min=0.1,
        tau_max=1.5,
        device='cpu'
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            tau_min: Minimum tau value
            tau_max: Maximum tau value
            device: Device to run on
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.device = device
        
        # Feature encoder: Extract features from preference pairs
        input_dim = (obs_dim + action_dim) * 2 + 4  # two trajectories + meta features
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Positional encoding (optional, helps distinguish pairs)
        self.use_positional_encoding = True
        if self.use_positional_encoding:
            self.positional_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better training
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Tau prediction head
        self.tau_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def extract_pair_features(self, observations, actions, rm_predictions, winning_idx, losing_idx):
        """
        Extract features for each preference pair
        
        Args:
            observations: [batch_size, trajectory_length, obs_dim] or [batch_size, obs_dim]
            actions: [batch_size, trajectory_length, action_dim] or [batch_size, action_dim]
            rm_predictions: [batch_size] predicted rewards
            winning_idx: indices of winning trajectories
            losing_idx: indices of losing trajectories
            
        Returns:
            pair_features: [num_pairs, feature_dim]
        """
        # Handle 3D inputs (with trajectory dimension) by summing over trajectory
        if observations.dim() == 3:
            # Sum over trajectory dimension: [batch, traj_len, obs_dim] -> [batch, obs_dim]
            observations = observations.sum(dim=1)
        if actions.dim() == 3:
            # Sum over trajectory dimension: [batch, traj_len, action_dim] -> [batch, action_dim]
            actions = actions.sum(dim=1)
        
        # Get winning and losing trajectories
        obs_win = observations[winning_idx]
        obs_lose = observations[losing_idx]
        action_win = actions[winning_idx]
        action_lose = actions[losing_idx]
        
        # Ensure they are 2D after indexing
        if obs_win.dim() == 3:
            obs_win = obs_win.sum(dim=1)
        if obs_lose.dim() == 3:
            obs_lose = obs_lose.sum(dim=1)
        if action_win.dim() == 3:
            action_win = action_win.sum(dim=1)
        if action_lose.dim() == 3:
            action_lose = action_lose.sum(dim=1)
        
        # Compute reward-based features
        reward_win = rm_predictions[winning_idx]
        reward_lose = rm_predictions[losing_idx]
        
        reward_diff = reward_win - reward_lose
        reward_sum = reward_win + reward_lose
        reward_diff_abs = torch.abs(reward_diff)
        
        # Compute difficulty/uncertainty features
        # Higher sigmoid gradient = more uncertain
        sigmoid_val = torch.sigmoid(reward_diff)
        uncertainty = sigmoid_val * (1 - sigmoid_val)
        
        # Meta features - ensure all are 1D first, then stack to 2D
        # Flatten any extra dimensions
        reward_diff = reward_diff.flatten()
        reward_diff_abs = reward_diff_abs.flatten()
        reward_sum = reward_sum.flatten()
        uncertainty = uncertainty.flatten()
        
        meta_features = torch.stack([
            reward_diff,
            reward_diff_abs,
            reward_sum,
            uncertainty
        ], dim=1)  # Stack to [num_pairs, 4]
        
        # Ensure obs and actions are also 2D
        if obs_win.dim() > 2:
            obs_win = obs_win.reshape(obs_win.size(0), -1)
        if obs_lose.dim() > 2:
            obs_lose = obs_lose.reshape(obs_lose.size(0), -1)
        if action_win.dim() > 2:
            action_win = action_win.reshape(action_win.size(0), -1)
        if action_lose.dim() > 2:
            action_lose = action_lose.reshape(action_lose.size(0), -1)
        
        # Concatenate all features
        pair_features = torch.cat([
            obs_win,
            obs_lose,
            action_win,
            action_lose,
            meta_features
        ], dim=1)
        
        return pair_features
    
    def forward(self, observations, actions, rm_predictions, winning_idx, losing_idx):
        """
        Forward pass: predict tau for each preference pair
        
        Args:
            observations: [batch_size, obs_dim]
            actions: [batch_size, action_dim]
            rm_predictions: [batch_size] predicted rewards
            winning_idx: indices of winning trajectories
            losing_idx: indices of losing trajectories
            
        Returns:
            tau: [num_pairs] predicted tau values
        """
        # Extract features for each pair
        pair_features = self.extract_pair_features(
            observations, actions, rm_predictions, winning_idx, losing_idx
        )  # [num_pairs, input_dim]
        
        # Encode features
        encoded = self.feature_encoder(pair_features)  # [num_pairs, hidden_dim]
        
        # Add batch dimension for transformer (expects [batch, seq, features])
        encoded = encoded.unsqueeze(0)  # [1, num_pairs, hidden_dim]
        
        # Add positional encoding
        if self.use_positional_encoding:
            encoded = self.positional_encoder(encoded)
        
        # Apply transformer (captures relationships between pairs)
        transformed = self.transformer(encoded)  # [1, num_pairs, hidden_dim]
        
        # Remove batch dimension
        transformed = transformed.squeeze(0)  # [num_pairs, hidden_dim]
        
        # Predict tau for each pair
        tau_logits = self.tau_predictor(transformed).squeeze(-1)  # [num_pairs]
        
        # Map to valid range using sigmoid
        tau_normalized = torch.sigmoid(tau_logits)
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_normalized
        
        return tau


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TauNetworkTrainer:
    """
    Trainer for the attention-based tau network
    
    Training strategy:
    1. Warm-up phase: Use Newton method to generate pseudo-labels
    2. Supervised learning: Train network to predict Newton results
    3. Fine-tuning: Jointly optimize with reward model
    """
    
    def __init__(
        self,
        tau_network,
        tau_min=0.1,
        tau_max=1.5,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device='cpu'
    ):
        self.tau_network = tau_network
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.device = device
        
        # Initialize network weights with smaller values for stability
        self._init_weights()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            tau_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8  # Add epsilon for numerical stability
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Training statistics
        self.training_step = 0
        self.loss_history = []
    
    def _init_weights(self):
        """Initialize weights for better diversity"""
        for module in self.tau_network.modules():
            if isinstance(module, torch.nn.Linear):
                # Use Xavier with gain=1.0 for better output diversity
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    # Add small random bias to encourage diversity
                    torch.nn.init.uniform_(module.bias, -0.1, 0.1)
            elif isinstance(module, torch.nn.LayerNorm):
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
    def compute_newton_tau(self, predicted_reward_diff, rho, n_iters=5):
        """
        Compute tau using Newton method (for generating training targets)
        
        This is the ORIGINAL method, used to create pseudo-labels
        """
        tau = torch.ones_like(predicted_reward_diff) * self.tau_max
        
        for i in range(n_iters):
            # Compute sigmoid with clamping for numerical stability
            sig_val = torch.sigmoid(predicted_reward_diff / tau)
            sig_val = torch.clamp(sig_val, 1e-7, 1.0 - 1e-7)
            
            # Gradient (with safe log)
            log_term = torch.clamp(2 * sig_val, min=1e-7)
            grad_tau = -torch.log(log_term) + (1 - sig_val) * (predicted_reward_diff / tau) + rho
            
            # Hessian
            hess_tau = ((predicted_reward_diff ** 2 / (tau ** 3 + 1e-8)) * 
                       (1 - sig_val) * sig_val)
            hess_tau = torch.clamp(hess_tau, min=1e-8)
            
            # Newton update
            newton_dir = -grad_tau / hess_tau
            tau = (tau + newton_dir).clamp_(min=self.tau_min, max=self.tau_max)
            
            # Check for NaN and break if found
            if torch.isnan(tau).any():
                print(f"Warning: NaN in tau at Newton iteration {i}, reverting to max")
                tau = torch.ones_like(predicted_reward_diff) * self.tau_max
                break
        
        return tau.detach()
    
    def train_step(self, observations, actions, rm_predictions, winning_idx, losing_idx, rho=0.1):
        """
        Single training step
        
        Args:
            observations: [batch_size, obs_dim]
            actions: [batch_size, action_dim]
            rm_predictions: [batch_size] predicted rewards
            winning_idx: indices of winning trajectories
            losing_idx: indices of losing trajectories
            rho: regularization parameter
            
        Returns:
            metrics: dict of training metrics
        """
        self.tau_network.train()
        
        # Clamp inputs for numerical stability BEFORE passing to network
        observations = torch.clamp(observations, -10, 10)
        actions = torch.clamp(actions, -10, 10)
        rm_predictions = torch.clamp(rm_predictions, -100, 100)
        
        # Check for NaN in inputs
        if torch.isnan(observations).any():
            print(f"ERROR: NaN in observations before network")
            return {
                'loss': float('nan'),
                'tau_mean': float('nan'),
                'tau_std': float('nan'),
                'tau_at_upper': 0.0,
                'tau_at_lower': 0.0,
                'diversity_loss': 0.0
            }
        if torch.isnan(actions).any():
            print(f"ERROR: NaN in actions before network")
            return {
                'loss': float('nan'),
                'tau_mean': float('nan'),
                'tau_std': float('nan'),
                'tau_at_upper': 0.0,
                'tau_at_lower': 0.0,
                'diversity_loss': 0.0
            }
        if torch.isnan(rm_predictions).any():
            print(f"ERROR: NaN in rm_predictions before network")
            return {
                'loss': float('nan'),
                'tau_mean': float('nan'),
                'tau_std': float('nan'),
                'tau_at_upper': 0.0,
                'tau_at_lower': 0.0,
                'diversity_loss': 0.0
            }
        
        # Predict tau using network
        tau_pred = self.tau_network(observations, actions, rm_predictions, winning_idx, losing_idx)
        
        # Debug: Check tau_pred immediately after network output
        if torch.isnan(tau_pred).any():
            print(f"ERROR: NaN in tau_pred from network output!")
            print(f"  obs range: [{observations.min().item():.3f}, {observations.max().item():.3f}]")
            print(f"  actions range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
            print(f"  rm_pred range: [{rm_predictions.min().item():.3f}, {rm_predictions.max().item():.3f}]")
            print(f"  num pairs: {len(winning_idx)}")
        
        # Compute target tau using Newton method
        reward_diff = rm_predictions[winning_idx] - rm_predictions[losing_idx]
        reward_diff = torch.clamp(reward_diff, -100, 100)  # Prevent extreme values
        tau_target = self.compute_newton_tau(reward_diff, rho)
        
        # Debug: Check tau_target
        if torch.isnan(tau_target).any():
            print(f"ERROR: NaN in tau_target from Newton method!")
            print(f"  reward_diff range: [{reward_diff.min().item():.3f}, {reward_diff.max().item():.3f}]")
            print(f"  rho: {rho}")
        
        # Ensure tau_target has same shape as tau_pred for MSE loss
        tau_target = tau_target.view_as(tau_pred)
        
        # Check for NaN in predictions or targets
        if torch.isnan(tau_pred).any() or torch.isnan(tau_target).any():
            print("Warning: NaN detected in tau predictions or targets, skipping update")
            return {
                'loss': float('nan'),
                'tau_mean': float('nan'),
                'tau_std': float('nan'),
                'tau_at_upper': 0.0,
                'tau_at_lower': 0.0,
                'diversity_loss': 0.0
            }
        
        # Supervised loss: MSE between predicted and Newton tau
        supervised_loss = F.mse_loss(tau_pred, tau_target)
        
        # Regularization: encourage diversity in tau values (MUCH STRONGER)
        tau_std = tau_pred.std() + 1e-8  # Add epsilon to prevent division by zero
        target_std = (self.tau_max - self.tau_min) * 0.3  # Target std = 0.42
        diversity_loss = F.mse_loss(tau_std, torch.tensor(target_std, device=self.device))
        
        # Additional diversity: penalize low variance
        variance_penalty = F.relu(0.01 - tau_pred.var())  # Penalize if var < 0.01
        
        # Regularization: prevent extreme values
        margin = (self.tau_max - self.tau_min) * 0.1
        upper_bound_loss = F.relu(tau_pred - (self.tau_max - margin)).mean()
        lower_bound_loss = F.relu((self.tau_min + margin) - tau_pred).mean()
        
        # Total loss - INCREASE diversity weight significantly
        total_loss = (
            0.5 * supervised_loss +        # Reduce supervised weight
            2.0 * diversity_loss +          # MUCH higher diversity weight
            1.0 * variance_penalty +        # Penalize collapse
            0.5 * (upper_bound_loss + lower_bound_loss)
        )
        
        # Check for NaN in loss
        if torch.isnan(total_loss):
            print("Warning: NaN detected in loss, skipping update")
            return {
                'loss': float('nan'),
                'tau_mean': tau_pred.mean().item() if not torch.isnan(tau_pred).any() else float('nan'),
                'tau_std': tau_pred.std().item() if not torch.isnan(tau_pred).any() else float('nan'),
                'tau_at_upper': 0.0,
                'tau_at_lower': 0.0,
                'diversity_loss': diversity_loss.item() if not torch.isnan(diversity_loss).any() else 0.0
            }
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Check for NaN in gradients
        has_nan_grad = False
        for param in self.tau_network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print("Warning: NaN detected in gradients, skipping update")
            self.optimizer.zero_grad()
            return {
                'loss': total_loss.item(),
                'tau_mean': tau_pred.mean().item(),
                'tau_std': tau_pred.std().item(),
                'tau_at_upper': 0.0,
                'tau_at_lower': 0.0,
                'diversity_loss': diversity_loss.item()
            }
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.tau_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update statistics
        self.training_step += 1
        self.loss_history.append(total_loss.item())
        
        # Compute metrics
        with torch.no_grad():
            tau_mean = tau_pred.mean().item()
            tau_median = tau_pred.median().item()
            tau_min_val = tau_pred.min().item()
            tau_max_val = tau_pred.max().item()
            tau_std_val = tau_pred.std().item()
            
            # Percentage at boundaries
            at_lower = (tau_pred < self.tau_min + 0.01).float().mean().item()
            at_upper = (tau_pred > self.tau_max - 0.01).float().mean().item()
        
        metrics = {
            'loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'bound_loss': (upper_bound_loss + lower_bound_loss).item(),
            'tau_mean': tau_mean,
            'tau_median': tau_median,
            'tau_std': tau_std_val,
            'tau_min': tau_min_val,
            'tau_max': tau_max_val,
            'tau_at_lower': at_lower,
            'tau_at_upper': at_upper,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'network_state_dict': self.tau_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_step': self.training_step,
            'loss_history': self.loss_history,
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.tau_network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_step = checkpoint['training_step']
        self.loss_history = checkpoint['loss_history']
