import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MeleeEncoderDecoder(nn.Module):
    """Encoder-decoder model for Melee game state prediction using only continuous features."""
    
    def __init__(self, continuous_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 3):
        """
        Args:
            continuous_dim: Dimension of continuous features
            d_model: Dimension of the transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()

        # Input projection for encoder and decoder
        self.encoder_proj = nn.Linear(continuous_dim, d_model)
        self.decoder_proj = nn.Linear(continuous_dim, d_model)

        self.pos_enc = nn.Parameter(torch.randn(1, 100, d_model))  # fixed max seq_len

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Optional CNN after transformer
        self.post_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final output projection to continuous feature space
        self.continuous_proj = nn.Linear(d_model, continuous_dim)

        self.continuous_dim = continuous_dim

    def forward(self, src_cont: torch.Tensor, tgt_cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_cont: Source continuous features (batch, seq_len, continuous_dim)
            tgt_cont: Target continuous features (batch, seq_len, continuous_dim)

        Returns:
            Predicted continuous values (batch, tgt_len, continuous_dim)
        """
        # Positional encoding + input projection
        src = self.encoder_proj(src_cont) + self.pos_enc[:, :src_cont.size(1), :]
        tgt = self.decoder_proj(tgt_cont) + self.pos_enc[:, :tgt_cont.size(1), :]

        # Transformer input needs (seq_len, batch, d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        memory = self.encoder(src)
        decoder_output = self.decoder(tgt, memory)

        # Back to (batch, seq_len, d_model)
        decoder_output = decoder_output.permute(1, 0, 2)

        # Apply 1D CNN
        x = decoder_output.permute(0, 2, 1)   # (batch, d_model, seq_len)
        x = self.post_cnn(x)
        x = x.permute(0, 2, 1)               # (batch, seq_len, d_model)

        cont_preds = self.continuous_proj(x)
        return cont_preds

    def compute_loss(self, cont_preds: torch.Tensor, cont_targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute MSE loss between predictions and targets.

        Args:
            cont_preds: Predicted continuous values (batch, seq_len, continuous_dim)
            cont_targets: Target continuous values (batch, seq_len, continuous_dim)

        Returns:
            Total loss and breakdown
        """
        cont_loss = F.mse_loss(cont_preds, cont_targets)
        return cont_loss, {"continuous": cont_loss}
