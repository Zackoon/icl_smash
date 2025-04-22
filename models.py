import torch
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F

# Comment out the entire EnumEmbeddingModule class
class EnumEmbeddingModule(nn.Module):
    """Module for embedding categorical/enum features from Melee game states."""
    
    def __init__(self, enum_dims: Dict[str, int], embedding_dims: Dict[str, int]):
        """Initialize embedding layers for each enum feature."""
        super().__init__()
        self.embeddings = nn.ModuleDict({
            'stage': nn.Embedding(enum_dims['stage'], embedding_dims['stage']),
            'p1_action': nn.Embedding(enum_dims['p1_action'], embedding_dims['p1_action']),
            'p1_character': nn.Embedding(enum_dims['p1_character'], embedding_dims['p1_character']),
            'p2_action': nn.Embedding(enum_dims['p2_action'], embedding_dims['p2_action']),
            'p2_character': nn.Embedding(enum_dims['p2_character'], embedding_dims['p2_character'])
        })
        
        self.total_embedding_dim = sum(embedding_dims.values())
        self.feature_names = list(enum_dims.keys())

    def forward(self, enum_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass to embed all enum features."""
        if set(enum_inputs.keys()) != set(self.feature_names):
            raise ValueError(
                f"Expected enum features {self.feature_names}, "
                f"got {list(enum_inputs.keys())}"
            )
        
        embeddings = []
        for name in self.feature_names:
            embedded = self.embeddings[name](enum_inputs[name])
            embeddings.append(embedded)
        
        return torch.cat(embeddings, dim=-1)

class MeleeEncoderDecoder(nn.Module):
    """Encoder-decoder model for Melee game state prediction."""
    
    def __init__(self, continuous_dim: int, 
                 enum_dims: Dict[str, int], 
                 embedding_dims: Dict[str, int], 
                 d_model: int = 128, 
                 nhead: int = 4, num_layers: int = 3):
        """Initialize the encoder-decoder model.
        
        Args:
            continuous_dim: Dimension of continuous features
            enum_dims: Dictionary of enum feature dimensions (vocab sizes)
            embedding_dims: Dictionary of embedding dimensions for each enum feature
            d_model: Dimension of the transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.enum_embedder = EnumEmbeddingModule(enum_dims, embedding_dims)
        self.enum_embed_dim = self.enum_embedder.total_embedding_dim
        
        # Projections for encoder/decoder inputs
        self.encoder_proj = nn.Linear(continuous_dim, d_model)
        self.decoder_proj = nn.Linear(continuous_dim, d_model)

        self.pos_enc = nn.Parameter(torch.randn(1, 100, d_model))  # fixed max seq_len

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projections for continuous and enum features
        self.continuous_proj = nn.Linear(d_model, continuous_dim)
        self.enum_projs = nn.ModuleDict({
            name: nn.Linear(d_model, dim) 
            for name, dim in enum_dims.items()
        })
        
        self.continuous_dim = continuous_dim
        self.enum_dims = enum_dims

    def forward(self, src_cont: torch.Tensor, 
                src_enum: Dict[str, torch.Tensor], 
                tgt_cont: torch.Tensor, 
                tgt_enum: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            src_cont: Source continuous features (batch, seq_len, continuous_dim)
            src_enum: Dict of source enum features
            tgt_cont: Target continuous features (batch, seq_len, continuous_dim)
            tgt_enum: Dict of target enum features
            
        Returns:
            Tuple containing:
                - Predicted continuous values (batch, tgt_len, continuous_dim)
                - Dict of predicted enum logits for each enum feature
        """
        # Step 1: Embed enums
        src_enum_embed = self.enum_embedder(src_enum)  # (batch, seq_len, enum_embed_dim)
        tgt_enum_embed = self.enum_embedder(tgt_enum)

        # Step 2: Concatenate continuous + enum embeddings
        src = self.encoder_proj(src_cont)
        tgt = self.decoder_proj(tgt_cont)
        src = torch.cat([src, src_enum_embed], dim=-1)
        tgt = torch.cat([tgt, tgt_enum_embed], dim=-1)

        # Step 3: Positional encoding + projection
        src = self.encoder_proj(src) + self.pos_enc[:, :src.size(1), :]
        tgt = self.decoder_proj(tgt) + self.pos_enc[:, :tgt.size(1), :]

        # Step 4: Transformer (permute to seq_len, batch, d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        memory = self.encoder(src)
        decoder_output = self.decoder(tgt, memory)
        
        # Permute back to (batch, seq_len, d_model)
        decoder_output = decoder_output.permute(1, 0, 2)

        # Generate predictions for both continuous and enum features
        cont_preds = self.continuous_proj(decoder_output)
        enum_preds = {
            name: self.enum_projs[name](decoder_output)
            for name in self.enum_dims.keys()
        }

        return cont_preds, enum_preds
    
    def compute_loss(self, cont_preds: torch.Tensor, 
                    enum_preds: Dict[str, torch.Tensor],
                    cont_targets: torch.Tensor, 
                    enum_targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute losses for continuous and enum predictions.
        
        Args:
            cont_preds: Predicted continuous values (batch, seq_len, continuous_dim)
            enum_preds: Dict of predicted logits (batch, seq_len, num_classes) for each enum
            enum_targets: Dict of target enum values (batch, seq_len) with integer labels
        
        Note: For cross_entropy, targets should be class indices (not one-hot)
              and predictions should be raw logits (not probabilities)
        """
        # MSE loss for continuous predictions
        cont_loss = F.mse_loss(cont_preds, cont_targets)
        
        # Cross entropy loss for each enum feature
        # Reshape to (batch*seq_len, num_classes) and (batch*seq_len) for logits and targets respectively
        enum_losses = {
            name: F.cross_entropy(
                enum_preds[name].reshape(-1, self.enum_dims[name]), # logits
                enum_targets[name].reshape(-1) # targets
            )
            for name in self.enum_dims.keys()
        }
        
        # Combine losses (you might want to add weights here)
        total_loss = cont_loss + sum(enum_losses.values())
        return total_loss, {"continuous": cont_loss, **enum_losses}
