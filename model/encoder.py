import torch
import torch.nn as nn
from typing import Optional
from .embedding import TransformerEmbedding
from .encoder_layer import EncoderLayer
from torch.utils.checkpoint import checkpoint

class TransformerEncoder(nn.Module):
    """
    Implements full Transformer Encoder model.
    Optimized with Gradient Checkpointing option.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_len: int,
        dropout: float,
        use_checkpointing: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_checkpointing = use_checkpointing
        
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
            
        return x