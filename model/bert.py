import torch
import torch.nn as nn

class BERTLanguageModel(nn.Module):
    def __init__(self, encoder, d_model: int, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU(approximate='tanh')
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(input_ids, mask)
        
        x = self.dense(encoder_outputs)
        x = self.activation(x)
        x = self.layer_norm(x)

        prediction_scores = self.decoder(x)
        return prediction_scores

    def tie_weights(self):
        if hasattr(self.encoder, 'embedding') and hasattr(self.encoder.embedding, 'token_embedding'):
            self.decoder.weight = self.encoder.embedding.token_embedding.weight
            print("Successfully tied input/output weights!")