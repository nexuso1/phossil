# Entrypoint for the Dummy models, which were sometimes used to 
# debug the training process

from token_classifier_base import TokenClassifier, TokenClassifierConfig
from transformers import AutoTokenizer
from training import parser, create_loss, run_training
import torch

class DummyClassifier(TokenClassifier):
    def __init__(self, config: TokenClassifierConfig, base_model) -> None:
        super().__init__(config, base_model)
        self.linear = torch.nn.Linear(1, 1) # So that the parameter list for optim isn't empty

    def forward(self, input_ids, attention_mask, batch_lens, **kwargs):
        return torch.zeros_like(input_ids, device=self.device).float()
    
    def predict(self, input_ids, attention_mask=None, return_dict=False, labels=None, **kwargs) -> torch.Tensor:
        preds = self.linear(input_ids.float().unsqueeze(-1))
        preds = preds - preds
        if labels is not None:
            return self.loss(preds.squeeze(), labels), preds
        return preds
    
    def train_predict(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor = None, return_dict=False, **kwargs):
        return self.predict(input_ids, attention_mask, return_dict, labels, **kwargs)

def create_model(args):
    config = TokenClassifierConfig(1, create_loss(args))
    model = DummyClassifier(config, None)
    return model, AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')

if __name__ == "__main__":
    args = parser.parse_args()
    run_training(args, create_model)