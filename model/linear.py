# This is the main entrypoint for running linear models
import torch

from utils import get_esm
from training import run_training, parser, create_loss
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from torch.nn import Module

class LinearClassifier(TokenClassifier):
    def __init__(self, config: TokenClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        base_hidden_size = base_model.config.hidden_size if 'hidden_size' in base_model.config else base_model.config.d_model
        self.classifier = torch.nn.Linear(base_hidden_size, config.n_labels)
        self.init_weights(self.classifier)

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    config = TokenClassifierConfig(n_labels=1, loss=create_loss(args))
    model = LinearClassifier(base_model=esm, config=config)
    
    # Freeze the base if we're not using lora (in that case, it is frozen when applying it)
    if not args.lora:
        model.set_base_requires_grad(False)
    return model, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
