from utils import get_esm
from training import  parser, create_loss, run_training
from token_classifier_base import TokenClassifierConfig, TokenClassifier
import torch

class BaselineClassifier(TokenClassifier):
    """
    Linear classifier on base model embeddings.
    """
    def __init__(self, config, base_model):
        super().__init__(config, base_model)
        base_hidden_size = base_model.config.hidden_size if 'hidden_size' in base_model.config else base_model.config.d_model
        self.classifier = torch.nn.Linear(base_hidden_size, config.n_labels)
        self.init_weights(self.classifier)
        self.set_base_requires_grad(False)
        self.base.eval()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.base.eval()
        return super().forward(input_ids, attention_mask, **kwargs)

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    config = TokenClassifierConfig(n_labels=1,loss=create_loss(args), base_type=args.type)
    model = BaselineClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    run_training(args, create_model)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
