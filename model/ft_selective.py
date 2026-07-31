# This file is the main way to run traninig on finetuning models

import ast
import torch

from utils import get_esm
from training import  parser, create_loss, run_training
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from dataclasses import dataclass, field
from torch.nn import Module
@dataclass
class SelectiveFinetuningClassifierConfig(TokenClassifierConfig):
    unfreeze_indices : list[int] = field(default_factory= lambda : [-1])
    dropout_rate = 0

class SelectiveFinetuningClassifier(TokenClassifier):
    def __init__(self, config: SelectiveFinetuningClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        base_hidden_size = base_model.config.hidden_size if 'hidden_size' in base_model.config else base_model.config.d_model
        self.classifier = torch.nn.Linear(base_hidden_size, config.n_labels)
        self.init_weights(self.classifier)
        self.set_base_requires_grad(False)
        self.set_indexed_layers_grad(config.unfreeze_indices, True)

        # Overrides dropout in the unfrozen base model layers
        self.set_dropout_unfrozen()
        
    def set_indexed_layers_grad(self, indices : list[int], req_grad_value : bool):
        indices = set(indices)
        self.modified_indices = indices
        root = self.base.encoder.layer if 'encoder' in [n for n, _ in self.base.named_children()] else self.base.transformer.blocks
        param_list = list(root.named_children())
        for i in indices:
            # index 0 contains the name, 1 the parameter
            for param in param_list[i][1].parameters():
                param.requires_grad = req_grad_value

    def set_dropout_prob(self, model, prob):
        """
        Sets the dropout probability for all dropout layers in a model.
        """
        for m in model.modules():
            if isinstance(m, (torch.nn.Dropout)):
                m.p = prob

    def set_dropout_unfrozen(self):
        root = self.base.encoder.layer if 'encoder' in [n for n, _ in self.base.named_children()] else self.base.transformer.blocks
        for i in self.modified_indices:
            self.set_dropout_prob(root[i], self.config.dropout_rate)
    

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    indices = ast.literal_eval(args.indices)
    config = SelectiveFinetuningClassifierConfig(n_labels=1,loss=create_loss(args), unfreeze_indices=indices,
                                                 base_type=args.type, dropout_rate=args.dropout)
    model = SelectiveFinetuningClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    run_training(args, create_model)
    
def add_arguments(parser):
    parser.add_argument('--indices', default="[-1]", help='Indices of base model layers to be unfrozen')
    
if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
