import ast
import torch

from utils import get_esm
from training import  parser, create_loss, run_training
from token_classifier_base import TokenClassifierConfig
from ft_selective import SelectiveFinetuningClassifier
from dataclasses import dataclass, field
from torch.nn import Module

@dataclass
class RecyclingFinetuningClassifierConfig(TokenClassifierConfig):
    unfreeze_indices : list[int] = field(default_factory= lambda : [-1])
    n_steps : int = 3

class RecyclingFinetuningClassifier(SelectiveFinetuningClassifier):
    def __init__(self, config: RecyclingFinetuningClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)

    def recycle_iteration(self, prev : torch.Tensor):
        # Assumes indices in a sequence
        for i in sorted(self.modified_indices):
                    
            base_out = self.base.encoder.layer[i](prev)
            if type(base_out) is tuple:
                base_out = base_out[0]
            prev = prev + base_out

        if self.base.encoder.emb_layer_norm_after:
            prev = self.base.encoder.emb_layer_norm_after(prev)

        return prev
        
    def classifier_features(self, first_pass_outputs : torch.Tensor, **kwargs):
        recycled_state = first_pass_outputs.detach()
        
        for i in range(self.config.n_steps):
            recycled_state = self.recycle_iteration(recycled_state)
            if i < self.config.n_steps - 1:
                recycled_state = recycled_state.detach()

        return recycled_state


def create_model(args):
    esm, tokenizer = get_esm(args.type)
    indices = ast.literal_eval(args.indices)
    config = RecyclingFinetuningClassifierConfig(n_labels=1,loss=create_loss(args), unfreeze_indices=indices,
                                                 base_type=args.type, n_steps=args.n_steps, dropout_rate=args.dropout)
    model = RecyclingFinetuningClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    run_training(args, create_model)
    
def add_arguments(parser):
    parser.add_argument('--indices', default="[-1, -2, -3]", help='Indices of base model layers to be unfrozen')
    parser.add_argument('--n_steps', default=3, type=int, help='Number of recycling steps')

if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
