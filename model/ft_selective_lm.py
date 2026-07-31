# This file is the main way to run training on finetuning + language modeling models
import ast
import torch

from utils import get_esm
from training import  parser, create_loss, run_training
from lm_model_base import LMModel, LMModelConfig
from ft_selective import SelectiveFinetuningClassifierConfig
from dataclasses import dataclass, field
from torch.nn import Module


@dataclass
class LMFinetuningClassifierConfig(LMModelConfig):
    unfreeze_indices : list[int] = field(default_factory= lambda : [-1])

class LMFinetuningClassifier(LMModel):
    def __init__(self, config: SelectiveFinetuningClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        self.classifier = torch.nn.Linear(base_model.config.hidden_size, 1)
        self.init_weights(self.classifier)
        self.set_base_requires_grad(False)
        self.set_indexed_layers_grad(config.unfreeze_indices, True)
        
    def set_indexed_layers_grad(self, indices : list[int], req_grad_value : bool):
        indices = set(indices)
        self.modified_indices = indices
        param_list = list(self.base.esm.encoder.layer.named_children())
        for i in indices:
            # index 0 contains the name, 1 the parameter
            for param in param_list[i][1].parameters():
                param.requires_grad = req_grad_value

    def forward(self, input_ids, attention_mask, **kwargs):
        base_output = self.base(attention_mask=attention_mask, input_ids=input_ids, output_hidden_states=True)
        return self.classifier(base_output.hidden_states[-1]), base_output


def create_model(args):
    esm, tokenizer = get_esm(args.type, masked_lm=True)
    indices = ast.literal_eval(args.indices)
    config = LMFinetuningClassifierConfig(loss=create_loss(args), unfreeze_indices=indices,
                                                 base_type=args.type, dropout=args.dropout)
    model = LMFinetuningClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    run_training(args, create_model)
    
def add_arguments(parser):
    parser.add_argument('--indices', default="[-1]", help='Indices of base model layers to be unfrozen')
    
if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
