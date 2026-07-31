# This is the main entrypoint for running kinase-aware models
import torch

from utils import get_esm
from ft_selective import add_arguments as add_ft_args
from training import run_training, parser, create_loss
from ast import literal_eval
from dataclasses import dataclass, field
from modules import ResidualMLP
from torch.nn import Module
from ft_selective import SelectiveFinetuningClassifier, SelectiveFinetuningClassifierConfig

@dataclass
class KinaseFTClassifierConfig(SelectiveFinetuningClassifierConfig):
    n_kinases = 389 # Based on the number of kinases in the Kinase-library tool
    mlp_layers : list[int] = field(default_factory=lambda : [256, 256, 256])

class KinaseFTClassifier(SelectiveFinetuningClassifier):
    def __init__(self, config: KinaseFTClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        self.kinase_head = torch.nn.Sequential(ResidualMLP(
            input_size=base_model.config.hidden_size,
            layer_sizes=config.mlp_layers,
            dropout=config.dropout_rate,
            norm=torch.nn.LayerNorm,
            activation=torch.nn.SiLU()),
            torch.nn.LazyLinear(config.n_kinases)
        )

        self.init_weights(self.kinase_head)

    def compute_loss(self, logits, labels, kinase_labels, kinase_logits, attention_mask=None, **kwargs):
        positive_labels = labels == 1
        valid_kinase_logits = kinase_logits[positive_labels]
        kinase_loss = self.loss(valid_kinase_logits.view(-1), kinase_labels.view(-1))

        return kinase_loss + super().compute_loss(logits, labels, attention_mask=attention_mask)
        
    def forward(self, input_ids, attention_mask, **kwargs):
        base_out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        kinase_logits = self.kinase_head(base_out[0])
        base_out['kinase_logits'] = kinase_logits
        return self.classifier(base_out[0]), base_out

def create_model(args):
    base, tokenizer = get_esm(args.type)
    config = KinaseFTClassifierConfig(1, loss=create_loss(args), mlp_layers=literal_eval(args.mlp_layers))
    classifier = KinaseFTClassifier(config, base)
    if not args.lora:
        classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def add_arguments(parser):
    add_ft_args(parser)
    parser.add_argument('--mlp_layers', type=str, default='[256, 256, 256]', help='Kinase Residual MLP classifier head layers')
    
def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    add_arguments(parser)

    args = parser.parse_args()
    main(args)