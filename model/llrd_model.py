# This file is the main way to run training on layer-wise learning rate decay models
import re
import torch

from utils import get_esm
from training import  parser, create_loss, run_training
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from dataclasses import dataclass
from torch.nn import Module

# Path to the transformer layers inside the base model. ESM2 and ESM-C name them differently.
ESM2_LAYER_PATH = 'encoder.layer'
ESMC_LAYER_PATH = 'transformer.blocks'

@dataclass
class LLRDClassifierConfig(TokenClassifierConfig):
    # Learning rate of a layer is decay times the learning rate of the layer above it
    decay : float = 0.9
    # Learning rate of the classification head. Defaults to the learning rate of the run.
    head_lr : float | None = None

class LLRDClassifier(TokenClassifier):
    """
    Fully finetuned token classifier with a layer-wise decayed learning rate.

    Every base model layer is trained, but the lower a layer sits the smaller its learning rate:
    the topmost layer trains at the learning rate of the run and each layer below it at "decay"
    times the one above. The embeddings train at the lowest rate of all. The idea is that the
    general features in the lower layers need less adjusting than the task specific upper ones.

    The training loop picks the resulting parameter groups up through get_param_groups().
    """
    def __init__(self, config: LLRDClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        base_hidden_size = base_model.config.hidden_size if 'hidden_size' in base_model.config else base_model.config.d_model
        self.classifier = torch.nn.Linear(base_hidden_size, config.n_labels)
        self.init_weights(self.classifier)

        self.layer_path = ESM2_LAYER_PATH if 'encoder' in [n for n, _ in self.base.named_children()] \
                          else ESMC_LAYER_PATH
        # Unlike the other finetuning classifiers, the whole base model is trained
        self.set_base_requires_grad(True)
        self.set_dropout_prob(self.base, config.dropout_rate)

    def get_layers(self):
        """
        The transformer layers of the base model, the topmost one last.
        """
        root = self.base
        for name in self.layer_path.split('.'):
            root = getattr(root, name)

        return root

    def get_layer_index(self, param_name : str):
        """
        Index of the base model layer a parameter belongs to, counted from the bottom.

        Parameters of the base model that sit outside the layers, the embeddings and the final
        norm, get -1, so that they end up one decay step below the first layer.
        """
        match = re.match(rf'base\.{re.escape(self.layer_path)}\.(\d+)\.', param_name)
        return int(match.group(1)) if match else -1

    def get_no_decay_params(self):
        """
        Names of the parameters weight decay should not be applied to, biases and norm weights.
        """
        no_decay = {name for name, _ in self.named_parameters() if name.endswith('bias')}
        for module_name, module in self.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                no_decay.update(f'{module_name}.{name}' for name, _ in module.named_parameters())

        return no_decay

    def get_param_groups(self, lr, weight_decay, fix_decay=False):
        """
        Optimizer parameter groups, one per (base model layer, weight decay) combination.

        Frozen parameters are left out, so that the frozen phase optimizes the head alone.
        """
        n_layers = len(self.get_layers())
        head_lr = self.config.head_lr if self.config.head_lr is not None else lr
        no_decay = self.get_no_decay_params() if fix_decay else set()

        groups = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith('base.'):
                index = self.get_layer_index(name)
                # The topmost layer trains at lr, every layer below it is decayed once more
                param_lr = lr * self.config.decay ** (n_layers - 1 - index)
            else:
                index, param_lr = 'head', head_lr

            decayed = name not in no_decay
            group = groups.setdefault((index, decayed),
                                      {'params' : [], 'lr' : param_lr,
                                       'weight_decay' : weight_decay if decayed else 0.0})
            group['params'].append(param)

        return list(groups.values())

    def freeze_phase(self):
        """
        Trains the classification head only.
        """
        self.set_base_requires_grad(False)
        self.set_dropout_prob(self.base, 0)

    def unfreeze_phase(self):
        """
        Trains the whole base model, every layer at its own decayed learning rate.
        """
        self.set_base_requires_grad(True)
        self.set_dropout_prob(self.base, self.config.dropout_rate)


def create_model(args):
    esm, tokenizer = get_esm(args.type)
    config = LLRDClassifierConfig(n_labels=1, loss=create_loss(args), base_type=args.type,
                                  dropout_rate=args.dropout, decay=args.llrd_decay, head_lr=args.head_lr)
    model = LLRDClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    if args.hpc:
        torch.set_float32_matmul_precision('high')
    run_training(args, create_model)

def add_arguments(parser):
    parser.add_argument('--llrd_decay', type=float, default=0.9,
                        help='Learning rate decay per base model layer, going from the top layer down.')
    parser.add_argument('--head_lr', type=float, default=None,
                        help='Learning rate of the classification head. Defaults to the learning rate of the run.')

if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
