# This file is the main way to run training on LoRA models
import ast
import torch

from utils import get_esm
from training import  parser, create_loss, run_training
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from dataclasses import dataclass, field
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import Module

# Names of the base model modules LoRA is applied to. ESM2 and ESM-C use different
# attention implementations, so they need different module names.
ESM2_TARGET_MODULES = ['query', 'key', 'value']
ESMC_TARGET_MODULES = ['layernorm_qkv.1', 'out_proj']

def default_target_modules(base_type : str):
    return ESMC_TARGET_MODULES if base_type.lower().endswith('-c') else ESM2_TARGET_MODULES

@dataclass
class LoRAClassifierConfig(TokenClassifierConfig):
    rank : int = 8
    alpha : int = 16
    lora_dropout : float = 0.05
    target_modules : list[str] = field(default_factory= lambda : ESM2_TARGET_MODULES)
    # Rank stabilized LoRA scales the adapters by alpha/sqrt(rank) instead of alpha/rank
    use_rslora : bool = True

class LoRAClassifier(TokenClassifier):
    """
    Token classifier with LoRA adapters (via peft) injected into the base model.

    The base model weights stay frozen, only the adapters and the classification head
    are trained. Unlike the finetuning classifiers, no base model layers are unfrozen.
    """
    def __init__(self, config: LoRAClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        base_hidden_size = base_model.config.hidden_size if 'hidden_size' in base_model.config else base_model.config.d_model
        self.classifier = torch.nn.Linear(base_hidden_size, config.n_labels)
        self.init_weights(self.classifier)

        # Base dropout is set before wrapping, so that the adapters keep their own lora_dropout
        self.set_base_dropout(config.dropout_rate)
        self.base = get_peft_model(self.base, self.build_peft_config(config))
        self.base.print_trainable_parameters()

    def build_peft_config(self, config : LoRAClassifierConfig):
        return LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            use_rslora=config.use_rslora,
            bias='none',
        )

    def set_base_dropout(self, prob):
        """
        Sets the dropout probability in the base model, leaving the dropout owned by the
        LoRA adapters untouched.
        """
        for name, module in self.base.named_modules():
            if isinstance(module, torch.nn.Dropout) and 'lora_dropout' not in name:
                module.p = prob

    def set_lora_requires_grad(self, req_grad_value : bool):
        for name, param in self.base.named_parameters():
            if 'lora_' in name:
                param.requires_grad = req_grad_value

    def freeze_phase(self):
        """
        Trains the classification head only, with the adapters frozen as well.
        """
        self.set_base_requires_grad(False)
        self.set_base_dropout(0)

    def unfreeze_phase(self):
        """
        Trains the adapters and the classification head. The base model stays frozen.
        """
        self.set_base_requires_grad(False)
        self.set_lora_requires_grad(True)
        self.set_base_dropout(self.config.dropout_rate)


def create_model(args):
    esm, tokenizer = get_esm(args.type)
    targets = ast.literal_eval(args.lora_targets) if args.lora_targets else default_target_modules(args.type)
    config = LoRAClassifierConfig(n_labels=1, loss=create_loss(args), base_type=args.type, dropout_rate=args.dropout,
                                  rank=args.lora_rank, alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                  target_modules=targets, use_rslora=not args.no_rslora)
    model = LoRAClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    if args.hpc:
        torch.set_float32_matmul_precision('high')
    run_training(args, create_model)

def add_arguments(parser):
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha (scaling factor)')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout probability of the LoRA layers')
    parser.add_argument('--lora_targets', type=str, default=None,
                        help='Base model modules to apply LoRA to. Defaults to the attention projections of the given model type.')
    parser.add_argument('--no_rslora', action='store_true', default=False,
                        help='Disable rank stabilized LoRA, scaling the adapters by alpha/rank instead of alpha/sqrt(rank).')
if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
