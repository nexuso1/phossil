# This file is the main way to run traninig on finetuning models
import ast
import torch

from utils import get_esm
from training import  parser, create_loss, run_training
from ft_selective import SelectiveFinetuningClassifier, SelectiveFinetuningClassifierConfig
from data_loading import parse_residues
from constants import esm_res_to_id_mapping
from dataclasses import dataclass, field
from torch.nn import Module

@dataclass
class MultiResidueFTConfig(SelectiveFinetuningClassifierConfig):
    residues : list[str] = field(default_factory= lambda : ['S', 'T'])
    # Residue -> base model token ID mapping. Defaults to the ESM2 vocabulary.
    residue_to_token_id : dict[str, int] | None = None

class MultiResidueFTClassifier(SelectiveFinetuningClassifier):
    """
    Selective finetuning classifier with a separate classification head per predicted residue.

    Every position is classified by the head belonging to its residue, so the output has the
    same shape as a standard token classifier. Positions holding a residue we do not predict
    are filled with 'unpredicted_logit' instead of being passed through a head.
    """
    unpredicted_logit = -1e4 # Logit given to positions no head is responsible for

    def __init__(self, config: MultiResidueFTConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        base_hidden_size = base_model.config.hidden_size if 'hidden_size' in base_model.config else base_model.config.d_model

        # Sorted, so that the head order does not depend on the order of the configured residues
        self.residues = sorted(set(config.residues))
        token_id_mapping = config.residue_to_token_id if config.residue_to_token_id is not None else esm_res_to_id_mapping
        self.residue_token_ids = [token_id_mapping[residue] for residue in self.residues]

        # The single head built by the parent is replaced by one head per residue
        del self.classifier
        self.classifiers = torch.nn.ModuleList([torch.nn.Linear(base_hidden_size, config.n_labels)
                                                for _ in self.residues])
        self.init_weights(self.classifiers)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass of the model. Does not make any changes to train/eval status, and does
        not calculate loss.

        Returns (logits, outputs of the base model).
        """
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        classifier_features = self.classifier_features(sequence_output, **kwargs)

        logits = torch.full((*classifier_features.shape[:-1], self.n_labels), self.unpredicted_logit,
                            dtype=classifier_features.dtype, device=classifier_features.device)
        for token_id, classifier in zip(self.residue_token_ids, self.classifiers):
            residue_mask = input_ids == token_id
            logits[residue_mask] = classifier(classifier_features[residue_mask])

        return logits, outputs

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    indices = ast.literal_eval(args.indices)
    residues = parse_residues(args.residues)
    config = MultiResidueFTConfig(n_labels=1,loss=create_loss(args), unfreeze_indices=indices,
                                                 base_type=args.type, dropout_rate=args.dropout, residues=residues,
                                                 residue_to_token_id={r : tokenizer.convert_tokens_to_ids(r) for r in residues})
    model = MultiResidueFTClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    run_training(args, create_model)

def add_arguments(parser):
    parser.add_argument('--indices', default="[-1]", help='Indices of base model layers to be unfrozen')

if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
