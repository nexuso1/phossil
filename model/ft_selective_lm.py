# This file is the main way to run traninig on finetuning models

import ast
from esm_train import get_esm
from training import  parser, create_loss, run_training
from classifiers import LMFinetuningClassifier, LMFinetuningClassifierConfig

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    indices = ast.literal_eval(args.indices)
    config = LMFinetuningClassifierConfig(loss=create_loss(args), unfreeze_indices=indices,
                                                 base_type=args.type)
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
