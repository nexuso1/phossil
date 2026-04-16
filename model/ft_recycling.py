import ast
from esm_train import get_esm
from training import  parser, create_loss, run_training
from classifiers import RecyclingFinetuningClassifier, RecyclingFinetuningClassifierConfig

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    indices = ast.literal_eval(args.indices)
    config = RecyclingFinetuningClassifierConfig(n_labels=1,loss=create_loss(args), unfreeze_indices=indices,
                                                 base_type=args.type, n_steps=args.n_steps)
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
