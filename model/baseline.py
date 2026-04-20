import ast
from esm_train import get_esm
from training import  parser, create_loss, run_training
from classifiers import BaselineClassifier, TokenClassifierConfig

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
