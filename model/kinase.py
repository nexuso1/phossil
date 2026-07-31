# This is the main entrypoint for running kinase-aware models

from classifiers import KinaseFTClassifier, KinaseFTClassifierConfig
from utils import get_esm
from ft_selective import add_arguments as add_ft_args
from training import run_training, parser, create_loss
from ast import literal_eval

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