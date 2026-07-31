from classifiers import RNNTokenClassifier, RNNTokenClassiferConfig
from training import parser, create_loss, run_training
from utils import get_esm

def create_model(args):
    base, tokenizer = get_esm(args.type)
    config = RNNTokenClassiferConfig(1, create_loss(args))
    config.sr_dim = args.sr_dim
    classifier = RNNTokenClassifier(config, base)
    if not args.lora:
        classifier.set_base_requires_grad(False)
        
    return classifier, tokenizer


def add_arguments(parser):
    parser.add_argument('--sr_dim', default=None, type=int, help='If set, will use a CNN sequence representation with the given dimension')

if __name__ == "__main__":
    add_arguments(parser)
    args = parser.parse_args()
    run_training(args, create_model)