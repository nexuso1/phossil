# This is the main entrypoint for running UniPTM-based models

from token_classifier_base import TokenClassifier, TokenClassifierConfig
from utils import get_esm
from training import run_training, create_loss, parser
import torch

class UniPTM(TokenClassifier):
    def __init__(self, config, base, emb_size, num_heads, num_layers, hidden_size, dropout_rate, pos_weight=None):
        super(UniPTM, self).__init__(base_model=base, config=config)
        self.cnn = torch.nn.Conv1d(in_channels=emb_size, out_channels=256, kernel_size=31, padding=15) 
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.transformer, num_layers=num_layers)
        self.fc1 = torch.nn.Linear(256, hidden_size)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.fc3 = torch.nn.Linear(hidden_size, 1)
        self.pos_weight = pos_weight

        self.base = base
    
    def forward(self, input_ids, attention_mask, **kwargs):
        base_out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        emb = base_out[0]
        emb = emb.transpose(1, 2)  
        emb = self.cnn(emb)
        emb = emb.transpose(1, 2)  
        x = self.encoder(emb)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x, base_out
    
    def train_predict(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor = None, return_dict=False, **kwargs):
        self.train()
        out = self(input_ids, attention_mask)[0]
        return self.weighted_BCEloss(labels != -1, labels, torch.sigmoid(out)), out 

    def weighted_BCEloss(self, mask, labels, outputs):
        mask = mask.squeeze(0).bool()
        true_y = labels.squeeze(0)[mask].float()
        pred_y = outputs.squeeze(0)[mask].squeeze(-1)
        weights = torch.ones_like(true_y)  
        if self.pos_weight is not None:
            weights[true_y == 1] = self.pos_weight  
        loss = torch.nn.functional.binary_cross_entropy(pred_y, true_y, weight=weights)
        return loss

    def BCEloss(self, batch, outputs):
        mask = batch['mask']
        mask = mask.squeeze(0).bool()
        true_y = batch['label'].squeeze(0)[mask].float()
        pred_y = outputs.squeeze(0)[mask].squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(pred_y, true_y)
        return loss

def create_model(args):
    conf = TokenClassifierConfig(1, loss = create_loss(args), base_type=args.type) # ignored
    base, tokenizer = get_esm(conf.base_type)
    classifier = UniPTM(conf, base, 1280, 8, 1, 128, 0.5, 3)
    classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)