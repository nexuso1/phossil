import torch
from base_model import BaseModel, BaseModelConfig
from dataclasses import dataclass

@dataclass
class LMModelConfig(BaseModelConfig):
    base_type : str = '650M'
    dropout : float = 0

@dataclass
class LMModelOuptut:
    logits : torch.Tensor
    base_output : dict

class LMModel(BaseModel):
    def __init__(self, config, base : torch.nn.Module):
        super().__init__(config)
        self.base = base
        self.mask_token_id = 32
        self.lm_loss = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

    def compute_loss(self, labels, attention_mask, outputs, orig_ids=None, input_ids=None, **kwargs):
        logits, base_output = outputs
        lm_logits = base_output.logits

        if attention_mask is not None:
            active_loss = labels.view(-1) != self.config.ignore_index
            active_logits = logits.reshape(-1, 1)
            valid_logits = active_logits[active_loss].flatten()
            valid_labels = labels.view(-1)[active_loss]
            loss = self.loss(valid_logits, valid_labels)

            if orig_ids is not None:
                mask = input_ids == self.mask_token_id
                valid_lm_logits = lm_logits[mask]
                
                loss += self.lm_loss(valid_lm_logits, orig_ids[mask])

        else:
            loss = self.loss(logits.view(-1, 1), labels.view(-1))
            if orig_ids is not None:
                loss += self.lm_loss(lm_logits.view(-1), orig_ids.view(-1))

        return loss
    
    def predict(self, input_ids, attention_mask, labels=None, return_dict=False, *args, **kwargs):
        """
        Prediction in eval mode.
        Outputs are the final classification logits as a Tensor.
        """
        self.eval()
        with torch.no_grad():
            if labels is not None:
                logits, outputs = self(input_ids, attention_mask, *args, **kwargs)
                loss = self.compute_loss(labels, attention_mask, (logits, outputs), input_ids=input_ids, **kwargs)
            else:
                return self(input_ids, attention_mask, *args, **kwargs)
            
            if return_dict:
                        return {
                            'loss' : loss,
                            'logits' : logits,
                            'outputs' : outputs
                        }
            
            return loss, logits

    def train_predict(self, input_ids, attention_mask, labels, *args, **kwargs):
        self.train()
        logits, base_output = self(input_ids, attention_mask, *args, **kwargs)
        loss = self.compute_loss(labels, attention_mask, (logits, base_output), input_ids=input_ids, **kwargs)
        return loss, logits
    
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        sequence_output = outputs.hidden_states[-1]

        return self.classifier(sequence_output), outputs