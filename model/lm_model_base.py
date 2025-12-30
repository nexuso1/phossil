import torch
from base_model import BaseModel, BaseModelConfig
from dataclasses import dataclass

@dataclass
class LMModelConfig(BaseModelConfig):
    base_type : str = '650M'
    dropout : float = 0

@dataclass
class LMModelInputs:
    input_ids : torch.Tensor
    lm_labels : torch.Tensor
    attention_mask : torch.Tensor = None
    labels : torch.Tensor = None

@dataclass
class LMModelOuptut:
    logits : torch.Tensor
    base_output = None

class LMModel(BaseModel):
    def __init__(self, config, base : torch.nn.Module):
        super().__init__(config)
        self.base = base
        self.lm_loss = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

    def compute_loss(self, inputs : LMModelInputs, outputs : LMModelOuptut, return_dict=False, **kwargs):
        logits = outputs.logits
        lm_logits = outputs.base_output.logits
        labels = inputs.labels

        if inputs.attention_mask is not None:
            active_loss = inputs.labels.view(-1) != self.config.ignore_index
            active_logits = logits.reshape(-1, self.n_labels)
            valid_logits = active_logits[active_loss].flatten()
            valid_lm_logits = lm_logits.reshape(-1, 1)[active_loss].flatten()
            loss = self.lm_loss(valid_lm_logits, inputs.lm_labels)
            valid_labels = labels.view(-1)[active_loss]
            loss += self.loss(valid_logits, valid_labels)

        else:
            loss = self.lm_loss(lm_logits.view(-1), inputs.lm_labels.view(-1))
            loss += self.loss(logits.view(-1, self.n_labels), labels.view(-1))

        if not return_dict:
            return loss, logits

        return {
            'loss' : loss,
            'logits' : logits,
            'hidden_states' : outputs.hidden_states,
            'attentions' : outputs.attentions,
            'outputs' : outputs
        } 
    
    def forward(self, inputs : LMModelInputs, **kwargs):
        outputs = self.base(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
        sequence_output = outputs.hidden_states[-1]

        return self.classifier(sequence_output), outputs