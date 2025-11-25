import torch
from transformers import AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL

class PIINerModel(torch.nn.Module):
    def __init__(self, model_name: str, quantize: bool = False):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        if quantize:
            self.quantize()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def quantize(self):
        # Apply dynamic quantization to Linear layers
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)

def create_model(model_name: str, quantize: bool = False):
    return PIINerModel(model_name, quantize)
