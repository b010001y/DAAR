import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel


class BERTSentenceEncoderPrompt(nn.Module):
    def __init__(self, config,ckptpath=None):
        nn.Module.__init__(self)
        if ckptpath != None:
            ckpt = torch.load(ckptpath)
            self.bert = BertModel.from_pretrained(config["pretrained_model"],state_dict=ckpt["bert-base"])
        else:
            self.bert = BertModel.from_pretrained(config["pretrained_model"])
        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        self.output_size = 768
    def forward(self, inputs, mask, mask_pos):
        outputs = self.bert(inputs, attention_mask=mask)
        tensor_range = torch.arange(inputs.size()[0])
        return outputs[0][tensor_range, mask_pos]


