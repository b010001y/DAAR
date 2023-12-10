import torch
import torch.nn as nn
import os
import json
import numpy as np


class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def get_parameters(self, mode = "numpy", param_dict = None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()


class proto_softmax_layer_bert_prompt(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __distance__(self, rep, rel):
        ## get the norm by the row. it is convenient for normalized the repensentation and calculate the distance
        rep_norm = rep / rep.norm(dim=1)[:, None]
        rel_norm = rel / rel.norm(dim=1)[:, None]
        res = torch.mm(rep_norm, rel_norm.transpose(0,1))
        ## the function of mm is to calculate the inner product that can measure the distance
        # res is a bach_size*relation_num adjacent matrix
        return res

    def __init__(self, sentence_encoder, num_class, id2rel, drop=0, config=None, rate=1.0, flag=-1):
        super(proto_softmax_layer_bert_prompt, self).__init__()

        self.config = config
        self.flag = flag
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.output_size
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias=False)
        self.drop = nn.Dropout(drop)
        self.id2rel = id2rel
        self.rel2id = {}
        for id, rel in id2rel.items():
            self.rel2id[rel] = id
        self.bestproto_list = []  # cxd
        self.bestproto = None  # cxd
        self.haveseenrelations = []  # cxd
        # self.haveseenrelations_process = []#cxd
        # self.currentrelations = []#cxd
        self.bestembedding = [[] for i in range(num_class)]

    ## get memorized best prototype in having be seen realtions
    def set_memorized_prototypes_midproto(self, protos):
        self.prototypes = protos.detach().to(self.config['device'])
        if self.bestproto != None:
            self.prototypes[self.haveseenrelations] = self.bestproto  # cxd

    # def set_memorized_prototypes_no_ce(self, protos):
    #     self.prototypes_no_ce = protos.detach().to(self.config['device'])
    #     if self.bestproto != None:
    #         self.prototypes_no_ce[self.haveseenrelations] = self.bestproto#cxd

    # def set_memorized_prototypes(self, protos):
    #     self.prototypes = protos.detach().to(self.config['device'])

    def get_feature(self, sentences, mask, mask_pos):
        rep = self.sentence_encoder(sentences, mask, mask_pos)
        return rep.cpu().data.numpy()

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis.cpu().data.numpy()

    def forward(self, sentences, mask, mask_pos):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(sentences, mask, mask_pos)  # (B, H)
        repd = self.drop(rep)
        logits = self.fc(repd)
        return logits, rep

    def mem_forward(self, rep):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem

    def mem_forward_update(self, rep, current_proto):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        dis_mem = self.__distance__(rep, current_proto)
        return dis_mem

    def save_bestproto(self, currentrelations):  # cxd
        self.haveseenrelations.extend(currentrelations)
        # self.haveseenrelations_process.append(self.haveseenrelations)
        # self.currentrelations.append(currentrelations)
        self.bestproto_list.append(self.prototypes[currentrelations])
        self.bestproto = torch.cat(self.bestproto_list, 0)

    # def save_memory_embedding(self, thisrel, save_embedding):
    #     save_embedding = save_embedding.detach().to(self.config['device'])
    #     self.bestembedding[thisrel].append(save_embedding)