import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataprocess import get_data_loader_bert_prompt
import numpy as np
from sklearn.cluster import KMeans

# todo: preceptron
# 1. current state: memory capacity, seen relation, current pos, refresh cycle
# 2. train a model as the perceptron to eval the history relations
# 3. set the memory capacity and refresh cycle by the effect of each seen relation
# 4. sent the all_seen_relation_effect_distribution to controller

class Perceptron():
    def __init__(self, config):
        self.config = config
        self.seen_relations = []
        self.model = None
        pass

    def eval_seen_relation(self, config, seen_relations, test_data_all):
        test_data = []
        for relation in seen_relations:
            for data in test_data_all:
                if data[0] == relation:
                    test_data.append(data)
        seen_relation_distribution = {}
        correct_count = {}
        class_count = {}
        for relation in seen_relations:
            seen_relation_distribution[relation] = 0
            correct_count[relation] = 0
            class_count[relation] = 0
        test_loader = get_data_loader_bert_prompt(config, test_data, shuffle=False, batch_size=30)
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid,
                rawtext, lengths, typelabels, masks, mask_pos) in enumerate(test_loader):
            labels = labels.to(config['device'])
            sentences = sentences.to(config['device'])
            masks = masks.to(config['device'])
            mask_pos = mask_pos.to(config['device'])

            # logits, rep = self.model(sentences, masks, mask_pos)
            # labels_np = labels.cpu().detach().numpy()
            # for label in labels_np:
            #     class_count[label] += 1
            # predict_res = torch.argmax(logits, dim=1)
            # predict_res = predict_res.view(-1, 1)
            # labels = labels.view(-1, 1)
            # equal_matrix = torch.eq(labels, predict_res)
            # # print(equal_matrix)
            # for idx, label in enumerate(labels_np):
            #     correct_count[label] += equal_matrix[idx]
            # # print(correct_count)
            logits, rep = self.model(sentences, masks, mask_pos)
            distance = self.model.get_mem_feature(rep)
            labels_np = labels.cpu().detach().numpy()
            for label in labels_np:
                class_count[label] += 1
            for index, logit in enumerate(logits):
                # n-th testdata score, 1*81
                score = distance[index]  # logits[index] + short_logits[index] + long_logits[index]
                ## predict score
                golden_score = score[labels_np[index]]
                # print(golden_score)
                # print(max(score))
                max_neg_score = -2147483647.0
                ## find the max neg_label score
                for i in neg_labels[index]:  # range(num_class):
                    if (i != labels[index]) and (score[i] > max_neg_score):
                        max_neg_score = score[i]
                ## calculate the correct number
                if golden_score > max_neg_score:
                    correct_count[labels_np[index]] += 1

        for relation in seen_relations:
            ratio = torch.tensor(correct_count[relation]/class_count[relation])
            seen_relation_distribution[relation] = ratio.unsqueeze(0)
        # print(seen_relation_distribution)
        return seen_relation_distribution

    def update_model(self, model):
        self.model = model



class Controller():
    def __init__(self):
        pass

    # a relation corresbond to x samples, now x  = 10
    def expand_the_memory_capacity(self, seen_relations, memory_capacity, expand_num):
        rel_num = len(seen_relations)
        expand_capacity = rel_num*expand_num
        expand_capacity_list = [None for _ in range(expand_capacity)]
        memory_capacity.append(expand_capacity_list)
        return expand_capacity


    def set_TRFC(self, seen_relation_effect_distribution):
        seen_distribution = []
        for k,v in seen_relation_effect_distribution.items():
            seen_distribution.append(seen_relation_effect_distribution[k])
        concat_tensor = torch.cat(seen_distribution, dim = 0)
        assign_ratio = (torch.ones_like(concat_tensor) - concat_tensor)*1
        tRFC = F.softmax(assign_ratio, dim=0)
        return tRFC # 1*10

    #get the min distance feature(informative samples) and dynamic assign it by the TRFC
    def assign_memory(self, TRFC, informative_sample, memory_capacity, expand_capacity, seen_relations, steps, proto_memory):
        # get len(TRFC)*expand_num
        # selected_informative_sample = []
        # for relation in seen_relations:
        #     temp = []
        #     for data in trainning_data:
        #         for item in data:
        #             if item[0] == relation:
        #                 temp.append(item)
        #     selected_informative_sample.append(temp)
        selected_informative_sample = []
        for data in informative_sample:
            for item in data:
                selected_informative_sample.append(item)
        add_num = []
        for ratio in TRFC:
            add_num.append(math.ceil(expand_capacity*ratio)) # at least 1
        ## here TRFC is dicimal
        for i, num in enumerate(add_num):
            if add_num[i] > len(selected_informative_sample[i]):
                add_num[i] = len(selected_informative_sample[i])
        i = len(add_num) - 1
        while sum(add_num) > expand_capacity:
            # while add_num[i] >= len(selected_informative_sample[i]):
            #     i += 1
            while add_num[i] == 1:
                i -= 1
                if i == -1:
                    i = len(add_num) - 1
            add_num[i] -= 1
            i -= 1
            if i == 0:
                i = len(add_num)-1

        # for x in range(len(seen_relations)):
        #     add_num[x] = int(expand_capacity/len(seen_relations))

        pos = 0
        for idx, num in enumerate(add_num):
            proto_idx = selected_informative_sample[idx][0][0]
            memory_capacity[steps][pos:pos+num] = selected_informative_sample[idx][:num]
            proto_memory[proto_idx].extend(selected_informative_sample[idx][:num])
            pos = pos + num


    def set_TREFI(self, config, seen_relation_effect_distribution, memory_capacity, steps):
        if config['refresh_mode'] == 'distributed_refresh':
            TREFI = 1
            pass
        elif config['refresh_mode'] == 'centralized_refresh':
            TREFI = 2
            pass
        else:
            TREFI = [None for _ in range(2)]
            TREFI[0] = 2
            TREFI[1] = len(memory_capacity[steps])/(config['train_epoch']/TREFI[0])
        return TREFI


    def update_parameter_by_TREFI(self, TREFI, review_epoch, parts_of_memory, epoch_interval, review_memory, config, memory_capacity, steps):
        if config['refresh_mode'] == 'distributed_refresh':
            parts_of_memory = len(memory_capacity[steps])*TREFI/config['train_epoch']
        elif config['refresh_mode'] == 'centralized_refresh':
            review_epoch = TREFI
        else:
            epoch_interval = TREFI[0]
            review_memory = TREFI[1]
            pass
        return parts_of_memory, review_epoch, epoch_interval, review_memory


    def select_informative_sample(self, config, model, current_train_data, num_sel_data, current_relations):
        divide_train_set = {}
        for relation in current_relations:
            divide_train_set[relation] = []  ##int
        for data in current_train_data:
            divide_train_set[data[0]].append(data)
        informative_sample = []
        rela_num = len(current_relations)
        for i in range(0, rela_num):
            thisrel = current_relations[i]
            thisdataset = divide_train_set[thisrel]
            data_loader = get_data_loader_bert_prompt(config, thisdataset, False, False)
            features = []
            for step, (
            labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,
            lengths,
            typelabels, masks, mask_pos) in enumerate(data_loader):
                sentences = sentences.to(config['device'])
                masks = masks.to(config['device'])
                mask_pos = mask_pos.to(config['device'])
                feature = model.get_feature(sentences, masks, mask_pos)
                features.append(feature)
            features = np.concatenate(features)
            num_clusters = min(num_sel_data, len(thisdataset))

            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
            distances = kmeans.fit_transform(features)
            sorted_list = sorted(zip(distances, thisdataset), key=lambda x: x[0])
            sorted_data = [data for _, data in sorted_list]
            informative_sample.append(sorted_data)

        return informative_sample

class Refresh():
    def __init__(self, pattern):
        self.pattern = pattern
        self.parts_of_memory = 0
        self.review_epoch = 0
        self.epoch_interval = 0
        self.review_memory = 0
        pass

    def set_parameter(self, parts_of_memory, review_epoch, epoch_interval, review_memory):
        self.parts_of_memory = parts_of_memory
        self.review_epoch = review_epoch
        self.epoch_interval = epoch_interval
        self.review_memory = review_memory
