import torch
import torch.nn as nn
from dataprocess import get_data_loader_bert_prompt
import torch.optim as optim
import numpy as np
import copy

def train_model_and_refresh_memory(config, modelforbase, refresh, train_data, current_epoch, memory_capacity, steps, current_proto):
    memory_capacity = memory_capacity[steps-1]
    trainning_data = copy.deepcopy(train_data)

    if config['refresh_mode'] == 'centralized_refresh':
        pos = int(config['train_epoch'] - 2)
        if current_epoch == pos:
            for i in range(refresh.review_epoch):
                capacity = int(len(memory_capacity) / 2)
                memory = memory_capacity[i*capacity:(i+1)*capacity]
                # train_model_universal(config, modelforbase, memory)
                train_prototypes(config, modelforbase, memory, current_proto, threshold=0.1)

        train_prototypes(config, modelforbase, trainning_data, current_proto, threshold=0.1)
        # train_model_universal(config, modelforbase, trainning_data)
    elif config['refresh_mode'] == 'distributed_refresh':
        p = int(refresh.parts_of_memory)
        if current_epoch < (config['train_epoch']-1):
            each_memory = memory_capacity[current_epoch*p:(current_epoch+1)*p]
        else: ## last epoch
            each_memory = memory_capacity[current_epoch*p:]
        trainning_data.extend(each_memory)
        train_prototypes(config, modelforbase, trainning_data, current_proto, threshold=0.1)
        # train_model_universal(config, modelforbase, trainning_data)
    else:
        interval = refresh.epoch_interval
        content = int(refresh.review_memory)
        if (current_epoch+1) % interval == 0:
            x = int((current_epoch+1) / interval)
            memory = memory_capacity[(x-1)*content:x*content]
            trainning_data.extend(memory)
        train_prototypes(config, modelforbase, trainning_data, current_proto, threshold=0.1)
        # train_model_universal(config, modelforbase, trainning_data)


def train_model_universal(config, modelforbase, trainning_data):
    data_loader = get_data_loader_bert_prompt(config, trainning_data, batch_size=config['batch_size_per_step'])
    modelforbase.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelforbase.parameters(), config['learning_rate'])

    loss_all = 0
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
    typelabels, masks, mask_pos) in enumerate(data_loader):
        modelforbase.zero_grad()
        labels = labels.to(config['device'])

        sentences = sentences.to(config['device'])
        masks = masks.to(config['device'])
        mask_pos = mask_pos.to(config['device'])

        logits, rep = modelforbase(sentences, masks, mask_pos)

        loss = criterion(logits, labels)

        loss.backward()
        loss_all += loss

        torch.nn.utils.clip_grad_norm_(modelforbase.parameters(), config['max_grad_norm'])  # cxd
        optimizer.step()
    print('loss: %.6f' % loss_all.item())

'''
here perceptron.model == basemodel 
of course, we can add the dynamic choosing mode of refresh
'''
def train_perceptron(perceptron, trainning_data):
    pass


def eval_model_each_epoch(config, modelforbase, test_data):
    modelforbase.eval()
    all_num = 0
    correct_num = 0

    test_dataloader = get_data_loader_bert_prompt(config, test_data, shuffle=False, batch_size=30)
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels, masks, mask_pos) in enumerate(test_dataloader):
        modelforbase.zero_grad()
        labels = labels.to(config['device'])

        sentences = sentences.to(config['device'])
        masks = masks.to(config['device'])
        mask_pos = mask_pos.to(config['device'])

        logits, rep = modelforbase(sentences, masks, mask_pos)

        all_num += labels.shape[0]
        predict_res = torch.argmax(logits, dim=1)
        predict_res = predict_res.view(-1, 1)
        labels = labels.view(-1, 1)
        correct_predictions = torch.eq(labels, predict_res)
        num_correct = torch.sum(correct_predictions).item()
        correct_num += num_correct

        # for index, logit in enumerate(logits):
        #     predict_res = torch.argmax(logit)
        #     true_res = labels[index]
    acc = correct_num/all_num
    return acc



def eval_model_in_seen_data(config, modelforbase, test_all_data):
    modelforbase.eval()
    all_num = 0
    correct_num = 0
    test_dataloader = get_data_loader_bert_prompt(config, test_all_data, shuffle=False, batch_size=30)
    for step, (
    labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
    typelabels, masks, mask_pos) in enumerate(test_dataloader):
        modelforbase.zero_grad()
        labels = labels.to(config['device'])

        sentences = sentences.to(config['device'])
        masks = masks.to(config['device'])
        mask_pos = mask_pos.to(config['device'])

        logits, rep = modelforbase(sentences, masks, mask_pos)

        all_num += labels.shape[0]
        predict_res = torch.argmax(logits, dim=1)
        predict_res = predict_res.view(-1, 1)
        labels = labels.view(-1, 1)
        correct_predictions = torch.eq(labels, predict_res)
        num_correct = torch.sum(correct_predictions).item()
        correct_num += num_correct

        # for index, logit in enumerate(logits):
        #     predict_res = torch.argmax(logit)
        #     true_res = labels[index]
    acc = correct_num / all_num
    return acc

def get_protos_by_proto_set(config, model, proto_set):
    rangeset = [0]
    proto_set_cat = []
    for i in proto_set:
        proto_set_cat += i
        rangeset.append(rangeset[-1] + len(i))
    data_loader = get_data_loader_bert_prompt(config, proto_set_cat, False, False)
    features = []
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels, masks, mask_pos) in enumerate(data_loader):
        sentences = sentences.to(config['device']) # this sentence represent the relation tokens
        masks = masks.to(config['device'])
        mask_pos = mask_pos.to(config['device'])
        feature = model.get_feature(sentences, masks, mask_pos) # in the first epoch, other is fake
        features.append(feature)
    features = np.concatenate(features)

    protos = []
    for i in range(len(proto_set)):
        protos.append(torch.tensor(features[rangeset[i]:rangeset[i+1],:].mean(0, keepdims = True)))
        ## there obtain the mean of the feature of index of rangeset[i] to rangeset[i+1] by the column
    protos = torch.cat(protos, dim=0) ## in the dim == 0
    return protos

def eval_model(config, basemodel, test_set):
    # print("One eval")
    # print("test data num is:\t",len(test_set))
    basemodel.eval()

    test_dataloader = get_data_loader_bert_prompt(config, test_set, shuffle=False, batch_size=30)
    allnum= 0.0
    correctnum = 0.0

    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels, masks, mask_pos) in enumerate(test_dataloader):
        # print(labels[0])
        # print(neg_labels[0])
        sentences = sentences.to(config['device'])
        masks = masks.to(config['device'])
        mask_pos = mask_pos.to(config['device'])
        logits, rep = basemodel(sentences, masks, mask_pos)
        # calculate the distance between rep and prototype.
        # The distance is regarded as the score, which belongs to the prototype with the smallest distance
        distances = basemodel.get_mem_feature(rep)
        short_logits = distances
        # print(short_logits[0])
        # print(logits.shape)  ## 30*81
        # print(short_logits.shape) ## 30*81
        # print(labels.shape) ## 30*81
        # print(labels)
        # print(neg_labels)
        # allnum += labels.shape[0]
        # short_logits = torch.from_numpy(short_logits)
        # short_logits = short_logits.to(config['device'])
        # predict_res = torch.argmax(short_logits, dim=1)
        # predict_res = predict_res.view(-1, 1)
        # labels = labels.view(-1, 1)
        # labels = labels.to(config['device'])
        # correct_predictions = torch.eq(labels, predict_res)
        # num_correct = torch.sum(correct_predictions).item()
        # correctnum += num_correct
        for index, logit in enumerate(logits): # 30*81, 30 batchsize, 81 class_count
            # n-th testdata score, 1*81
            score = short_logits[index]  # logits[index] + short_logits[index] + long_logits[index]
            allnum += 1.0
            ## predict score
            golden_score = score[labels[index]] #label: 0-80,
            # print(golden_score)
            # print(max(score))
            max_neg_score = -2147483647.0
            ## find the max neg_label score
            for i in neg_labels[index]:  # range(num_class):
                if (i != labels[index]) and (score[i] > max_neg_score):
                    max_neg_score = score[i]
            ## calculate the correct number
            if golden_score > max_neg_score:
                correctnum += 1

    acc = correctnum / allnum
    # print(acc)
    basemodel.train()
    return acc

def train_prototypes(config, model, train_set, current_proto, threshold=0.2):
    data_loader = get_data_loader_bert_prompt(config, train_set, batch_size=config['batch_size_per_step'])
    model.train()
    criterion = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()
    softmax = nn.Softmax(dim=0)
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    model.set_memorized_prototypes_midproto(current_proto)

    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
    typelabels, masks, mask_pos) in enumerate(data_loader):
        model.zero_grad()
        labels = labels.to(config['device'])
        sentences = sentences.to(config['device'])
        masks = masks.to(config['device'])
        mask_pos = mask_pos.to(config['device'])
        logits, rep = model(sentences, masks, mask_pos)
        logits_proto = model.mem_forward(rep)
        # fprint(logits_proto, labels)
        loss1 = criterion(logits_proto, labels)
        # print(loss1)
        loss2 = torch.tensor(0.0).to(config['device'])
        for index, logit in enumerate(logits):  # batch_size*num_class
            score = logits_proto[index]
            maxscore = score[labels[index]]
            size = score.shape[0]
            maxsecondmax = [maxscore]
            secondmax = -100000
            for j in range(size):
                if j != labels[index] and score[j] > secondmax:
                    secondmax = score[j]
            maxsecondmax.append(secondmax)
            for j in range(size):
                if j != labels[index] and maxscore - score[j] < threshold:
                    maxsecondmax.append(score[j])  ## maxsecondmax == most similar pro
            maxsecond = torch.stack(maxsecondmax, 0)
            maxsecond = torch.unsqueeze(maxsecond, 0)
            la = torch.tensor([0]).to(config['device'])  ## why is zero?
            loss2 += criterion(maxsecond, la)
        loss2 /= logits.shape[0]

        loss3 = torch.tensor(0.0).to(config['device'])
        for index, logit in enumerate(logits):
            preindex = labels[index]
            if preindex in model.haveseenrelations:
                loss3 += mseloss(softmax(rep[index]), softmax(model.prototypes[preindex]))
        loss3 /= logits.shape[0]

        loss4 = torch.tensor(0.0).to(config['device'])
        for index, logit in enumerate(logits):
            preindex = labels[index]
            if preindex in model.haveseenrelations:
                best_distrbution = model.mem_forward_update(rep[index].view(1, -1), model.bestproto)
                current_distrbution = model.mem_forward_update(model.prototypes[preindex].view(1, -1), model.bestproto)
                loss4 += mseloss(best_distrbution, current_distrbution)
        loss4 /= logits.shape[0]

        # print(loss3)
        # print(loss4)

        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()