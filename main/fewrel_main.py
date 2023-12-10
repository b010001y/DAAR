import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import json
import copy

from model.encode import BERTSentenceEncoderPrompt
from process.dataprocess_tacred import data_sampler_bert_prompt_deal_first_task, get_data_loader_bert_prompt
from model.model import proto_softmax_layer_bert_prompt
from util.util import set_seed
from core.core import Perceptron, Controller, Refresh
from core.function import train_perceptron, train_model_and_refresh_memory, \
    eval_model_each_epoch, eval_model_in_seen_data, train_model_universal, \
    get_protos_by_proto_set, eval_model, train_prototypes

def select_prototype_memory(current_relations, trainning_data, proto_memory):
    divide_train_set = {}
    for relation in current_relations:
        divide_train_set[relation] = []  ##int
    for data in trainning_data:
        divide_train_set[data[0]].append(data)
    rela_num = len(current_relations)
    mem_set = {}
    for i in range(0, rela_num):
        thisrel = current_relations[i]
        thisdataset = divide_train_set[thisrel]
        mem_set[thisrel] = []
        # print(len(thisdataset))
        for i in range(len(thisdataset)):
            instance = thisdataset[i]
            ###change tylelabel
            instance[11] = 3
            ###add to mem data
            mem_set[thisrel].append(instance)
            proto_memory[thisrel].append(instance)
    return proto_memory

def del_selected_train_data(current_relations, trainning_data, proto_memory):
    divide_train_set = {}
    for relation in current_relations:
        divide_train_set[relation] = []  ##int
    for data in trainning_data:
        divide_train_set[data[0]].append(data)
    del_length = len(divide_train_set[trainning_data[0][0]])
    for x in current_relations:
        del proto_memory[x][0:del_length]
        # del proto_memory[x][1:del_length+1]
    return proto_memory

def main():
    ### config the setting ###
    f = open('config_fewRel/config_fewrel_5and10.json.json', 'r')
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'])
    config['neg_sampling'] = False

    config["rel_cluster_label"] = "data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_" + "0.npy"
    config['training_file'] = "data/fewrel/CFRLdata_10_100_10_5/train_" + "0.txt"
    config['valid_file'] = "data/fewrel/CFRLdata_10_100_10_5/valid_" + "0.txt"
    config['test_file'] = "data/fewrel/CFRLdata_10_100_10_5/test_" + "0.txt"

    config['first_task_k-way'] = 10
    config['k-shot'] = 5

    # config["rel_cluster_label"] = "data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_" + "0.npy"
    # config['training_file'] = "data/tacred/CFRLdata_10_100_10_5/train_" + "0.txt"
    # config['valid_file'] = "data/tacred/CFRLdata_10_100_10_5/valid_" + "0.txt"
    # config['test_file'] = "data/tacred/CFRLdata_10_100_10_5/test_" + "0.txt"
    #
    # config['first_task_k-way'] = 5
    # config['k-shot'] = 5

    config['refresh_mode'] = 'asynchronous_refresh'
    # config['refresh_mode'] = 'centralized_refresh'
    # config['refresh_mode'] = 'distributed_refresh'
    print(config['refresh_mode'])

    config['train_epoch'] = 6
    ### config the setting ###

    ## set some important variable ##
    template = 'no use of prompt'
    if config['prompt'] == 'hard-complex':
        template = 'the relation between e1 and e2 is mask'
    elif config['prompt'] == 'hard-simple':
        template = 'e1 mask e2 .'
    print('Template: %s' % template)

    threshold = 0.1
    ## set some important variable ##

    ## initialize the encoder model ##
    encoderforbase = BERTSentenceEncoderPrompt(config)
    original_vocab_size = len(list(encoderforbase.tokenizer.get_vocab()))
    print('Vocab size: %d' % original_vocab_size)
    ## initialize the encoder model ##

    ## get data and process it ##
    sampler = data_sampler_bert_prompt_deal_first_task(config, encoderforbase.tokenizer, template)
    ## get data and process it ##

    ## load the refresh model and perceptron model ##
    modelforbase = proto_softmax_layer_bert_prompt(encoderforbase, num_class=len(sampler.id2rel), id2rel=sampler.id2rel,
                                                   drop=0, config=config)
    modelforbase = modelforbase.to(config['device'])

    refresh = Refresh(config['refresh_mode'])
    perceptron = Perceptron(config)
    ## load the refresh model and perceptron model ##

    controller = Controller()
    memory_capacity = []
    review_epoch = 0  ## design for centralized refresh
    parts_of_memory = 0  ## design for distributed refresh
    epoch_interval = 0  ## this two variables are designed for asynchronous refresh
    review_memory = 0
    selected_informative_samples = []
    proto_memory = []
    for i in range(len(sampler.id2rel)):
        proto_memory.append([sampler.id2rel_pattern[i]])

    set_seed(config, 3407)
    sampler.set_seed(3407)
    expand_num = 2

    ## for loop, add new relations
    for steps, (
    trainning_data, valid_data, test_data, test_all_data_splited, test_all_data, seen_relations, current_relations,
    _) in enumerate(sampler):
        current_train_data = trainning_data[steps]
        best_acc = 0

        proto_memory = select_prototype_memory(current_relations, current_train_data, proto_memory)
        current_proto = get_protos_by_proto_set(config, modelforbase, proto_memory)
        # modelforbase.set_memorized_prototypes_midproto(current_proto)
        # currentalltest = []
        # for mm in range(len(test_data)):
        #     currentalltest.extend(test_data[mm])
        # acc = eval_model(config, modelforbase, currentalltest)
        # if acc > best_acc:
        #     best_acc = acc
        # print("step:\t", steps, "\taccuracy_whole:\t", best_acc)
        # results = [eval_model(config, modelforbase, item) for item in test_data]
        # print("step:\t", steps, "\taccuracy_average:\t", results)
        # results_average = np.array(results).mean()
        # print("step:\t", steps, "\taccuracy_average:\t", results_average)
        if steps == 0:
            for i in range(config['train_epoch']):
                ## there seem not train hard neg but train the model use the training_data and use multi-loss to optimize
                train_prototypes(config, modelforbase, current_train_data, current_proto, threshold=threshold)
                # compute mean accuarcy
                currentalltest = []
                for mm in range(len(test_data)):
                    currentalltest.extend(test_data[mm])
                acc = eval_model(config, modelforbase, currentalltest)
                if acc > best_acc:
                    best_acc = acc
            print("step:\t", steps, "\taccuracy_whole:\t", best_acc)
            results = [eval_model(config, modelforbase, item) for item in test_data]
            print("step:\t", steps, "\taccuracy_average:\t", results)
            results_average = np.array(results).mean()
            print("step:\t", steps, "\taccuracy_average:\t", results_average)

            perceptron.update_model(modelforbase)
            # print(seen_relations[:-10])
            seen_relation_effect_distribution = perceptron.eval_seen_relation(config, seen_relations, test_all_data)
            expand_capacity = controller.expand_the_memory_capacity(seen_relations, memory_capacity, expand_num)
            TRFC = controller.set_TRFC(seen_relation_effect_distribution)
            selected_informative_sample = controller.select_informative_sample(config, modelforbase, current_train_data, 1, current_relations)
            selected_informative_samples.append(selected_informative_sample)
            controller.assign_memory(TRFC, selected_informative_samples, memory_capacity, expand_capacity, seen_relations, steps, proto_memory)
            ## it can be designed as dynamic number by the distance of last seen_relation_effect_distribution and current seen_relation_effect_distribution
            TREFI = controller.set_TREFI(config, seen_relation_effect_distribution, memory_capacity, steps)
            parts_of_memory, review_epoch, epoch_interval, review_memory = \
                controller.update_parameter_by_TREFI(TREFI, review_epoch, parts_of_memory, epoch_interval,
                                                     review_memory,
                                                     config, memory_capacity, steps)
            refresh.set_parameter(parts_of_memory, review_epoch, epoch_interval, review_memory)
        else:
            for i in range(config['train_epoch']):
                train_model_and_refresh_memory(config, modelforbase, refresh, current_train_data, i, memory_capacity, steps, current_proto)
                train_perceptron(perceptron, current_train_data)
                currentalltest = []
                for mm in range(len(test_data)):
                    currentalltest.extend(test_data[mm])
                acc = eval_model(config, modelforbase, currentalltest)
                if acc > best_acc:
                    best_acc = acc
            print("step:\t", steps, "\taccuracy_whole:\t", best_acc)
            results = [eval_model(config, modelforbase, item) for item in test_data]
            print("step:\t", steps, "\taccuracy_average:\t", results)
            results_average = np.array(results).mean()
            print("step:\t", steps, "\taccuracy_average:\t", results_average)

            perceptron.update_model(modelforbase)
            # print(seen_relations[:-10])
            seen_relation_effect_distribution = perceptron.eval_seen_relation(config, seen_relations, test_all_data)
            expand_capacity = controller.expand_the_memory_capacity(seen_relations, memory_capacity, expand_num)
            TRFC = controller.set_TRFC(seen_relation_effect_distribution)
            selected_informative_sample = controller.select_informative_sample(config, modelforbase, current_train_data, 1, current_relations)
            selected_informative_samples.append(selected_informative_sample)
            controller.assign_memory(TRFC, selected_informative_samples, memory_capacity, expand_capacity, seen_relations, steps, proto_memory)
            ## it can be designed as dynamic number by the distance of last seen_relation_effect_distribution and current seen_relation_effect_distribution
            TREFI = controller.set_TREFI(config, seen_relation_effect_distribution, memory_capacity, steps)
            parts_of_memory, review_epoch, epoch_interval, review_memory = \
                controller.update_parameter_by_TREFI(TREFI, review_epoch, parts_of_memory, epoch_interval,
                                                     review_memory,
                                                     config, memory_capacity, steps)
            refresh.set_parameter(parts_of_memory, review_epoch, epoch_interval, review_memory)

            # if type(TREFI) == int or type(TREFI) == float:
            #     if TREFI < 2:
            #         refresh.pattern = 'distributed refresh'
            #     elif TREFI > 2:
            #         refresh.pattern = 'centralized refresh'
            # else:
            #     refresh.pattern = 'asynchronous refresh'
            #
        proto_memory = del_selected_train_data(current_relations, current_train_data, proto_memory)


main()
