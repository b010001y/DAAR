import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import json
import copy

from encode import BERTSentenceEncoderPrompt
from dataprocess import data_sampler_bert_prompt_deal_first_task, get_data_loader_bert_prompt
from model import proto_softmax_layer_bert_prompt
from util import set_seed
from core import Perceptron, Controller, Refresh
from function import train_perceptron, train_model_and_refresh_memory, \
    eval_model_each_epoch, eval_model_in_seen_data, train_model_universal, \
    get_protos_by_proto_set, eval_model, train_prototypes


'''
1. config the setting
2. set the template
3. obtained the data and process it
4. load the model
5. for loop:
6.   set the different seed to obtain the final result
7.   set the mem_set = {}, set the proto_memory = fakeContent
8.   for loop:
9.      pull the data, include current relations and training data and seen data
10.     in the first step, get the fake prototype, in other step, get the last prototype from prototype_memory
11.     sample pos/neg instances for mem_set and prototype_memory.
12.     get the new prototype and train the model for classify the relations and update the prototype
13.     select the best instance storing in mem_set and proto_memory and storing negative instance in memset
14.     get the new prototype and train the model for classify the relations and update the prototype, use Lclass loss
15.     get the new prototype and use the mem_set to update the model of classifying the relations and update the prototype, use Lcon loss
16.     get the new prototype and save the best prototype, record the current relations into seen relations
17.     eval the average model auc and eval the whole model auc
18.  obtain the all_data average auc and  all_data whole auc
19 output the final results
'''

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
    ## dram controller, dram perceptron, dram refresh module, tREFI(refresh cycle) tRFC(memory capacity)
    # todo: 1. config the setting, include device, batch_size, file_path etc.
    # 2. due to use the prompt learning, we need to set the prompt template
    # 3. obtained the data and process it to the format which we want
    # 4. load the model which is being written
    # 5. for loop:
    # 6.    preceptron gets the current state and sent to the controller
    # 7.    controller set the tREFI and tRFC by the current state
    # 8.    get the memory and new_training data
    # 9.    for loop:
    # 10.       refresh module execute the corresbonding operations by the tRFI and tRFC
    # 11.       train some model
    # 12.   eval the model
    #
    #
    #
    # we think the distributed refresh(refresh the part of all seen relations memory in each epoch) will underfit the model beacause the memory will distract the model
    # and centralized refresh(refresh the all seen relation memory in continue epochs) will overfit the model because the a large amount of historical knowledge is reviewed by the model at once
    # so the asynchronous refresh(refresh the part of all seen relation memory every few epochs) will get the balance of studying and forgetting

    # todo: preceptron
    # 1. current state: memory capacity, seen relation, current pos, refresh cycle
    # 2. train a model as the perceptron to eval the history relations
    # 3. set the memory capacity and refresh cycle by the effect of each seen relation
    # 4. sent the all_seen_relation_effect_distribution to controller

    # preceptron can be a classification to choose the refresh mode

    # todo: controller
    # 1. current state: memory capacity, seen relation, current pos, refresh cycle
    # 2. expand the capacity of memory by the seen relation
    # 3. memory capacity: softmax(all_seen_relation_effect_distribution) and normalized to get the memory assign proportion
    # 4. refresh cycle: for distributed refresh, setting the revolutions(zhuan shu) of repeat
    # 5. for centralized refresh, set the epochs of repeat
    # 6. for asynchronous refresh, set the the frequency of replaying alterable memory
    # tRFC: a vector records the memory assign proportion. 1*len(seen relations)
    # tREFI: for distributed refresh, it is the number about the revolution(zhuan shu) of repeat
    # for centralized refresh, it is the number about epoch of repeat
    # for asynchronous refresh, it include frequency and the size of memory for each refresh
    # we need to sure the three pattern train the same data
    # about the specific number of tREFI, it can be a decimal and more detailed should be consider

    # todo: refresh module
    # 1. recieve the tRFC and tREFI and match the refresh pattern
    # 2. include prototype and classify relation model
    # 3. take other methods

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

            # if steps < 3:
            #     config['refresh_mode'] = 'distributed_refresh'
            # else:
            #     config['refresh_mode'] = 'asynchronous_refresh'
            #     train_model_universal(config, modelforbase, current_train_data)
            #     currentalltest = []
            #     for mm in range(len(test_data)):
            #         currentalltest.extend(test_data[mm])
            #     acc = eval_model_each_epoch(config, modelforbase, currentalltest)
            #     if acc > best_acc:
            #         best_acc = acc
            # print("step:\t", steps, "\taccuracy_whole:\t", best_acc)
            # results = [eval_model_each_epoch(config, modelforbase, item) for item in test_data]
            # # res = eval_model_in_seen_data(config, modelforbase, test_all_data)
            # print("step:\t", steps, "\taccuracy_average:\t", results)

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

            # if steps < 3:
            #     config['refresh_mode'] = 'distributed_refresh'
            # else:
            #     config['refresh_mode'] = 'asynchronous_refresh'
            #     currentalltest = []
            #     for mm in range(len(test_data)):
            #         currentalltest.extend(test_data[mm])
            #
            #     acc = eval_model_each_epoch(config, modelforbase, currentalltest)
            #     if acc > best_acc:
            #         best_acc = acc
            # print("step:\t",steps,"\taccuracy_whole:\t", best_acc)
            # results = [eval_model_each_epoch(config, modelforbase, item) for item in test_data]
            # # res = eval_model_in_seen_data(config, modelforbase, test_all_data)
            # print("step:\t",steps,"\taccuracy_average:\t", results)

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
