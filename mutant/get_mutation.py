import copy
import pickle
import random, os
import numpy as np
import copy


def filter_mutants(testcase_name_mutant_id_set, mutant_id_testcase_set, logger):

    testcase_mutant_matrix = np.zeros((len(testcase_name_mutant_id_set), len(mutant_id_testcase_set)))
    mutant_id_index = {}
    index = 0
    mutant_id_list = list(mutant_id_testcase_set.keys())
    for mutant_id in mutant_id_list:
        mutant_id_index[mutant_id] = index
        index += 1

    testcase_name_index = {}
    index = 0
    testcase_name_list = list(testcase_name_mutant_id_set.keys())
    for testcase_name in testcase_name_list:
        testcase_name_index[testcase_name] = index
        index += 1

    remove_live_mutant_id = set()
    remove_mutant_id_killed_by_test_without_logs = set()
    logger.info("Build matrix")
    for mutant_id, testcase_name_set in mutant_id_testcase_set.items():
        temp = np.zeros(len(testcase_name_mutant_id_set))
        for testcase_name in testcase_name_set:
            if testcase_name in testcase_name_index:
                temp[testcase_name_index[testcase_name]] = 1
        if sum(temp) == 0:
            remove_live_mutant_id.add(mutant_id)
        testcase_mutant_matrix[:, mutant_id_index[mutant_id]] = temp
    
    logger.info('The mutants which were killed by the testcases without log: %d' %
                len(remove_mutant_id_killed_by_test_without_logs))
    if len(remove_live_mutant_id) > 0:
        logger.info("Error! Exist live mutants")
        exit(-1)
    else:
        logger.info("No live mutants")

    available_mutant_id_list = copy.deepcopy(mutant_id_list)
    logger.info("Identify duplicate mutants!")
    remove_duplicate_mutant_id_index_flag = np.zeros(len(mutant_id_index))
    same_mutants_index = {}
    for index, mutant_id_h in enumerate(available_mutant_id_list):
        h = mutant_id_index[mutant_id_h]  # h is index in matrix
        if remove_duplicate_mutant_id_index_flag[h] == 0:
            x_vector = testcase_mutant_matrix[:, h]
            same_mutants_index[h] = set()
            for mutant_id_g in available_mutant_id_list[index+1:]:
                g = mutant_id_index[mutant_id_g]  # g is index in matrix
                if remove_duplicate_mutant_id_index_flag[g] == 0:
                    y_vector = testcase_mutant_matrix[:, g]
                    if (x_vector == y_vector).all():
                        same_mutants_index[h].add(g)
                        remove_duplicate_mutant_id_index_flag[g] = 1
    for d in range(len(remove_duplicate_mutant_id_index_flag)):  # d is index in matrix
        if remove_duplicate_mutant_id_index_flag[d] == 1:
            available_mutant_id_list.remove(mutant_id_list[d])
    logger.info("Remaining %d mutants." % len(available_mutant_id_list))

    return available_mutant_id_list


def select_limited_mutants(new_selected_mutant_id_list, group_size, logger):
    mutant_index_list = list(range(len(new_selected_mutant_id_list)))  # index in new_selected_mutant_id_list
    random.shuffle(mutant_index_list)
    group_num = int(len(new_selected_mutant_id_list) / group_size)
    group_num = min(group_num, 100)
    result_mutant_id_matrix = np.zeros((group_num, group_size))
    logger.info("Randomly select %d mutants from %d" % (group_num * group_size, len(new_selected_mutant_id_list)))
    shuffle_mutant_index_list = mutant_index_list[:group_num * group_size]
    for i in range(group_num):
        temp_mutant_id = []
        for j in range(i * group_size, (i + 1) * group_size):
            temp_mutant_id.append(new_selected_mutant_id_list[shuffle_mutant_index_list[j]])
        result_mutant_id_matrix[i] = temp_mutant_id
    
    return result_mutant_id_matrix

