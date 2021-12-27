from data.embedding import read_vec
from prioritize.prioritize_distance import get_distance_matrix
import pickle
import numpy as np


def max_distance_to_unsorted(distance_list):
    max_distance = float('-inf')
    max_unsorted_id = None
    for testcase_id, distance in distance_list.items():
        if distance > max_distance:
            max_distance = distance
            max_unsorted_id = testcase_id
    if max_unsorted_id is None:
        max_unsorted_id = list(distance_list.keys())[0]
    return max_unsorted_id


def select_first_testcase_arp(test_case_event_sequence, testcase_list):
    testcase_event_coverage_dict = {}
    for name, log_id_list in test_case_event_sequence.items():
        testcase_event_coverage_dict[name] = len(set(log_id_list))
    max_event_coverage_testcase = max(testcase_event_coverage_dict, key=testcase_event_coverage_dict.get)
    current_testcase_id = testcase_list.index(max_event_coverage_testcase)

    return current_testcase_id


def arp_results(test_case_event_sequence, testcase_list,
                dis_matrix, extend_name_list, logger):
    all_prioritize_results = []
    logger.info("Start to get all prioritize results with max-min arp")
    result_list_id = []
    unsorted_testcase_id_set = set()
    distance_list = {}  # min distance between each unsorted test case and test case in result_list_id
    for i in range(len(testcase_list)):
        unsorted_testcase_id_set.add(i)
        distance_list[i] = float('inf')

    while len(unsorted_testcase_id_set) > 1:
        if len(unsorted_testcase_id_set) == len(testcase_list):
            current_testcase_id = select_first_testcase_arp(test_case_event_sequence,
                                                            testcase_list)
        else:
            current_testcase_id = max_distance_to_unsorted(distance_list)
        result_list_id.append(current_testcase_id)
        unsorted_testcase_id_set.remove(current_testcase_id)
        distance_list.pop(current_testcase_id)
        for temp_testcase_id in unsorted_testcase_id_set:
            if dis_matrix[current_testcase_id, temp_testcase_id] < distance_list[temp_testcase_id]:
                distance_list[temp_testcase_id] = dis_matrix[current_testcase_id, temp_testcase_id]

    # add the last one
    assert len(unsorted_testcase_id_set) == 1
    for k in unsorted_testcase_id_set:
        result_list_id.append(k)
    for h in result_list_id:
        all_prioritize_results.append(testcase_list[h])

    if len(extend_name_list) > 0:
        all_prioritize_results.extend(extend_name_list)
    logger.info(all_prioritize_results)
    logger.info(len(all_prioritize_results))
    return all_prioritize_results


def adaptive_random_prioritization(test_case_event_sequence,
                                   distance_option,
                                   process_module_testcase_emb_file,
                                   all_name_without_log_list, logger):
    logger.info("Prioritize based on vector")
    testcase_vec = read_vec(process_module_testcase_emb_file)
    testcase_with_log_list = list(testcase_vec.keys())
    dis_matrix = get_distance_matrix(testcase_with_log_list, testcase_vec, distance_option, logger)
    all_prioritize_results = arp_results(test_case_event_sequence, testcase_with_log_list,
                                         dis_matrix, all_name_without_log_list, logger)
    return all_prioritize_results


def event_coverage_prioritization(process_module_testcase_emb_file, all_name_without_log_list,
                                  prioritization, logger):
    testcase_vec = read_vec(process_module_testcase_emb_file)
    testcase_with_log_list = list(testcase_vec.keys())
    all_prioritize_results = []

    if prioritization == "additional":
        standard_vec = np.zeros((1, list(testcase_vec.values())[0].shape[0]))
        for i in range(len(testcase_with_log_list)):
            event_coverage_dict = {}
            for testcase in testcase_vec:
                event_coverage_number = 0
                for index, event in enumerate(testcase_vec[testcase]):
                    if event == 1 and standard_vec[0][index] == 0:
                        event_coverage_number += 1
                event_coverage_dict[testcase] = event_coverage_number
            max_coverage_testcase = max(event_coverage_dict, key=event_coverage_dict.get)
            all_prioritize_results.append(max_coverage_testcase)
            for index, event in enumerate(testcase_vec[max_coverage_testcase]):
                if event == 1:
                    standard_vec[0][index] = 1
            testcase_vec.pop(max_coverage_testcase)
    else:
        event_coverage_dict = {}
        for testcase in testcase_vec:
            event_coverage_number = 0
            for index, event in enumerate(testcase_vec[testcase]):
                if event == 1:
                    event_coverage_number += 1
            event_coverage_dict[testcase] = event_coverage_number
        for testcase in sorted(event_coverage_dict.items(), key=lambda item: (item[1], item[0]), reverse=True):
            all_prioritize_results.append(testcase[0])
    if len(all_name_without_log_list) > 0:
        all_prioritize_results.extend(all_name_without_log_list)
    return all_prioritize_results


def record_prioritization_results(prioritization_result_file, prioritize_results_list, logger):
    with open(prioritization_result_file, 'w') as f:
        for testcase in prioritize_results_list:
            f.write(testcase + '\n')
    logger.info("Record prioritization results in %s" % prioritization_result_file)


def read_prioritization_results(prioritize_result_file):  # additional mutant coverage
    with open(prioritize_result_file, 'r', encoding="utf-8") as f:
        all_prioritize_results = f.read().splitlines()
    return all_prioritize_results


def ideal_prioritization(testcase_name_mutant_id_set, result_mutant_id_matrix,
                         ideal_prioritization_result_file, logger):  # additional mutant coverage
    logger.info("Create ideal prioritize results %s" % ideal_prioritization_result_file)
    all_prioritize_results = []
    available_mutant_id_set = set(result_mutant_id_matrix.flatten())

    remain_testcase = set(testcase_name_mutant_id_set.keys())
    while len(available_mutant_id_set) > 0:
        max_mutant_coverage_number = 0
        max_coverage_testcase = None
        for testcase in remain_testcase:
            mutant_coverage_number = len(available_mutant_id_set & testcase_name_mutant_id_set[testcase])
            if mutant_coverage_number >= max_mutant_coverage_number:
                max_mutant_coverage_number = mutant_coverage_number
                max_coverage_testcase = testcase
        if max_coverage_testcase:
            available_mutant_id_set = available_mutant_id_set - testcase_name_mutant_id_set[max_coverage_testcase]
            all_prioritize_results.append(max_coverage_testcase)
            remain_testcase.remove(max_coverage_testcase)
        else:
            break
    all_prioritize_results.extend(list(remain_testcase))
    return all_prioritize_results


def get_prioritize_result(prioritization, test_case_event_sequence, distance_option,
                          process_module_testcase_emb_file, all_name_without_log_list,
                          test_case_2_mutant_id_set, result_mutant_id_matrix,
                          process_module_prioritize_results_file, logs_representation, logger):
    if prioritization == "arp":
        # ARP
        prioritize_results = adaptive_random_prioritization(test_case_event_sequence,
                                                            distance_option,
                                                            process_module_testcase_emb_file,
                                                            all_name_without_log_list, logger)

    elif prioritization == "additional" or prioritization == "total":
        # total & additional
        if logs_representation == "semantics":
            logger.error("The total and additional prioritization strategies cannot be applied to "
                         "the log vectors produced by the semantics-based representation strategy.")
            exit(-1)
        prioritize_results = event_coverage_prioritization(process_module_testcase_emb_file,
                                                           all_name_without_log_list,
                                                           prioritization, logger)

    elif prioritization == "ideal":
        prioritize_results = ideal_prioritization(test_case_2_mutant_id_set, result_mutant_id_matrix,
                                                  process_module_prioritize_results_file, logger)
    else:
        logger.error("Error prioritize_option: %d" % prioritization)
        prioritize_results = []
        exit(-1)

    return prioritize_results
