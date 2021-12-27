from mutant.get_mutation import *
from sklearn import metrics
import math


def get_apfd_list(result_mutant_id_matrix, name_mutant_id_set, prioritize_results, logger):
    logger.info('Get apfd results.')
    logger.info("prioritization: ")
    logger.info(prioritize_results)
    apfd_list_prioritization = []
    group_count = result_mutant_id_matrix.shape[0]
    mutant_count = result_mutant_id_matrix.shape[1]
    for group_num in range(group_count):
        group_mutant_id_list = result_mutant_id_matrix[group_num]
        assert len(group_mutant_id_list) == mutant_count
        tf_list = []
        group_mutant_count = 0
        tf = len(prioritize_results)
        for mutant_id in group_mutant_id_list:
            group_mutant_count += 1
            for name_index in range(len(prioritize_results)):
                testcase_name = prioritize_results[name_index]
                if mutant_id in name_mutant_id_set[testcase_name]:
                    tf = name_index + 1
                    break
            tf_list.append(tf)
        apfd = 1 - sum(tf_list) / (len(prioritize_results) * group_mutant_count) + 1 / (len(prioritize_results) * 2)
        apfd_list_prioritization.append(apfd)
    logger.info("The APFD list:")
    logger.info(apfd_list_prioritization)
    logger.info("mean APFD: %.4f" % np.mean(apfd_list_prioritization))
    print('APFD: ' + str(f'{np.mean(apfd_list_prioritization):0.4f}'), end='\n')
    return apfd_list_prioritization


def get_mutant_kill_number(prioritization_results, result_mutant_id_list, name_mutant_id_set):
    y = []
    result_mutant_id_set = set(result_mutant_id_list)
    temp_mutant_set = set()
    for index, name in enumerate(prioritization_results):
        temp_mutant_set = temp_mutant_set | (name_mutant_id_set[name] & result_mutant_id_set)
        y.append(len(temp_mutant_set))
    return y


def get_auc(x, top_n_testcase_num, prioritization_result, result_mutant_id_list, test_case_mutant_id_set):
    kill_number = get_mutant_kill_number(prioritization_result, result_mutant_id_list, test_case_mutant_id_set)
    auc = []
    for x_ind, num in enumerate(top_n_testcase_num):
        top_n_kill_num = kill_number[:num]
        if len(x[x_ind]) == 1:
            auc.append(top_n_kill_num[0])
        else:
            auc.append(metrics.auc(x[x_ind], top_n_kill_num))
    return auc


def get_auc_ratio(result_mutant_id_matrix, test_case_mutant_id_set, prioritize_results,
                  ideal_result, logger):
    logger.info("Computing auc ratio and recall.")
    top_n = [0.25, 0.5, 0.75, 1]
    x = []
    testcase_num = len(ideal_result)
    top_n_testcase_num = []
    for rate in top_n:
        top_n_testcase_num.append(math.ceil(testcase_num * rate))
    for num in top_n_testcase_num:
        x.append(range(num))  # x coordinate
    result_mutant_id_list = result_mutant_id_matrix.flatten()
    auc_ideal = get_auc(x, top_n_testcase_num, ideal_result, result_mutant_id_list, test_case_mutant_id_set)

    top_10_testcase_index = []
    for top_10_testcase in ideal_result[:10]:
        if top_10_testcase not in prioritize_results:
            continue
        top_10_testcase_index.append(prioritize_results.index(top_10_testcase))
    logger.info("Index of ideal top-10 testcases in our prior results: ")
    logger.info(top_10_testcase_index)
    auc_prior = get_auc(x, top_n_testcase_num, prioritize_results, result_mutant_id_list, test_case_mutant_id_set)

    auc_ratio_results = []
    for rate in range(len(top_n)):
        auc_ratio_results.append([])

    for rate_ind in range(len(top_n)):
        auc_ratio_results[rate_ind].append(auc_prior[rate_ind] / auc_ideal[rate_ind])
            
    result = []
    for ind, rate in enumerate(top_n):
        logger.info("Auc ratio results in top-%d%%: %s" % (int(rate * 100), str(auc_ratio_results[ind])))
        logger.info("mean auc ratio in top-%d%%: %f" % (int(rate * 100), np.mean(auc_ratio_results[ind])))
        result.append(str(f'{np.mean(auc_ratio_results[ind]):0.4f}'))
    logger.info("\t".join(result))
    for index, percent in enumerate(top_n):
        print("RAUC-" + str(int(percent * 100)) + "%:", end=' ')
        print(result[index], end='\t')
