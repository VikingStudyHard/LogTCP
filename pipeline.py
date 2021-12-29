import argparse
import logging
import os
import sys
from data.collect_log import *
from data.embedding import *
from mutant.get_mutation import *
from mutant.apfd import *
from prioritize.prioritization import *
from util.config import *

sys.path.extend(["../../", "../", "./"])

if __name__ == '__main__':
    dataset_dir = "dataset"
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--project', default="shiro", type=str,
                            help='the name of the under-test project')
    arg_parser.add_argument('--module', default="core", type=str,
                            help='the name of the under-test module')
    arg_parser.add_argument('--logs_representation', default="semantics", type=str,
                            help='Three log representation strategies.'
                                 'count: the count-based representation; '
                                 'ordering: the ordering-based representation; '
                                 'semantics: the semantics-based representation. ')
    arg_parser.add_argument('--prioritization', default="arp", type=str,
                            help='Three studied prioritization strategies and the ideal strategy. '
                                 'arp: the adaptive random prioritization strategy; '
                                 'total: the total prioritization strategy; '
                                 'additional: the additional prioritization strategy; '
                                 'ideal: the ideal prioritization results. ')
    arg_parser.add_argument('--distance_option', default="e", type=str,
                            help='e: euclidean distance; '
                                 'c: cosine distance; '
                                 'm: manhattan distance; '
                                 'n: no distance. ')

    args, extra_args = arg_parser.parse_known_args()
    project = args.project
    module = args.module
    logs_representation = args.logs_representation.lower()
    prioritization = args.prioritization.lower()
    distance_option = args.distance_option.lower()
    group_size = 5

    if prioritization == "ideal":
        logs_representation = "none"
        distance_option = "n"
    elif prioritization == "arp":
        if distance_option not in {"e", "c", "m"}:
            print("Wrong distance_option: %s. Please check it." % distance_option)
            exit(-1)
        if logs_representation not in {"count", "ordering", "semantics"}:
            print("Wrong logs_representation: %s. Please check it." % logs_representation)
            exit(-1)
    elif prioritization == "total" or prioritization == "additional":
        distance_option = "n"
        if logs_representation not in {"count", "ordering"}:
            print("Wrong logs_representation: %s. Please check it." % logs_representation)
            exit(-1)
    else:
        print("Wrong prioritization_option: %s. Please check it." % prioritization)
        exit(-1)

    log_path = os.path.join("log", project, module)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger_name = "pri-" + prioritization \
                  + "_dis-" + distance_option \
                  + "_rep-" + logs_representation

    hdlr = logging.FileHandler(os.path.join(log_path, logger_name + ".txt"))
    logger = logging.getLogger("main")
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    # Begin to prepare input log data.
    process_path = os.path.join(dataset_dir, "process")
    if not os.path.exists(process_path):
        os.mkdir(process_path)
    input_path = os.path.join(dataset_dir, 'input')
    input_project_path = os.path.join(input_path, project)
    input_module_path = os.path.join(input_project_path, module)

    if not os.path.exists(input_module_path):
        logger.error("Do not exist input_module_path: %s. " % input_module_path)
        exit(-1)

    config_file = os.path.join(input_module_path, "config.ini")
    logger.info("Load configfile %s" % config_file)
    if not os.path.exists(config_file):
        logger.error("Do not exist config_file: %s. Please check. " % config_file)
        exit(-1)

    process_project_path = os.path.join(process_path, project)
    if not os.path.exists(process_project_path):
        os.mkdir(process_project_path)
        logger.info("Create process_project_path: %s. " % process_project_path)

    process_module_path = os.path.join(process_project_path, module)
    if not os.path.exists(process_module_path):
        os.mkdir(process_module_path)
        logger.info("Create process_module_path: %s. " % process_module_path)

    log_path = os.path.join(input_module_path, 'drain_results')
    event_list_file = os.path.join(log_path, "log_events.txt")
    all_line_event_file = os.path.join(log_path, "all_line_event.txt")

    # ------------------------------
    # Get token embeddings
    # ------------------------------

    test_case_file = os.path.join(input_module_path, 'test_case.txt')
    if not os.path.exists(test_case_file):
        logger.error("Do not exist test_case_file: %s. Please check. " % test_case_file)
        exit(-1)

    all_test_case_list = []
    with open(test_case_file, "r", encoding="utf-8") as reader:
        for line in reader.readlines():
            all_test_case_list.append(line.strip())

    all_test_case_id_2_name = {}
    for i in range(len(all_test_case_list)):
        all_test_case_id_2_name[i] = all_test_case_list[i]

    test_case_event_sequence_file = os.path.join(input_module_path, 'test_case_2_event_sequence.txt')
    if not os.path.exists(test_case_event_sequence_file):
        logger.error("Do not exist test_case_event_sequence_file: %s. Please check. " % test_case_event_sequence_file)
        exit(-1)
    test_case_event_sequence = {}
    with open(test_case_event_sequence_file, "r", encoding="utf-8") as reader:
        for line in reader.readlines():
            tokens = line.strip().split()
            test_case_event_sequence[tokens[0]] = [int(i) for i in tokens[1:]]

    test_case_with_log_list = list(test_case_event_sequence.keys())

    all_name_without_log_list = sorted(list(set(all_test_case_list) - set(test_case_with_log_list)))
    logger.info("There are %d test cases without logs:" % len(all_name_without_log_list))
    logger.info(all_name_without_log_list)

    testcase_vec_path = os.path.join(process_module_path, "testcase_vec")
    if not os.path.exists(testcase_vec_path):
        os.mkdir(testcase_vec_path)
        logger.info("Create testcase_vec_path.")
    logger.info("Get logs vectors.")

    if prioritization != "ideal":
        logger.info("Get embeddings")

        embedding_path = os.path.join(process_module_path, "embed")
        if not os.path.exists(embedding_path):
            os.mkdir(embedding_path)
            logger.info("Create embedding_path %s. " % embedding_path)

        logs_representation_path = os.path.join(embedding_path, logs_representation)
        if not os.path.exists(logs_representation_path):
            os.mkdir(logs_representation_path)
            logger.info("Create logs_representation_path %s. " % logs_representation_path)

        if logs_representation == "semantics":
            process_module_pure_event_path = os.path.join(logs_representation_path, "preprocess_events" + ".txt")
            if not os.path.exists(process_module_pure_event_path):
                get_pure_events(process_module_pure_event_path, event_list_file, logger)

            pure_event_token_emb_file = os.path.join(logs_representation_path, "token_emb.vec")
            config = Configurable(config_file, extra_args)
            fasttext_model = config.fasttext_model
            fasttext_dim = config.fasttext_dim
            fasttext_thread = config.fasttext_thread
            fasttext_lr = config.fasttext_lr
            fasttext_epoch = config.fasttext_epoch
            fasttext_word_ngrams = config.fasttext_word_ngrams

            if not os.path.exists(pure_event_token_emb_file):
                process_module_fasttext_corpus = os.path.join(logs_representation_path, "fasttext_corpus.txt")
                if not os.path.exists(process_module_fasttext_corpus):
                    get_fasttext_corpus(process_module_pure_event_path, all_line_event_file,
                                        process_module_fasttext_corpus)
                    logger.info("Finish getting fasttext corpus.")
                get_token_embedding(process_module_fasttext_corpus,
                                    pure_event_token_emb_file,
                                    logger,
                                    fasttext_model=fasttext_model,
                                    fasttext_dim=fasttext_dim,
                                    fasttext_word_ngrams=fasttext_word_ngrams,
                                    fasttext_thread=fasttext_thread,
                                    fasttext_lr=fasttext_lr,
                                    fasttext_epoch=fasttext_epoch)

            process_module_pure_events_emb_file_name = logs_representation + "_token.vec"
            process_module_pure_events_emb_file = os.path.join(logs_representation_path,
                                                               process_module_pure_events_emb_file_name)

            if not os.path.exists(process_module_pure_events_emb_file):
                get_event_vec(pure_event_token_emb_file,
                              process_module_pure_events_emb_file,
                              process_module_pure_event_path, logger)
        else:
            process_module_pure_events_emb_file = None

        # Get logs vectors.
        process_module_testcase_emb_file_name = "testcase_" + logs_representation + ".vec"
        process_module_testcase_emb_file = os.path.join(testcase_vec_path,
                                                        process_module_testcase_emb_file_name)

        if not os.path.exists(process_module_testcase_emb_file):
            logger.info("Create process_module_testcase_emb_file %s " % process_module_testcase_emb_file)
            get_logs_embedding(logs_representation_path, logs_representation, prioritization,
                               event_list_file, process_module_testcase_emb_file,
                               logger, process_module_pure_events_emb_file, test_case_event_sequence)
        else:
            logger.info("Read from process_module_testcase_emb_file %s " % process_module_testcase_emb_file)
    else:
        process_module_testcase_emb_file = None

    prioritize_results_path = os.path.join(process_module_path, "prioritize_results")
    if not os.path.exists(prioritize_results_path):
        os.mkdir(prioritize_results_path)
        logger.info("Create prioritize_results_path %s. " % prioritize_results_path)
    process_module_prioritize_results_file_name = logger_name + ".txt"
    process_module_prioritize_results_file = os.path.join(prioritize_results_path,
                                                          process_module_prioritize_results_file_name)

    # ---------------------
    #    Select mutants
    # ---------------------
    logger.info("Get mutants: ")
    relationship_file = os.path.join(input_module_path, "mutant_id_2_test_case_id.txt")
    if not os.path.exists(relationship_file):
        logger.error("Do not exist mutant_info : %s. Please check. " % relationship_file)
        exit(-1)

    mutant_process_path = os.path.join(process_module_path, "mutant")
    if not os.path.exists(mutant_process_path):
        os.mkdir(mutant_process_path)
        logger.info("Create mutant_process_path %s " % mutant_process_path)
    mutant_id_index_after_filter_file = os.path.join(mutant_process_path, "mutant_id_after_filtering.txt")
    selected_mutant_group_file = os.path.join(mutant_process_path, "selected_mutant.txt")

    mutant_id_2_test_case_set = {}
    with open(relationship_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            token = line.strip().split()
            mutant_id_2_test_case_set[int(token[0])] = set([all_test_case_id_2_name[int(i)] for i in token[1:]])

    test_case_2_mutant_id_set = {}
    for test_case in all_test_case_list:
        test_case_2_mutant_id_set[test_case] = set()
        for mutant_id, test_case_set in mutant_id_2_test_case_set.items():
            if test_case in test_case_set:
                test_case_2_mutant_id_set[test_case].add(mutant_id)

    if not os.path.exists(selected_mutant_group_file):
        logger.info("Filter mutants...")
        if os.path.exists(mutant_id_index_after_filter_file):
            with open(mutant_id_index_after_filter_file, "rb") as f:
                available_mutant_id_list = pickle.load(f)
        else:
            available_mutant_id_list = filter_mutants(test_case_2_mutant_id_set, mutant_id_2_test_case_set, logger)
            with open(mutant_id_index_after_filter_file, "wb") as f:
                pickle.dump(available_mutant_id_list, f)
        logger.info("Create selected_mutants. ")
        result_mutant_id_matrix = select_limited_mutants(available_mutant_id_list, group_size, logger)
        with open(selected_mutant_group_file, "wb") as t:
            pickle.dump(result_mutant_id_matrix, t)

    else:
        logger.info("Read selected mutants from %s" % selected_mutant_group_file)
        with open(selected_mutant_group_file, "rb") as f:
            result_mutant_id_matrix = pickle.load(f)

    logger.info("result_mutant_id_matrix:")
    logger.info(result_mutant_id_matrix)

    # ------------------------------
    # Start to prioritize.
    # ------------------------------
    logger.info("Start to prioritize: ")

    if os.path.exists(process_module_prioritize_results_file):
        logger.info("Read prioritize results from %s" % process_module_prioritize_results_file)
        prioritize_results = read_prioritization_results(process_module_prioritize_results_file)
    else:
        prioritize_results = get_prioritize_result(prioritization, test_case_event_sequence,
                                                   distance_option,
                                                   process_module_testcase_emb_file,
                                                   all_name_without_log_list,
                                                   test_case_2_mutant_id_set, result_mutant_id_matrix,
                                                   process_module_prioritize_results_file,
                                                   logs_representation, logger)
        record_prioritization_results(process_module_prioritize_results_file, prioritize_results, logger)
    logger.info("Finish getting prioritization results.")

    # get APFD
    apfd_list = get_apfd_list(result_mutant_id_matrix, test_case_2_mutant_id_set, prioritize_results, logger)

    # -----------------
    #   get auc ratio  
    # -----------------

    ideal_prioritization_results_file_name = "pri-ideal_dis-n_rep-none.txt"
    ideal_prioritization_results_file = os.path.join(prioritize_results_path, ideal_prioritization_results_file_name)
    if os.path.exists(ideal_prioritization_results_file):
        logger.info("Read ideal prioritization results from %s" % ideal_prioritization_results_file)
        ideal_results = read_prioritization_results(ideal_prioritization_results_file)
    else:
        ideal_results = ideal_prioritization(test_case_2_mutant_id_set, result_mutant_id_matrix,
                                             process_module_prioritize_results_file, logger)
        record_prioritization_results(ideal_prioritization_results_file, ideal_results, logger)

    get_auc_ratio(result_mutant_id_matrix, test_case_2_mutant_id_set, prioritize_results,
                  ideal_results, logger)

    logger.info("\n\n\n")
