import fasttext
import os, sys, pickle
from collections import Counter
import numpy as np
from data.collect_log import load_events
import sys

sys.setrecursionlimit(10000)


def save_fasttext_vec(fasttext_model, vec_path):
    words = fasttext_model.get_words()
    with open(vec_path, 'w', encoding='utf-8') as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(fasttext_model.get_dimension()) + "\n")
        # line by line, you append vectors to VEC file
        for w in words:
            v = fasttext_model.get_word_vector(w)
            v_str = ""
            for vi in v:
                v_str += " " + str(vi)
            try:
                file_out.write(w + v_str + '\n')
            except:
                pass


def get_fasttext_corpus(process_project_module_pure_event_path, all_line_event_file,
                        process_project_module_fasttext_corpus):
    with open(process_project_module_pure_event_path, 'r', encoding="utf-8") as reader:
        all_event_count = 0
        all_event = {}
        for line in reader.readlines():
            all_event_count += 1
            all_event[all_event_count] = line

    all_line_temp = load_events(all_line_event_file)  # line : log id
    with open(process_project_module_fasttext_corpus, 'w', encoding="utf-8") as w:
        for i in range(len(all_line_temp)):
            w.write(all_event[all_line_temp[i]] + ' ')


def get_token_embedding(process_project_module_fasttext_corpus, process_project_module_pure_event_token_emb_file,
                        logger,
                        fasttext_model='skipgram', fasttext_dim=50, fasttext_word_ngrams=3,
                        fasttext_thread=2, fasttext_lr=0.01, fasttext_epoch=5):
    if not os.path.exists(process_project_module_fasttext_corpus):
        logger.error("No fasttext corpus.")
        exit(-1)
    logger.info('Training FastText model...')
    # pretrainedVectors = config.pretrained_embeddings_file
    logger.info("Start to train fasttext models with model %s, dim %d, lr %s, epoch %d, thread %d" %
                (fasttext_model, fasttext_dim, fasttext_lr, fasttext_epoch, fasttext_thread))

    text_model = fasttext.train_unsupervised(process_project_module_fasttext_corpus, minCount=1, maxn=10,
                                             wordNgrams=fasttext_word_ngrams,
                                             model=fasttext_model, dim=fasttext_dim,
                                             epoch=fasttext_epoch, lr=fasttext_lr, thread=fasttext_thread)
    save_fasttext_vec(text_model, process_project_module_pure_event_token_emb_file)
    logger.info("Finish training fasttext vec. ")


def get_event_vec(module_pure_event_token_emb_file, process_project_module_pure_events_emb_file,
                  process_project_module_pure_event_path, logger):
    word_vocab = {}
    embedding_dim = -1
    logger.info("Read vectors from %s" % module_pure_event_token_emb_file)
    with open(module_pure_event_token_emb_file, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            values = line.strip().split()
            if len(values) > 2:
                if embedding_dim == -1:
                    embedding_dim = len(values) - 1
                if len(values) == embedding_dim + 1:
                    word, embed = values[0], np.asarray(values[1:], dtype=np.float)
                    word_vocab[word] = embed

    event_tokens = {}
    if not os.path.exists(process_project_module_pure_event_path):
        logger.error("No process_project_pure_event_path")
        exit(-1)

    with open(process_project_module_pure_event_path, 'r', encoding="utf-8") as reader:
        line_num = 0
        for line in reader.readlines():
            line_num += 1
            tokens = line.strip().split()
            event_tokens[line_num] = tokens

    event_vec = {}
    for log_id, tokens in event_tokens.items():
        place_holder = np.zeros(embedding_dim)
        if tokens[0] == "this_is_an_empty_event":
            event_vec[log_id] = place_holder
        else:
            for token in tokens:
                if token in word_vocab.keys():
                    emb = word_vocab[token]
                else:
                    emb = np.zeros(embedding_dim)
                place_holder += emb
            event_vec[log_id] = place_holder

    with open(process_project_module_pure_events_emb_file, 'w', encoding='utf-8')as writer:
        writer.write(str(len(event_vec)) + ' ' + str(embedding_dim) + "\n")
        for event_id, embed in event_vec.items():
            embed = ' '.join([str(x) for x in embed.tolist()])
            writer.write(' '.join([str(event_id), embed]) + "\n")
    logger.info("Save pure events embeddings. ")


def read_vec(emb_file):
    vec = {}
    with open(emb_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            values = line.strip().split()
            if len(values) == 2:
                key_count, dim = [int(x) for x in line.strip().split()]
            else:
                if len(values) == dim + 1:
                    key, embedding = values[0], np.asarray(values[1:], dtype=np.float)
                    vec[key] = embedding
    assert len(vec) == key_count
    return vec


def save_testcase_vec(process_module_testcase_emb_file, testcase_vec, embedding_dim, logger):
    logger.info("Obtain testcase_vec. ")
    with open(process_module_testcase_emb_file, 'w', encoding='utf-8') as writer:
        writer.write(str(len(testcase_vec)) + ' ' + str(embedding_dim) + "\n")
        name_list = sorted(list(testcase_vec.keys()))
        for name in name_list:
            embed = ' '.join([str(x) for x in testcase_vec[name].tolist()])
            writer.write(' '.join([name, embed]) + "\n")
    logger.info("Save testcase_vec. ")


def get_logs_embedding(logs_representation_path, logs_representation, prioritization,
                       event_list_file, process_module_testcase_emb_file,
                       logger, process_module_pure_events_emb_file, test_case_event_list):
    with open(event_list_file, 'r', encoding="utf-8") as reader:
        all_event_count = 0
        for line in reader.readlines():
            if len(line) > 0:
                all_event_count += 1

    testcase_vec = {}
    embedding_dim = 0
    if logs_representation == "semantics":
        event_vec = read_vec(process_module_pure_events_emb_file)
        for name, log_id_list in test_case_event_list.items():
            embedding_dim = len(event_vec[str(1)])
            place_holder = np.zeros(embedding_dim)
            for log_id in log_id_list:
                if str(log_id) in event_vec.keys():
                    emb = event_vec[str(log_id)]
                else:
                    emb = np.zeros(embedding_dim)
                place_holder += emb
            testcase_vec[name] = place_holder

    elif logs_representation == "count":
        embedding_dim = all_event_count
        norm_max_log_counter = Counter()
        norm_min_log_counter = Counter()
        if prioritization == "arp":
            for log_list in test_case_event_list.values():
                log_counter = Counter(log_list)
                for cur_log_id, count_num in log_counter.items():
                    cur_log_id -= 1
                    if norm_max_log_counter[cur_log_id] < count_num:
                        norm_max_log_counter[cur_log_id] = count_num
                    if norm_min_log_counter[cur_log_id] > count_num:
                        norm_min_log_counter[cur_log_id] = count_num

        for name, log_id_list in test_case_event_list.items():
            place_holder = np.zeros(embedding_dim)
            log_counter = Counter(log_id_list)
            if prioritization == "arp":
                for cur_log_id, count_num in log_counter.items():
                    cur_log_id -= 1
                    # normalization
                    if norm_max_log_counter[cur_log_id] != norm_min_log_counter[cur_log_id]:
                        place_holder[cur_log_id] = (count_num - norm_min_log_counter[cur_log_id]) / (
                            norm_max_log_counter[cur_log_id] - norm_min_log_counter[cur_log_id])
                    else:
                        place_holder[cur_log_id] = 1
            elif prioritization == "total" or prioritization == "additional":
                for cur_log_id in log_counter.keys():
                    cur_log_id -= 1
                    place_holder[cur_log_id] = 1
            else:
                logger.error("Error prioritization %s " % prioritization)
                exit(-1)
            testcase_vec[name] = place_holder

    elif logs_representation == 'ordering':
        level_name_tuple_counter = {}
        tuple_set = set()
        tuple_info = "tuple_info_bi-gram.txt"
        tuple_info_file = os.path.join(logs_representation_path, tuple_info)

        if not os.path.exists(tuple_info_file):
            logger.info("Create tuple_info_file %s " % tuple_info_file)
            for name, log_id_list in test_case_event_list.items():
                tuple_counter = Counter()
                for index in range(len(log_id_list) - 1):
                    ngram_tuple = tuple(log_id_list[index: index + 2])
                    tuple_counter[ngram_tuple] += 1
                    tuple_set.add(ngram_tuple)
                level_name_tuple_counter[name] = tuple_counter
            with open(tuple_info_file, "wb") as f:
                pickle.dump((level_name_tuple_counter, tuple_set), f)
        else:
            logger.info("Read from tuple_info_file %s " % tuple_info_file)
            with open(tuple_info_file, "rb") as f:
                level_name_tuple_counter, tuple_set = pickle.load(f)

        index = 0
        tuple_2_id = {}

        norm_max_tuple_counter = {}
        norm_min_tuple_counter = {}
        for i in tuple_set:
            tuple_2_id[i] = index
            norm_max_tuple_counter[index] = -float('inf')
            norm_min_tuple_counter[index] = float('inf')
            index += 1

        for name, tuple_counter in level_name_tuple_counter.items():
            for ngram_tuple, counter in tuple_counter.items():
                ngram_tuple_id = tuple_2_id[ngram_tuple]
                if counter > norm_max_tuple_counter[ngram_tuple_id]:
                    norm_max_tuple_counter[ngram_tuple_id] = counter
                if counter < norm_min_tuple_counter[ngram_tuple_id]:
                    norm_min_tuple_counter[ngram_tuple_id] = counter
        embedding_dim = len(tuple_2_id)
        for name, tuple_counter in level_name_tuple_counter.items():
            place_holder = np.zeros(embedding_dim)
            for ngram_tuple, counter in tuple_counter.items():
                if prioritization == "arp":
                    place_holder[tuple_2_id[ngram_tuple]] = counter
                elif prioritization == "total" or prioritization == "additional":
                    place_holder[tuple_2_id[ngram_tuple]] = 1
                else:
                    logger.error("Error prioritization %s " % prioritization)
                    exit(-1)
            if prioritization == "arp":
                for tuple_id in range(len(place_holder)):
                    if place_holder[tuple_id] != 0:
                        if norm_max_tuple_counter[tuple_id] != norm_min_tuple_counter[tuple_id]:
                            place_holder[tuple_id] = (place_holder[tuple_id] - norm_min_tuple_counter[tuple_id]) / \
                                                     (norm_max_tuple_counter[tuple_id] - norm_min_tuple_counter[
                                                         tuple_id])
                        else:
                            place_holder[tuple_id] = 1
            testcase_vec[name] = place_holder

    else:
        logger.warning("Error logs_representation %s " % logs_representation)
        exit(-1)
    save_testcase_vec(process_module_testcase_emb_file, testcase_vec, embedding_dim, logger)
