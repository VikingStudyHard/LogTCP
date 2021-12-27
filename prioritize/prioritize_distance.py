import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def cosine_distance(x, y):
    temp = np.linalg.norm(x) * (np.linalg.norm(y))
    if temp == 0:
        return 1
    else:
        return 1 - np.dot(x, y) / temp


def compute_distance(distance_option, x, y, logger):
    if distance_option == "e":
        return euclidean_distance(x, y)
    elif distance_option == "m":
        return manhattan_distance(x, y)
    elif distance_option == "c":
        return cosine_distance(x, y)
    else:
        logger.error("Error distance_option: %s" % distance_option)
        exit(-1)


def get_distance_matrix(testcase_list, testcase_vec, distance_option, logger):
    t = len(testcase_vec)
    dis_matrix = np.zeros((t, t))
    logger.info("Computing distance matrix with distance_option %s" % distance_option)
    for i in range(len(testcase_list)):
        for j in range(i + 1, len(testcase_list)):
            x = testcase_vec[testcase_list[i]]
            y = testcase_vec[testcase_list[j]]
            dis_matrix[i, j] = compute_distance(distance_option, x, y, logger)
            dis_matrix[j, i] = dis_matrix[i, j]
    return dis_matrix
