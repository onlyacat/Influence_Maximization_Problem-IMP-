# -*-coding:utf-8-*-
import getopt
import multiprocessing
import sys
from random import uniform
from time import time

from numpy import random as rd

FILE_NAME = ""
FILE_SEED = ""
TIME = None
CPU = VER_NUM = EDGE_NUM = -1
SEED = EDGES = []
KEY = dict()
KEY_REVERSE = dict()
MODEL = "IC"


def error(traceback=None):
    if not traceback:
        print "python ISE.py -i <social network> -s <seed set> -m <diffusion model> " \
              "-b <termination type> -t <time budget> -r <random seed>"
    else:
        print traceback


def get_opt(argv):
    global FILE_NAME, FILE_SEED, TIME, MODEL
    file_name = file_seed = temp_model = termination_type = time = random_seed = None

    if len(argv) == 0:
        error()
    try:
        opts, args = getopt.getopt(argv, "hi:s:m:b:t:r:")
    except getopt.GetoptError:
        error()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            error()
            sys.exit()
        elif opt == "-i":
            file_name = str(arg)
        elif opt == "-s":
            file_seed = str(arg)
        elif opt == "-m":
            temp_model = str(arg)
        elif opt == "-b":
            termination_type = int(arg)
        elif opt == "-t":
            time = int(arg)
        elif opt == "-r":
            random_seed = str(arg)

    # print file_name, file_seed, temp_model, termination_type, time, random_seed

    if file_name is None or file_seed is None or temp_model is None or termination_type is None or random_seed is None:
        error("some information missing.")
        sys.exit(2)

    if termination_type == 1 and time is None:
        error("please input the time budget")
        sys.exit(2)

    FILE_NAME, FILE_SEED, TIME, MODEL = file_name, file_seed, time, temp_model


def get_data():
    global VER_NUM, EDGE_NUM, EDGES, KEY, SEED, CPU
    ans = []
    with open(FILE_NAME, 'r') as content:
        arr = content.readlines()

    for x in arr:
        ans.append(x.strip('\n').split(' '))

    VER_NUM, EDGE_NUM = int(ans[0][0]), int(ans[0][1])

    for x in ans[1:]:
        n1 = int(x[0])
        n2 = int(x[1])
        value = float(x[2])

        EDGES.append((n1, n2, value))
        if n1 in KEY:
            KEY[n1].append((n2, value))
        else:
            KEY[n1] = [(n2, value)]
        if n2 in KEY_REVERSE:
            KEY_REVERSE[n2].append((n1, value))
        else:
            KEY_REVERSE[n2] = [(n1, value)]

    with open(FILE_SEED, 'r') as content:
        SEED = content.readlines()

    CPU = multiprocessing.cpu_count()
    if CPU not in range(1, 20):
        CPU = 1
    if CPU > 8:
        CPU = 8

    # print "CPU: ", CPU


def independent_cascade_model(seed):
    status = [False] * (VER_NUM + 1)
    queue = []

    for y in seed:
        cy = int(y)
        queue.append(cy)
        status[cy] = True

    num = len(queue)

    while queue:
        current_point = queue.pop(0)
        if current_point not in KEY:
            continue

        for near_point in KEY[current_point]:
            if not status[near_point[0]]:
                if uniform(0, 1) <= near_point[1]:
                    num = num + 1
                    queue.append(near_point[0])
                    status[near_point[0]] = True
    return num


def linear_threshold_model(seed):
    status = [False] * (VER_NUM + 1)
    queue = []

    for y in seed:
        cy = int(y)
        queue.append(cy)
        status[cy] = True

    # threshold = [0]
    # for y in status[1:]:
    #     threshold.append(uniform(0, 1))

    threshold = rd.rand(len(status))
    threshold[0] = 0

    num = len(queue)

    while queue:
        new_queue = []
        for each_seed in queue:
            if each_seed not in KEY:
                continue

            for each_inactive_neighbor in KEY[each_seed]:
                each_inactive_neighbor_index = each_inactive_neighbor[0]
                if status[each_inactive_neighbor_index] is False:
                    activated_neighbors = 0

                    for z in KEY_REVERSE[each_inactive_neighbor_index]:
                        if status[z[0]] is True:
                            activated_neighbors = activated_neighbors + z[1]

                    if activated_neighbors > threshold[each_inactive_neighbor_index]:
                        new_queue.append(each_inactive_neighbor_index)
                        status[each_inactive_neighbor_index] = True

        num = num + len(new_queue)
        queue = new_queue

    return num


def multi_greedy_init(k=10000.0):
    pool = multiprocessing.Pool(CPU)
    result = []

    for_each = int(k / CPU)
    # print for_each
    for i in range(0, CPU):
        result.append(
            (pool.apply_async(testing_model_for_multi,
                              args=(SEED, i, VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, MODEL, for_each))))

    pool.close()
    pool.join()

    sp = []
    for x in result:
        sp.append(x.get())

    # print sp

    avg = sum(sp)

    return avg


def testing_model_for_multi(seed, i, vn, en, e, k, kr, m, num):
    global VER_NUM, EDGE_NUM, EDGES, KEY, MODEL, KEY_REVERSE
    VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, MODEL = vn, en, e, k, kr, m
    # print "starting processing ", i
    count = 0
    if MODEL == "IC":
        # print "starting IC"
        for x in range(0, num):
            count = count + independent_cascade_model(seed)
    else:
        for x in range(0, num):
            count = count + linear_threshold_model(seed)

    return count


if __name__ == '__main__':
    try:
        t_ini = time()
        get_opt(sys.argv[1:])
        get_data()
        # print TIME
        # if TIME is None:
        print multi_greedy_init(30000) / 30000.0
            # print "TIME:", time() - t_ini
        # else:
        #     t1 = time()
        #     a1 = multi_greedy_init()
        #     t2 = time()
        #     t_a = (t2 - t1)
        #     t_remain = TIME - (t2 - t_ini)
        #     # print t_remain
        #     n = a2 = 0
        #     if t_remain > 5:
        #         n = 0.9 * 10000 * float(t_remain) / t_a
        #         # print n
        #         a2 = multi_greedy_init(n)
        #
        #     # print "COUNT:", n + 10000
        #     print (a2 + a1) / (10000 + n)
        #     # print "TIME:", time() - t_ini

    except IOError:
        error("INVALID INPUT FILE")
# except Exception:
#     error()

# count.append(independent_cascade_model())
# print count

# count = 0
# if model == "IC":
#     print "model IC"
#     for x in range(0, 10001):
#         count = count + independent_cascade_model(SEED)
# else:
#     for x in range(0, 10001):
#         count = count + linear_threshold_model(SEED)
#
# print count / 10000.0

# def linear_threshold_model_b():
#     status = [False] * (VER_NUM + 1)
#     threshold = [0]
#     queue = []
#
#     for y in SEED:
#         cy = int(y)
#         queue.append(cy)
#         status[cy] = True
#
#     for y in status[1:]:
#         threshold.append(uniform(0, 1))
#         # threshold.append(1)
#
#     num = len(queue)
#
#     while 1:
#         cnum = 0
#         for y in KEY_REVERSE:
#             if status[y] is False:
#                 activated_neighbors = 0
#                 for z in KEY_REVERSE[y]:
#                     if status[z[0]] is True:
#                         activated_neighbors = activated_neighbors + z[1]
#
#                 if activated_neighbors > threshold[y]:
#                     cnum = cnum + 1
#                     status[y] = True
#
#         num = num + cnum
#
#         if cnum is 0:
#             break
#
#     return num
