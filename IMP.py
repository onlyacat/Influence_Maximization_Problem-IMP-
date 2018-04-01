import getopt
import math
import multiprocessing
import sys
from math import exp
from random import uniform
from time import time

from numpy import random as rd

TERMINATION_TYPE = CPU = VER_NUM = EDGE_NUM = -1
DD_ANS = EDGES = []
KEY = dict()
KEY_REVERSE = dict()
c = 0.5
MODEL = "IC"
SEED_SIZE = 4
FILE_NAME = "NetHEPT.txt"
TIME = None
RUNNING_NUM = 4000


def error(traceback=None):
    if not traceback:
        print "python IMP.py -i <social network> -s <seed set> -m <diffusion model> " \
              "-b <termination type> -t <time budget> -r <random seed>"
    else:
        print traceback


def get_opt(argv):
    global FILE_NAME, SEED_SIZE, TIME, MODEL, TERMINATION_TYPE
    file_name = seed_size = temp_model = termination_type = time = random_seed = None

    if len(argv) == 0:
        raise TypeError
    try:
        opts, args = getopt.getopt(argv, "hi:k:m:b:t:r:")
    except getopt.GetoptError:
        error()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            error()
            sys.exit()
        elif opt == "-i":
            file_name = str(arg)
        elif opt == "-k":
            seed_size = int(arg)
        elif opt == "-m":
            temp_model = str(arg)
        elif opt == "-b":
            termination_type = int(arg)
        elif opt == "-t":
            time = int(arg)
        elif opt == "-r":
            random_seed = str(arg)

    # print file_name, file_seed, temp_model, termination_type, time, random_seed

    if file_name is None or seed_size is None or temp_model is None or termination_type is None or random_seed is None:
        error("some information missing.")
        sys.exit(2)

    if termination_type == 1 and time is None:
        error("please input the time budget")
        sys.exit(2)

    FILE_NAME, SEED_SIZE, TIME, MODEL, TERMINATION_TYPE = file_name, seed_size, time, temp_model, termination_type


def get_data():
    global VER_NUM, EDGE_NUM, EDGES, KEY, CPU
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


def greedy(seed, i, vn, en, e, k, kr, ci, m):
    global VER_NUM, EDGE_NUM, EDGES, KEY, c, MODEL, KEY_REVERSE
    VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, c, model = vn, en, e, k, kr, ci, m
    # print "starting processing ", i
    inf = []
    for k in seed:
        num = testing_model(k)

        inf.append([num, k[0]])
        # print (i, num, k)

    # print inf
    return inf


def multi_greedy_init(seed):
    pool = multiprocessing.Pool(CPU)
    result = []

    k = RUNNING_NUM
    for_each = int(k / CPU)
    for i in range(0, CPU):
        result.append(
            (pool.apply_async(testing_model_for_multi,
                              args=(seed, i, VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, c, MODEL, for_each))))

    pool.close()
    pool.join()

    sp = []
    for x in result:
        sp.append(x.get())

    avg = sum(sp) / len(sp)
    # print (avg, seed)
    return avg


def testing_model_for_multi(seed, i, vn, en, e, k, kr, ci, m, num):
    global VER_NUM, EDGE_NUM, EDGES, KEY, c, MODEL, KEY_REVERSE
    VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, c, model = vn, en, e, k, kr, ci, m
    # print "starting processing ", i
    count = 0
    if model == "IC":
        # print "starting IC"
        for x in range(0, num):
            count = count + independent_cascade_model(seed)
    else:
        for x in range(0, num):
            count = count + linear_threshold_model(seed)

    return count / float(num)


def testing_model(seed):
    count = 0
    if MODEL == "IC":
        # print "starting IC"
        for x in range(0, RUNNING_NUM):
            count = count + independent_cascade_model(seed)
    else:
        for x in range(0, RUNNING_NUM):
            count = count + linear_threshold_model(seed)

    return count / float(RUNNING_NUM)


def calculate_pi(seed):
    pi = []
    for x in range(1, VER_NUM + 1):
        if x in seed:
            continue

        if x in KEY:
            inf = 0
            for y in KEY[x]:
                if y[0] not in seed:
                    inf = inf + y[1]

            if inf != 0:
                pi.append(((len(KEY[x]) + 1 - exp(-inf)), x))

    return pi


def HPG(k):
    s = []
    for x in range(0, k):
        pi = calculate_pi(s)
        pi.sort(reverse=True)

        for y in pi:
            if y[1] not in s:
                s.append(y[1])
                break

    # print s

    # print s
    return s


def HPG_testing(k):
    s = []
    pi = calculate_pi(s)
    pi.sort(reverse=True)

    # print s
    for y in range(0, k):
        s.append(pi[y][1])

    # print s
    return s


def degree_discount(k):
    seed = []
    degree = dict()
    d_degree = dict()
    d_degree_in_list = []
    t = [0] * (VER_NUM + 1)

    for index in range(1, VER_NUM + 1):
        count_degree = 0
        if index in KEY:
            for each_key in KEY[index]:
                count_degree = count_degree + each_key[1]

        degree[index] = count_degree
        d_degree[index] = count_degree
        d_degree_in_list.append([count_degree, index])

    for i in range(0, k):
        d_degree_in_list.sort(reverse=True)
        for x in d_degree_in_list:
            if x[1] not in seed:
                u = x[1]
                break

        seed.append(u)

        if u not in KEY:
            continue

        for v in KEY[u]:
            vvalue = v[0]

            if vvalue in seed:
                continue

            t[vvalue] = t[vvalue] + 1
            d_degree[vvalue] = degree[vvalue] - 2 * t[vvalue] - (degree[vvalue] - t[vvalue]) * t[vvalue] * 0.01
            for x in d_degree_in_list:
                if x[1] is vvalue:
                    x[0] = d_degree[vvalue]
                    break

    seed.sort(reverse=True)
    # print seed
    return seed


def greedy_init(seed):
    num = testing_model(seed)
    # print (num, seed)
    # inf.sort(reverse=True)
    return num


def calculate_2nd_ver(k):
    seed = []
    seed_mapping = []
    for x in range(1, VER_NUM + 1):
        seed.append(x)

    for x in seed:
        value = 0
        if x in KEY:
            for y in KEY[x]:
                value = value + y[1]

                if y in KEY:
                    for z in KEY[y]:
                        value = value + y[1] * z[1]

                        if z in KEY:
                            for k in KEY[z]:
                                value = value + y[1] * z[1] * k[1]

        seed_mapping.append([value, x])

    seed_mapping.sort(reverse=True)

    seed = []

    for x in range(0, k):
        seed.append(seed_mapping[x][1])

    # print seed
    return seed


def combination_num(n, x):
    return math.factorial(n) / math.factorial(n - x) / math.factorial(x)


def ini_celf(seed_can):
    seed = []

    for x in seed_can:
        seed.append([x])

    can_lis = []
    for x in range(0, CPU):
        can_lis.append(seed[x::CPU])

    pool = multiprocessing.Pool(CPU)
    result = []

    # print can_lis

    for i in range(0, CPU):
        result.append(
            (pool.apply_async(greedy, args=(can_lis[i], i, VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, c, MODEL))))

    pool.close()
    pool.join()

    sp = []
    for x in result:
        sp.extend(x.get())

    sp.sort(reverse=True)
    # print sp
    return sp


def deciding_rest(seed, can):
    k = SEED_SIZE - len(seed)

    # print "IN REST num is:", k

    for x in can:
        if k == 0:
            break
        else:
            if x[1] not in seed:
                seed.append(x[1])
                k = k - 1

    return seed


def celf(can):
    c1 = can.pop(0)
    seed = [c1[1]]

    all_inf = c1[0]
    for y in range(1, SEED_SIZE):
        if TERMINATION_TYPE == 1:
            used_time = time() - TIME_START
            if used_time - TIME > -10:
                return deciding_rest(seed, can)

        best_can = can[0]
        seed.append(best_can[1])
        temp_inf = multi_greedy_init(seed)
        # temp_inf = greedy_init(seed)
        best_can[0] = temp_inf - all_inf
        seed.remove(best_can[1])

        for z in can[1:]:

            if z[0] > best_can[0]:
                seed.append(z[1])
                temp_inf_i = multi_greedy_init(seed)
                # temp_inf_i = greedy_init(seed)
                z[0] = temp_inf_i - all_inf
                seed.remove(z[1])
                if z[0] > best_can[0]:
                    best_can = z
                    temp_inf = temp_inf_i

        nc = can.pop(can.index(best_can))
        seed.append(nc[1])
        all_inf = temp_inf

    return seed


def hui():
    global RUNNING_NUM, DD_ANS
    size = SEED_SIZE * 20

    if size > 500:
        size = SEED_SIZE * 10
    if size > 500:
        size = SEED_SIZE * 8
    if size > 500:
        size = SEED_SIZE * 5
    if size > VER_NUM:
        size = VER_NUM

    # print "SIZE:", size

    if MODEL == "LT":
        RUNNING_NUM = 2400

    cl = set()
    # cl = set(HPG(100))
    cl.update(set(HPG_testing(int(size / 2))))
    DD_ANS = degree_discount(size)
    cl.update(set(DD_ANS))
    cl.update(set(calculate_2nd_ver(size)))

    # print len(cl)
    # print cl

    return cl


if __name__ == '__main__':
    TIME_START = time()

    get_opt(sys.argv[1:])
    get_data()

    t2 = time()

    can_list = hui()

    t3 = time()

    c = ini_celf(list(can_list))

    t4 = time()

    final = celf(c)

    for x in final:
        print x

    t5 = time()

    # print "SPENDING TIME : ", t5 - TIME_START
    # print "in data:", t2 - TIME_START
    # print "in hu:", t3 - t2
    # print "in ini greed:", t4 - t3
    # print "in greed:", t5 - t4
    # ini_celf(can_list)

    # print combination_num(len(can_list), 50)
    # greedy(4, can_list)

    # ini_ps(can_list, 4)

    # can = list(combinations(can_list, 2))
    # print can
    # greedy_init(can,0)

    # ltdag()

# def calculate_ltp_dag(v):
#     sita = 1 / 320.0
#     vp = set()
#     ep = set()
#     for every_edge in EDGES:
#         if every_edge[1] != v:
#             continue
#
#         if every_edge[2] > sita and every_edge[0] not in vp:
#             vp.add((every_edge[0], every_edge[1]))
#             ep.add(every_edge[0])
#
#             if every_edge[0] in KEY_REVERSE:
#                 for x in KEY_REVERSE[every_edge[0]]:
#                     vp.add((x[0], every_edge[0]))
#                     ep.add(x[0])
#
#     # print list(vp)
#     # print len(vp)
#     # print list(ep)
#     # print len(ep)
#     return [len(ep) + len(vp), v, list(vp), list(ep)]
#
#
# def calculate_force(D):
#     v = D[1]
#     f = dict()
#     for u in D[3]:
#         f[(u, v)] = 0
#
#     rou = []
#     queue = [v]
#     for u in D[3]:
#         pass
#
#
# def ltdag():
#     S = []
#     D = []
#     for ver in range(1, VER_NUM + 1):
#         D.append(calculate_ltp_dag(ver))
#
#     D.sort(reverse=True)
#
#     D = D[0]
#
#     print D
#
# def testing():
#     sss = []
#     for x in [4, 10, 20, 30, 40, 50, 100]:
#         sss.append(["num: ", x, " HPG: ", testing_model(HPG(x)), " DD: ", testing_model(degree_discount(x)), " C2V: ",
#                     testing_model(calculate_2nd_ver(x))])
#
#     print sss


# def ini_ps(can_list, k):
#
#     can = list(combinations(can_list, k))
#     can_list = []
#     for x in range(0, CPU):
#         can_list.append(can[x::CPU])
#
#     pool = multiprocessing.Pool(CPU)
#     result = []
#
#     print can_list
#
#     for i in range(0, CPU):
#         result.append(
#             (pool.apply_async(greedy, args=(can_list[i], i, VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, c, model))))
#
#     pool.close()
#     pool.join()
#
#     sp = []
#     for x in result:
#         sp.extend(x.get())
#
#     sp.sort(reverse=True)
#     print sp
#     return sp
#
#
