import math
import multiprocessing
from itertools import combinations
from math import exp
from random import uniform
from copy import deepcopy as dc

CPU = VER_NUM = EDGE_NUM = -1
EDGES = []
KEY = dict()
KEY_REVERSE = dict()
c = 0.5
MODEL = "IC"


def get_data():
    global VER_NUM, EDGE_NUM, EDGES, KEY, CPU
    ans = []
    with open("NetHEPT.txt", 'r') as content:
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


def ini_ps(can_list, k):
    cpu = multiprocessing.cpu_count()
    if cpu not in range(1, 20):
        cpu = 1
    if cpu > 8:
        cpu = 8

    print cpu

    can = list(combinations(can_list, k))
    # can_list = [can[i:i + cpu] for i in range(0, len(can), cpu)]
    can_list = []
    for x in range(0, cpu):
        can_list.append(can[x::cpu])

    pool = multiprocessing.Pool(cpu)
    result = []

    print can_list

    for i in range(0, cpu):
        vn,en,e,k,kr,ci,m = dc(VER_NUM),dc(EDGE_NUM),dc(EDGES),dc(KEY),dc(KEY_REVERSE),dc(c),dc(MODEL)
        result.append((pool.apply_async(greedy, args=(can_list[i], i, vn,en,e,k,kr,ci,m))))

    pool.close()
    pool.join()

    sp = []
    for x in result:
        sp.extend(x.get())

    sp.sort(reverse=True)
    print sp
    return sp


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
    threshold = [0]
    queue = []

    for y in seed:
        cy = int(y)
        queue.append(cy)
        status[cy] = True

    for y in status[1:]:
        threshold.append(uniform(0, 1))

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


def greedy(seed, i, vn, en, e, k, kr, ci, m):
    global VER_NUM, EDGE_NUM, EDGES, KEY, c, MODEL, KEY_REVERSE
    VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, c, model = vn, en, e, k, kr, ci, m
    print "starting processing ", i
    inf = []
    for k in seed:
        num = testing_model(k)

        inf.append((num, k))
        print (i, num, k)

    # inf.sort(reverse=True)

    print inf
    return inf


def greedy_init(seed, i):
    # global VER_NUM, EDGE_NUM, EDGES, KEY, c, model, KEY_REVERSE
    # VER_NUM, EDGE_NUM, EDGES, KEY, KEY_REVERSE, c, model = vn, en, e, k, kr, ci, m
    print "starting processing ", i
    inf = []
    for k in seed:
        num = testing_model(k)

        inf.append((num, k))
        print (i, num, k)

    inf.sort(reverse=True)

    print inf
    return inf


def testing_model(seed):
    count = 0
    if MODEL is "IC":
        for x in range(0, 10001):
            count = count + independent_cascade_model(seed)
    else:
        for x in range(0, 10001):
            count = count + independent_cascade_model(seed)

    return count / 10000.0


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

    print s
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

    print seed
    return seed


def calculate_ltp_dag(v):
    sita = 1 / 320.0
    vp = set()
    ep = set()
    for every_edge in EDGES:
        if every_edge[1] != v:
            continue

        if every_edge[2] > sita and every_edge[0] not in vp:
            vp.add((every_edge[0], every_edge[1]))
            ep.add(every_edge[0])

            if every_edge[0] in KEY_REVERSE:
                for x in KEY_REVERSE[every_edge[0]]:
                    vp.add((x[0], every_edge[0]))
                    ep.add(x[0])

    # print list(vp)
    # print len(vp)
    # print list(ep)
    # print len(ep)
    return [len(ep) + len(vp), v, list(vp), list(ep)]


def calculate_force(D):
    v = D[1]
    f = dict()
    for u in D[3]:
        f[(u, v)] = 0

    rou = []
    queue = [v]
    for u in D[3]:
        pass


def ltdag():
    S = []
    D = []
    for ver in range(1, VER_NUM + 1):
        D.append(calculate_ltp_dag(ver))

    D.sort(reverse=True)

    D = D[0]

    print D


def calculate_ini_ver(k):
    seed = []
    seed_mapping = []
    for x in range(1, VER_NUM + 1):
        if x not in KEY_REVERSE:
            seed.append(x)

    for x in seed:
        value = 0
        if x in KEY:
            for y in KEY[x]:
                value = value + y[1]

        seed_mapping.append([value, x])

    seed_mapping.sort(reverse=True)

    seed = []

    for x in range(0, k):
        seed.append(seed_mapping[x][1])

    return seed


def calculate_2nd_ver(k):
    seed = []
    seed_mapping = []
    for x in range(1, VER_NUM + 1):
        if x not in KEY_REVERSE:
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

    print seed
    return seed


def combination_num(n, x):
    return math.factorial(n) / math.factorial(n - x) / math.factorial(x)


if __name__ == '__main__':
    get_data()
    can_list = set()

    can_list = set(HPG(6))
    can_list.update(set(degree_discount(6)))
    # can_list.update(set(calculate_2nd_ver(3)))

    print len(can_list)
    print can_list

    print combination_num(len(can_list), 2)

    # greedy(4, can_list)

    ini_ps(can_list, 2)
    # can = list(combinations(can_list, 2))
    # print can
    # greedy_init(can,0)

    # ltdag()
