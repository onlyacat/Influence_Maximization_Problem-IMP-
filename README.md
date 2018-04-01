# REPORT FOR IMP_PROBLEMS

@[onlyacat|2017.12.20]


[TOC]

---
## 1.Preliminaries
1. Use **python2.7** as the coding language.
2. Use **HPG(hybrid potential-influence greedy) algorithm** [^HPG] as one of the heuristic algorithms in searching the best influenced seed set.
3. Use another heuristic algorithm called **Degree Discount** [^DD] to find the initial seed set
4. Use greedy algorithm **CELF algorithm** [^CELF] to obtain the last set high-performantly.
5. Testing sample : 
>|   sample name | verticles  |  edges   |
| :--------:   | :-----:  | :----:  |:-----:  |
| `NetHEPT`| 15233  |   32235 |
|    `network` |   62 |   159|

6. The computer used for testing :
CPU: Intel `Core-m cy30` 4-core
RAM : 8 GB


---

## 2.Methodology
### 2.1 ISE : calculating influence
In this part , get the input requirements and read the data. 
Then calculate the final influence with loops 10000.


### 2.2 IMP : heuristic algorithms 
In this part, we will use two validated heuristic algorithms and a simple calculating function to emerge a initial seed set.  The size of set is decided by the number of verticles in the graph.

#### 2.2.1 Degree Function
Degree Function is just a simple function that calculating the out-degree of each verticle and then sorting. It is known that the influence and the chance of propagation is mostly decided by the arrows outward. Function shows below:
$$I_v = \sum_{n=i}^jD(v,n)$$

#### 2.2.2 HPG Algorithm
This algorithm takes more **potential influence** into consideration. It means that in the seed selection, it will choose the verticle containing more influence range instead. The two factor will be counted in calculating are: $outDegree$ and $inf$, which represent the out-degree of the verticle and the sum of the influence to all inactive neighbors if the verticle is active. Function shows below:
$$inf(u) = \sum_{v\in \overline N(u),v\not\in \overline A(u)}b_{uv}$$
$$PI(u) = outDegree(u) + (1-e^{-inf(u)})$$

In the function the $\overline N(u)$ means the set of verticle $u$ that its out degree neigbors and the  $\overline A(u)$ means the verticle that from $\overline N(u)$ and is active.


#### 2.2.3 Degree Discount Algorithm
 The algorithm is similar to `HPG ALGORITHM` that it pays more attention to the two verticles 
overlapping problems. Since two verticles are overlaied, it will limit the range of the propagation and weaken the final influence. To avoid this problem, this algorithm gives a function and for every verticle in seed set, it will give a factor to all the neighbor and to weaken their `ddv`. The function shows below:
$$dd_v = d_v + 2t_v - (d_v-t_v)t_vp$$


### 2.3 IMP : CELF Greedy Algorithm
In the main search: greedy search, we use the `CELF ALGORITHM`, which can faster the running speed. It find the useful property named `Sunmoduled Function`, which means that it only need to calculate the fisrt of all the verticles and this is its best performance in set. 
In the rest part, the algorithm will choose the performance that is larger of the current best performance.

---

## 3.EmpiricalVerification
### 3.1 Testing in heuristic algorithms
For testing, we will choose a seed set (`size = 20`) from this three algorithm and run `ISE` to get the influence. The result table shows below:
|   algorithm | model   |  influence  |model   |  influence  |
| :--------:   | :-----:  | :----:  |:-----:  | :----:  |
| Degree     | IC |   251     |LT |   303     |
|    HPG      |   IC   |   478 |LT |   585     |
| Degree Discount       |    IC    |  604  |LT |   714     |

The table shows that in these three algorithms, Degree Discount has the best result in generating initial  seed set, and simple Degree is worst.

### 3.2 Testing in overall algorithms

The testing result shows below, containing the heuristic algorithms and greedy algorithms .

|   sample | model   |  seed size |seed set |  spending time(s)  | influence|
| :--------:   | :-----:  | :----:  |:-----:  | :----:  | :----:  |
| network       | IC |   4|[56, 58, 53, 62]|  21     |27
|    network       |   LT|   4 |[56, 58, 53, 48]|   9 |32
| network       | IC |   10|[56, 58, 53, 62, 28, 48, 50, 61, 60, 45]|  77     |43
|  network |   LT|   10 |[56, 58, 53, 48, 62, 50, 28, 60, 61, 45]|   52 |55
| NetHEPT|    IC    |  4|[6024, 267, 37, 47] |   165 |279
| NetHEPT|    LT|  4|[6024, 2119, 1434, 37] |   176     | 338
| NetHEPT|    IC    |  10|[6024, 2119, 37, 47, 1434, 1241, 66, 3210, 156, 6573] |   302     |509
| NetHEPT       |    LT|  10|[6024, 2119, 1434, 37, 47, 66, 3210, 1241, 753, 6573] |   322     |626

[^HPG]: 田家堂. (2012). 在线社会网络中影响最大化问题的研究. (Doctoral dissertation, 复旦大学).

[^DD]: Chen W, Wang Y, Yang S. Efficient influence maximization in social networks[C]// ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Paris, France, June 28 - July. DBLP, 2009:199-208.

[^CELF]: Leskovec J, Krause A, Guestrin C, et al. Cost-effective outbreak detection in networks[C]// ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2007:420-429.