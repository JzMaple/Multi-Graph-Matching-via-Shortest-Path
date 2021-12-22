# Unifying Offline and Online Multi-Graph Matching via Finding Shortest Paths on Supergraph

#### Introduction

This work is based on the paper [*Unifying Offline and Online Multi-Graph Matching via Finding Shortest Paths on Supergraph*](https://ieeexplore.ieee.org/document/9076840), which is accepted by **TPAMI**.

It addresses the problem of multiple graph matching (MGM) by considering both offline batch mode and online setting. We explore the concept of cycle-consistency over pairwise matchings and formulate the problem as finding optimal composition path on the supergraph, whose vertices refer to graphs and edge weights denote score function regarding consistency and affinity. By our theoretical study we show that the offline and online MGM on supergraph can be converted to finding all pairwise shortest paths and single-source shortest paths respectively. We adopt the Floyd algorithm and shortest path faster algorithm (SPFA)  to effectively find the optimal path. 

In this repository, we release code and data for both MATLAB and Python version.

#### Citation

If you find our work or code useful in your research, please consider citing:

```latex
@article{jiang2020unifying,
  author={Jiang, Zetian and Wang, Tianzhe and Yan, Junchi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Unifying Offline and Online Multi-Graph Matching via Finding Shortest Paths on Supergraph}, 
  year={2021},
  volume={43},
  number={10},
  pages={3648-3663},
  doi={10.1109/TPAMI.2020.2989928}
}
```