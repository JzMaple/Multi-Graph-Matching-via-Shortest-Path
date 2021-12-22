# Unifying Offline and Online Multi-Graph Matching via Finding Shortest Paths on Supergraph

#### Introduction

This work is based on the paper [*Unifying Offline and Online Multi-Graph Matching via Finding Shortest Paths on Supergraph*](https://ieeexplore.ieee.org/document/9076840), which is accepted by **TPAMI**.

It addresses the problem of multiple graph matching (MGM) by considering both offline batch mode and online setting. We explore the concept of cycle-consistency over pairwise matchings and formulate the problem as finding optimal composition path on the supergraph, whose vertices refer to graphs and edge weights denote score function regarding consistency and affinity. By our theoretical study we show that the offline and online MGM on supergraph can be converted to finding all pairwise shortest paths and single-source shortest paths respectively. We adopt the Floyd algorithm and shortest path faster algorithm (SPFA)  to effectively find the optimal path. 

In this repository, we release code and data for both MATLAB and Python version.



#### Get Started

MATLAB version is developed and tested on MATLAB2018b, and Python version is developed and tested with Ubuntu 16.04, Python 3.6.9, Pytorch 1.4.0, cuda11.0 for Python code.

1. Download the dataset by running `./ download_data.sh`

2. For MATLAB version, just run `run_offline.m` and `run_online.m`. The default configuration are settled in .`/matlab/setArgs`

3. For Python version, default configuration files are stored in `./python/experiments`.

   ```
   cd ./python
   python run_offline.py ./experiments/offline.json
   ```

   Online setting has not been supported yet.

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