
# Distributed Hierarchical Deep Reinforcement Learning for Semantic-aware Resource Allocation

Tensorflow implementation of the [paper](https://ieeexplore.ieee.org/document/11006945) "Distributed Hierarchical Deep Reinforcement Learning for Semantic-aware Resource Allocation". 

# Introduction
Beyond the traditional quality of experience (QoE) optimization that focuses on bit transmission, taking into account semantic QoE can better optimize the network to improve user experience. This paper develops a novel distributed hierarchical deep reinforcement learning (DHDRL) framework with a two-layer control strategy in different timescales to optimize QoE in the downlink multi-cell network including the semantic rate and accuracy. Specifically, we decompose the optimization problem into two sub-problems, beamforming and power allocation as well as semantic resource allocation, resulting in an infinite-horizon discrete and a finite-horizon continuous-time Markov decision processes. Two-layer neural networks (NNs) are proposed to solve the sub-problems. At the network level, a distributed high-level NN is proposed to mitigate the inter-cell interference and improve the network performance on a large timescale, while at the link level, another distributed low-level NN to improve the semantic compression rate and meet various semantic accuracy and rate requirements on a small timescale. Numerical results show that DHDRL achieves at least a 19.07% performance gain over the single-layer method, maintains effectiveness in high-dimensional scenarios, and exhibits strong generalization across a variety of multi-objective optimization problems.

![ ](./figure/DHDRL-SystemModel.png)
>  Illustration of the proposed semantic-awaremulti-cell communication network. Each BS serves the corresponding UTs and interferes with the non-serving UTs at the same time, and dynamically adjusts the beamforming, power allocation, and semantic compression rate according to the user requirements and channel conditions, so as to improve the transmission performance.


# Prerequites
* [Python 3.7]


# Quick Start

Install the environment and run the hier_main.py function directly for training

# Experimental results


## QoE score versus times lots for different schemes

![ ](./figure/TimeslotVSqoe_S10_P20_C3-eps-converted-to.png)

# Citation

Please use the following BibTeX citation If you find the code is useful for your research:

```
@ARTICLE{yu2025distributed,
  author={Yu, Kaiwen and He, Qi and Yu, Chuanhang and Yang, Xingyu and Wu, Gang},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Distributed Hierarchical Deep Reinforcement Learning for Semantic-aware Resource Allocation}, 
  year={2025},
  volume={},
  number={},
  pages={1-13},
  keywords={Optimization;Quality of experience;Semantic communication;Resource management;Array signal processing;Wireless networks;Artificial neural networks;Accuracy;Vectors;Interference;Deep reinforcement learning;semantic-aware network;semantic communications;semantic compression;beamforming and power allocation},
  doi={10.1109/TVT.2025.3571485}}
```
