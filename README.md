

# Recommend

K. Yu, Q. He, C. Yu, X. Yang and G. Wu, "Distributed Hierarchical Deep Reinforcement Learning for Semantic-aware Resource Allocation," in IEEE Transactions on Vehicular Technology, doi: 10.1109/TVT.2025.3571485.

If you find the code is useful for your research, please cite the aforementioned paper.


# Distributed Hierarchical Deep Reinforcement Learning for Semantic-aware Resource Allocation

Tensorflow implementation of the [paper](https://ieeexplore.ieee.org/document/11006945) "Distributed Hierarchical Deep Reinforcement Learning for Semantic-aware Resource Allocation". Part of this work has been presented at the IEEE International Conference on Communications 2023 [paper](https://ieeexplore.ieee.org/document/10279009)

# Introduction
Beyond the traditional quality of experience (QoE) optimization that focuses on bit transmission, taking into account semantic QoE can better optimize the network to improve user experience. This paper develops a novel distributed hierarchical deep reinforcement learning (DHDRL) framework with a two-layer control strategy in different timescales to optimize QoE in the downlink multi-cell network including the semantic rate and accuracy. Specifically, we decompose the optimization problem into two sub-problems, beamforming and power allocation as well as semantic resource allocation, resulting in an infinite-horizon discrete and a finite-horizon continuous-time Markov decision processes. Two-layer neural networks (NNs) are proposed to solve the sub-problems. At the network level, a distributed high-level NN is proposed to mitigate the inter-cell interference and improve the network performance on a large timescale, while at the link level, another distributed low-level NN to improve the semantic compression rate and meet various semantic accuracy and rate requirements on a small timescale. Numerical results show that DHDRL achieves at least a 19.07% performance gain over the single-layer method, maintains effectiveness in high-dimensional scenarios, and exhibits strong generalization across a variety of multi-objective optimization problems.

![ ](./figure/DHDRL-SystemModel.pdf)
>  Illustration of the proposed semantic-awaremulti-cell communication network.
![ ](./figure/MultiTimeBPAandSRA.pdf)
>  Illustration of the proposed multi-timescale intelligent BPA and SRA strategy for the semantic-aware wireless networks.

![ ](./figure/HierLeanringFramework.pdf)
>   Illustration of distributed hierarchical intelligent decision-making process.


# Prerequites
* [Python 3.7]


# Quick Start



# Experimental results


## QoE score versus times lots for different schemes, tested in. (a) 3 cells; (b) 7 cells; (c) 19 cells.

![ ](./figure/results_CompressionRate_AWGN.png)
>  Performance at different CRs over the AWGN channel, where the SNR is 10 dB.

![ ](./figure/TimeslotVSqoe_S10_P20_C3-eps-converted-to.pdf)
> (a)

![ ](./figure/TimeslotVSqoe_S10_P20_C7-eps-converted-to.pdf)
> (b)

![ ](./figure/TimeslotVSqoe_S10_P20_C19-eps-converted-to.pdf)
> (c)

## Visualization Results

![ ](./figure/K_heatmap-eps-converted-to.pdf)
>  Heatmap of semantic compression rates for different semantic rate requirements.


# Citation

Please use the following BibTeX citation if you use this repository in your work:

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

# Contact
Please contact 1203304410@qq.com if you have any questions about the codes.
