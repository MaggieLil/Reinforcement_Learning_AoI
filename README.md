# Reinforcement Learning for Age of Information Minimization

University project focused on applying reinforcement learning to minimize the **Age of Information (AoI)** in an energy harvesting wireless communication system.

The project is based on the research paper "Learning to Minimize Age of Information over an Unreliable Channel with Energy Harvesting."

## Project Overview

In wireless communication systems, it is important not only to successfully deliver data, but also to keep the information at the receiver up to date.

The **Age of Information (AoI)** measures how old the currently available information is. The main objective of this project is to learn a transmission policy that minimizes the long-term average AoI while taking into account:

* limited battery capacity,
* stochastic energy harvesting,
* unreliable wireless transmission,
* the trade-off between transmitting and saving energy.

## System Model

The environment represents an energy harvesting communication system with:

* a finite battery,
* stochastic energy arrivals,
* probabilistic transmission success,
* an AoI state,
* two possible actions: **idle** or **transmit**.

The state is represented by:

```text
(battery level, AoI)
```

A transmission is possible only when enough energy is available.

The immediate cost is the current AoI, and reinforcement learning is used to learn a policy that minimizes the long-term average cost.

## Implemented Methods

The project implements and compares the following approaches:

1. **GR-learning with Softmax exploration**
2. **GR-learning with epsilon-greedy exploration**
3. **Q-learning with epsilon-greedy exploration**
4. **Greedy baseline**

### GR-learning

GR-learning is an average-cost reinforcement learning approach. Instead of optimizing a discounted cumulative reward, it directly estimates the long-term average cost.

The implementation maintains:

* a state-action value function `Q`,
* an estimate of the average cost `J`,
* state-action visit counts,
* an exploration policy.

### Q-learning

Q-learning uses discounted cumulative rewards. Since the environment is formulated as a cost minimization problem, the AoI cost is converted into a negative reward:

```text
reward = -AoI
```

The Q-learning approach uses epsilon-greedy exploration.

## Exploration Strategies

Two exploration strategies are evaluated:

### Softmax

The Softmax policy assigns higher probability to actions with lower estimated cost. The exploration temperature gradually decreases during learning.

### Epsilon-greedy

With probability `epsilon`, a random feasible action is selected. Otherwise, the currently best estimated action is selected.

The epsilon value is gradually reduced during learning.

## Experiments

Each algorithm is evaluated over multiple independent runs using the same random seeds for a given run, allowing a fair comparison of the stochastic environment.

For each run, the following metrics are recorded:

* cumulative average AoI,
* moving average AoI,
* learned average cost for GR-learning,
* average cost for Q-learning,
* Softmax temperature,
* epsilon value.

The results from the independent runs are averaged to obtain the final performance curves.

## Results

The experiments compare the long-term AoI performance of all four approaches.

The main comparison is based on the cumulative average AoI. Lower values indicate better performance.

### Cumulative Average AoI

![Cumulative Average AoI](results/DEFAULT/comparison_cumulative.png)

The experimental results show that **GR-learning with Softmax exploration achieves the lowest average AoI** among the evaluated methods. GR-learning with epsilon-greedy performs slightly worse, while the greedy baseline and Q-learning achieve higher average AoI in the conducted experiments.

## Technologies

* Python
* NumPy
* Matplotlib
* Reinforcement Learning
* Git / GitHub

## Project Structure

```text
Reinforcement-Learning-AoI/
│
├── agent.py
├── config.py
├── environment.py
├── gr_agent.py
├── main.py
├── policy.py
├── plots.py
├── q_agent.py
├── utils.py
│
├── results/
│   └── default/
│       ├── comparison_cumulative.png
│       ├── comparison_moving.png
│       ├── config.csv
│       ├── gr_epsilon.csv
│       ├── gr_softmax.csv
│       ├── gr_softmax_vs_greedy.png
│       ├── gr_softmax_vs_epsilon.png
│       └── q_learning.csv
│
├── requirements.txt
├── README.md
└── LICENSE
```

## How to Run

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

The simulation results and plots are saved in the `results/` directory.

## Reference

This project is based on the following research paper:

> Elif Tugce Ceran, Deniz Gunduz, and Andras Gyorgy,
> *"Learning to Minimize Age of Information over an Unreliable Channel with Energy Harvesting,"*
> arXiv:2106.16037v1, 2021.
> [Paper on arXiv](https://arxiv.org/abs/2106.16037v1)

The paper provides the theoretical model and reinforcement learning approach, while this repository contains our Python implementation and experimental evaluation.

The implementation in this repository was developed as part of a university project and is intended for educational and experimental purposes.
