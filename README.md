# ACiD

Implementation of NeurIPS 2023 paper [ACiD: Accelerating Asynchronous Communication in Decentralized Deep Learning](https://arxiv.org/pdf/2306.08289.pdf).

## Requirements
* [pytorch](https://pytorch.org/)
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)

## Usages
We implement an **Asynchronous Data Parallel** (ADP) model wrapper for distributed training *(analogous to the Distributed Data Parallel (DDP) native to Pytorch)*, with the possibility to apply our ACiD momentum using the ```--apply_acid``` argument.

### Asynchronous Data Parallel

Our ADP wrapper (see [adp.py](https://github.com/AdelNabli/ACiD/blob/main/adp.py)) allows each Neural Network to be hosted on a different GPU, and to perform gradient steps at its own pace in the main thread.
In a seperated thread, peer-to-peer model averagings are performed, requiring to synchronize only 2 models at a time. This is done *in the background, in parallel* of the gradient computations, in opposition to ```All-Reduce``` based methods such as DDP synchronizing all workers and performing communications *after* gradient computations.
The code for training the Neural Network is thus very similar to standard DDP (see [main.py](https://github.com/AdelNabli/ACiD/blob/main/main.py ) ).

### p2p asynchronous communications

Our code handles 3 graphs topologies at this time:
* ```--graph_topology complete```: all edges of the complete graph are considered between all the workers. A separate routine running in the background of worker 0 pairs the first 2 available workers for communications to minimize workers idle time.
* ```--graph_topology exponential```: implement the exponential graph of [SGP](https://arxiv.org/pdf/1811.10792.pdf ) and [AD-PSGD](https://arxiv.org/pdf/1710.06952.pdf) papers. 
* ```--graph_topology cycle```: test a poorly connected graph topology.

For the ```cycle``` and ```exponential``` graph topology, it is possible to set to ```True``` the ```--deterministic_neighbor``` argument. In that case, p2p communications will happen in a predetermined order by cycling through the edges *(e.g., for the cycle graph, we will force every other edge to "spike" and then the complementary ones).* If ```False```, when a worker is available for its next communication, it will communicate with the first of its neighbors it sees available, reducing idle time.

* ```--rate_com``` governs how many p2p averaging happen for each worker between 2 gradient computations. In particular, a value < 1 could be set to perform (a stochastic version of) [local SGD]( https://arxiv.org/abs/1805.09767 ).
* If ```--deterministic_coms``` is set to ```False```, then a Poisson Point Process is implemented, and ```--rate_com``` p2p communication happen between 2 gradients only **in expectation**.

### Large Batch training setting

To obtain a linear speedup of the training time with respect to the number of workers, we do **not** divide our batch-size by the number of workers. Thus, we implement the learning rate scheduler of [Goyal et al. 2017](https://arxiv.org/abs/1706.02677) scaling ```lr``` proportionally to the number of workers (set ```--use_linear_scaling```), and do not apply weight-decay to the biases and batch norm learnable parameters (set ```--filter_bias_and_bn ```). Our training finishes when the **sum of all samples seen by all workers** corresponds to the number set with the ```--n_epoch_if_1_worker``` argument. GPUs not computing gradient at the same speed, this inevitably means that the fastest workers will perform more gradient steps than the slowest ones, reducing training time. We perform a global average of our models before, and after our training.

An example script to launch a SLURM job for training ResNet18 on CIFAR10 using 32 GPUs is provided in [adp.slurm](https://github.com/AdelNabli/ACiD/blob/main/adp.slurm). You might want to install the [hostlist]( https://pypi.org/project/hostlist/) package first in that case.

### WARNINGS

* For theoretical reasons, ```--apply_acid``` can only be set to ```True``` for non-complete graph topologies. ACiD hyper-parameters are automatically computed from the Graph's Laplacian using their theoretical values.
* For implementation reasons, our code currently handles ```--apply_acid``` only if SGD with momentum is used (i.e., a ```--momentum``` argument different than 0).

## Run this code on Jean-Zay

* Change the conda environment path in [adp.slurm](https://github.com/AdelNabli/ACiD/blob/main/adp.slurm), and add the ```#SBATCH -A xxx@a100``` line to link to your A100 account.
* Change the path to CIFAR10 dataset in the ```data_loader``` function of [data_utils.py]( https://github.com/AdelNabli/ACiD/blob/main/utils/data_utils.py ).

## Citation
```bibtex
@misc{nabli2023,
    title={$\textbf{A}^2\textbf{CiD}^2$: Accelerating Asynchronous Communication in Decentralized Deep Learning},
    author={Adel Nabli and Eugene Belilovsky and Edouard Oyallon},
    year={2023},
    eprint={2306.08289},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Acknowledgement

This work was granted access to the HPC/AI resources of IDRIS under the allocation AD011013743 made by GENCI. As our code was developped on [Jean-Zay](http://www.idris.fr/eng/jean-zay/), it is designed for a SLURM based cluster.
We also thank [Louis Fournier](https://github.com/fournierlouis ) for his helpful insights for the design of this version of ADP.