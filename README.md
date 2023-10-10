# ACiD

Implementation of NeurIPS 2023 paper [ACiD: Accelerating Asynchronous Communication in Decentralized Deep Learning](https://arxiv.org/pdf/2306.08289.pdf). \

## Requirements
* [pytorch](https://pytorch.org/)
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)

## Usages
We implement an **Asynchronous Data Parallel** (ADP) model wrapper for distributed training *(analogous to the Distributed Data Parallel (DDP) native to Pytorch)*, with the possibility to apply our ACiD momentum using the ```--apply_acid``` argument.

### Asynchronous Data Parallel

Our ADP wrapper (see [adp.py](https://github.com/AdelNabli/ACiD/blob/main/adp.py)) allows each Neural Network to be hosted on a different GPU, and perform gradient steps at its own pace in the main thread.
In a seperated thread, peer-to-peer model averaging are performed, requiring to synchronize only 2 models at at time. This is done *in the background, in parallel* of the gradient computations, in opposition to ```All-Reduce``` based methods such as DPP synchronizing everybody and performing communications *after* gradient computations.
The code for training the Neural Network is thus very similar to standard DDP (see [main.py](https://github.com/AdelNabli/ACiD/blob/main/main.py ) ).

### p2p asynchronous communications

Our code handles at the time 3 graphs topologies:
* ```--graph_topology complete```: all edges of the complete graph are considered between all the workers. A separate routine running in the background of worker 0 pairs the first 2 available workers for communications to minimize workers idle time.
* ```--graph_topology exponential```: implement the exponential graph of [SGP](https://arxiv.org/pdf/1811.10792.pdf ) and [AD-PSGD](https://arxiv.org/pdf/1710.06952.pdf) papers. 
* ```--graph_topology cyle```: test a poorly connected graph topology.
**WARNING:** for theoretical reasons, ```--apply_acid``` can only be set to ```True``` for non-complete graph topologies. ACiD hyper-parameters are automatically computed from the Graph's Laplacian using their theoretical values.

For the ```cycle``` and ```exponential``` graph topology, it is possible to set to ```True``` the ```--deterministic_neighbor``` argument. In that case, p2p communications will happen in a predetermined order by cycling through the edges *(e.g., for the cycle graph, we will force every other edge to "spike" and then the complementary ones).* If ```False```, when a worker is available for its next communication, it will communicate with the first of its neighbors it sees available, reducing idle time.

* ```--rate_com``` governs how many p2p averaging happen for each worker between 2 gradient computations.
* if ```--deterministic_coms``` is set to ```False```, then a Poisson Point Process is implemented, and ```--rate_com``` p2p communication happen between 2 gradients only **in expectation**.

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