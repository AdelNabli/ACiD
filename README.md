# ACiD

Implementation of NeurIPS 2023 paper [$A^2CiD^2$: Accelerating Asynchronous Communication in Decentralized Deep Learning](https://arxiv.org/pdf/2306.08289.pdf). \

## Requirements
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [pytorch](https://pytorch.org/)

## Usages
We implement an *Asynchronous Data Parallel* (ADP) model wrapper for distributed training (analogous to the Distributed Data Parallel (DDP) native to Pytorch), with the possibility to add our ACiD momentum using the ```--apply_acid``` argument.

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

This work was granted access to the HPC/AI resources of IDRIS under the allocation AD011013743 made by GENCI. As our code was tested on [Jean-Zay](http://www.idris.fr/eng/jean-zay/), it is catered for a SLURM based cluster.