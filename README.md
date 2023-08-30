# NN-4-change-detection

This repository contains all the necessary to reproduce the work "*Implicit neural representation for change detection*" by P. Naylor, D. Di Carlo, A. Traviglia, M. Yamada and M. Fiorucci.
You can find the paper [here](https://arxiv.org/abs/2307.15428).

# Usage

## Installation
Most of the required packages are listed in the `environment.yml` file, and [Nextflow](https://www.nextflow.io/) should be installed to launch the pipeline successfully.

## Nextflow configuration
Please set the nextflow.config file to match your environment.
We use two profiles, one for our local environment (debugging and prototyping) and a second for the cluster when we launch extensive experiments.
## Configuration files

These files can be found in the `exp_config` folder and are YAML files.
The first parameters correspond to the config file, the output folder, the data path and the format.
The following parameters control the training and hyper-parameters.
In particular, a hyperparameter is a categorical list or a range given by a min and a max.

For each Nextflow file, a process will be spawned for each `single_method`, `double_method` and `feature_method`.


## `txt` files

The pipeline accepts `txt` files as input in the following format: `NAME_0.txt` and `NAME_1.txt`.
Modify the variable `paired_txt` in the file `nextflow/main.nf`.
Please set the `extension` parameter to `txt`.

## `ply` files

The pipeline accepts `ply` files as input in the following format: `NAME_0.ply` and `NAME_1.ply`.
Please set the `extension` parameter to `ply`.

## Available experiment

We make available two experiments:

### Simulated airborne LiDAR dataset

The data UrbDCD can be accessed [here](https://ieee-dataport.org/open-access/urb3dcd-urban-point-clouds-simulated-dataset-3d-change-detection). We used version 1 and 2 for the paper.

In particular, it is UrbDCD-v2 for the colab.
### Benchmark
To reproduce the tables presented in the paper and the plots, we first have to run for each configuration and dataset:
```
make paper_home
```
This command will produce a CSV file `result/benchmark.csv` with a line per method, dataset, and associated metrics.
To produce the boxplots shown in Figure 4 and Table 1 and A.1, please run the following:
```
python python/plots/boxplots.py --csv benchmark.csv
```
In addition, in the folder `result/paper`, you should find distribution plots and maps corresponding to Figures 3 and 5.

To run the ablation study presented in Section 5.2, please run the following:
```
make paper_ablation
```
To produce the `benchmark_ablation.csv` file, which contains the metrics for each dataset, configuration and lambda values.
Finally, to produce Figure 6, please run the following:
```
python python/plots/hyper_parameter.py --csv benchmark_ablation.csv
```

# Citation
If you use our code, please cite us!

@misc{naylor2023implicit,
      title={Implicit neural representation for change detection}, 
      author={Peter Naylor and Diego Di Carlo and Arianna Traviglia and Makoto Yamada and Marco Fiorucci},
      year={2023},
      eprint={2307.15428},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
