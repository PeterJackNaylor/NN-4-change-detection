# NN-4-change-detection

This repository contains all the necessary to reproduce the work "*IMPERSONATE: IMPlicit nEural RepreSentatiON chAnge deTEction*" by anonymous authors.


<!-- Peter Naylor, Diego Di Carlo, Arianna Traviglia, Makoto Yamada and Marco Fiorucci. -->
<!-- You can find the paper [here](not working) (UNPUBLISHED). -->

# Usage

## Installation
Most of the required packages are listed in the `environment.yml` file and [Nextflow](https://www.nextflow.io/) should be installed to successfully launch the pipeline.

## Nextflow configuration
Please set the nextflow.config file to match your environment.
We use two profiles, one for our local environment (debug and prototyping), and a second for the cluster when we launch large experiments.
## Configuration files

This files can be found in the `exp_config` folder and present themselves in a yaml file.
The first parameters correspond to the config file itself, the output folder, the data path and format.
The next parameter control the training and hyper-parameters.
In particular, a hyperparameter is a categorical list or a range given by a min and a max.

For each Nextflow file, process will be spawned for each `single_method`, `double_method` and `feature_method`.


## `txt` files

The pipeline accepts `txt` files as input in the following format: `NAME_0.txt` and `NAME_1.txt`.
Modify the variable `paired_txt` in the file `nextflow/main.nf`.
Please set the `extension` parameter to `txt`.

## `ply` files

The pipeline accepts `ply` files as input in the following format: `NAME_0.ply` and `NAME_1.ply`.
Please set the `extension` parameter to `ply`.

## Available experiment

We make available two experiments:

### Benchmark
To reproduce the tables presented in the paper and the plots, we first have to run for each configuration and dataset:
```
make paper_home
```
This command will produce a csv file `result/benchmark.csv`



```
make paper_ablation
```

## Archaeological looting experiment




## File plots


## Output files

They can be found in the csv file `result/benchmark.csv`.
