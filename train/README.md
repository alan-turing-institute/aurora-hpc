# Aurora Training Experiments

The code in this folder is for performing various training related experiments.
See below for instructions for how to run them.

## Running within an interactive session

To run the interactives session scripts, first ensure you're on a compute node by running the following or an equiavlent `srun` command (you'll need to update the QoS and account details):

```sh
srun --qos turing --account usjs9456-ati-test --time 1:00:00 --nodes 1 --gpus 1 --cpus-per-gpu 36 --mem 16384 --pty /bin/bash
```

## Queued jobs using sbatch

All sbatch scripts have a QoS and account details set in them.
The parameters used for these will depend on your account and so should be adjusted accordingly.

```
#SBATCH --qos turing
#SBATCH --account usjs9456-ati-test
```

## Baskerville training using FSDP

The case of a single node can be run within an srun interactive session or scheduled using the sbatch scripts.

All scripts should be run from within the `aurora-hpc/train/batch` directory.

To run in an interactive session use:

```sh
./bask-local-train-fsdp.sh
```

Only one node and one GPU is supported interacively (although up to four GPUs would be possible by editing the bash script).
Since the `NO_SHARD` option is set no model sharding is performed, only data parallel, meaning that the strategy is equivalent to Distributed Data Parallel.

The script will set up the environment, including creating an appropriate virtual environment, automatically.
The only part that must be performed manually is downloading of the data.

To schedule the experiments for 1 node with 1 GPUs, 1 node with 4 GPUs, 2 nodes each with 2 GPUs (4 GPUs total), 2 nodes each with 4 GPUs (8 GPUs total) and 4 nodes each with 4 GPUs (16 GPUs total), run the following respectively.:

```sh
sbatch ./bask-train-fsdp-1x1.sh
sbatch ./bask-train-fsdp-1x4.sh
sbatch ./bask-train-fsdp-2x4.sh
sbatch ./bask-train-fsdp-2x8.sh
sbatch ./bask-train-fsdp-4x4.sh
```

These will schedule the jobs to run when the resources become available.
The scripts will set up the virtual environment and everything needed apart from downloading the data.

Each of the scheduled jobs will perform four runs to allow averaging of the results.

If you just want a single run, the following script can be used:

```sh
sbatch bask-train-fsdp-4x4.sh
```

This is set up to run on 2 nodes with 2 GPUs and to perform just a single run.
Edit the script header to test other combinations.

## Baskerville bandwidth

All bandwidth experiments should be run within an interactive session and from within the `aurora-hpc/train/batch` directory.

To run the disk bandwidth experiments:

```sh
./bask-srun-diskbw.sh
```

To run the GPU bandwidth experiments:

```sh
./bask-srun-gpubw.sh
```
