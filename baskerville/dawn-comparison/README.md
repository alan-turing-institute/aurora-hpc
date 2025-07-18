# Comparison Utilities

This directory contains useful utilities for performing a comparison between DAWN and Baskerville results.

## Set up

### Dawn results

In order to generate the graphs, you must first ensure the DAWN results are copied into this directory.
You should have 4 files with the following filenames:
```
preds_{i}-dawn.pkl
```

### Baskerville results

To generate the Baskerville results, run the `inference-timing.py` script using the `batch-inference-timing.sh` batch script:
```bash
sbatch batch-inference-timing.sh
```

This will create 4 files in the current directory with the following filenames:
```preds_{i}.pkl
```

You should rename them to:
```
preds_{i}-bask.pkl
```

## Generating results and graphs

To run the script to generate the results on Baskerville and output the graphs, use the following:
```
sbatch batch-comparison.py
```

## Manual graph generation

While working on the graphs it can be convenient to run the graph generation script manually.
This can be done in an `srun` shell:
```
srun --qos turing --account usjs9456-ati-test --time 10:00:00 --nodes 1 \\
  --gpus 1 --cpus-per-gpu 36 --mem 65536 --pty /bin/bash
```

Then source the following file to set up the environment:
```
. ./batch-srun.sh
```

Finally run the graph generation script. Any errors will cause the `srun` session to abort, so we block error return values when running this for convenience during development.
```
python compare-results.py || true
```

## Output graphs

Graphs will be output in both PNG and PDF format, as the following files:
```
plot-errors.pdf
plot-errors.png
plot-losses.pdf
plot-losses.png
plot-pvg-bask.pdf
plot-pvg-bask.png
plot-pvg-dawn.pdf
plot-pvg-dawn.png
plot-var-losses.pdf
plot-var-losses.png
```
