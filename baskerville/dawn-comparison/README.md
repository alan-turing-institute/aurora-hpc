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
```bash
sbatch batch-comparison.py
```

## Manual graph generation

While working on the graphs it can be convenient to run the graph generation script manually.
This can be done in an `srun` shell:
```bash
srun --qos turing --account usjs9456-ati-test --time 10:00:00 --nodes 1 \\
  --gpus 1 --cpus-per-gpu 36 --mem 65536 --pty /bin/bash
```

Then source the following file to set up the environment:
```bash
. ./batch-srun.sh
```

Finally run the graph generation script.
The value 4 passed in as the `-n` parameter is the number of `preds` files to use.
In general this should be left as four to match the files generated as explained above.
```bash
python compare-results.py -d "../../downloads" -i "pdf" -n 4
```

## Output graphs

Graphs will be output in in the format spacified on the command line for the `-i` parameter.
If you followed the above steps these will be in PDF format (PNG and SVG are also supported).
```
plot-errors.pdf
plot-error-comparison.pdf
plot-losses.pdf
plot-pvg-bask.pdf
plot-pvg-dawn.pdf
plot-std-dev-comparison.pdf
plot-var-losses.pdf
plot-weatherbench-comparison.pdf
```
