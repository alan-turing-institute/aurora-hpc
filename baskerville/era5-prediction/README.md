# Aurora ERA5 Preduction example

https://microsoft.github.io/aurora/example_era5.html

## Set up

Clone the repository:
```
git clone --recursive https://github.com/alan-turing-institute/aurora-hpc.git
```

Get your API key from the Climate Data Store (see the page linked above) and store it in the `cdsapi.config` file:

## Download the data

```
sbatch batch-download.sh
```

## Perform the prediction

```
sbatch batch-runmodel.py
```
