# Aurora on Isambard-AI

Based on ../baskerville setup, but adapted for Isambard-AI.

https://microsoft.github.io/aurora/example_era5.html

## Set up

Clone the repository:
```
git clone --recursive https://github.com/alan-turing-institute/aurora-hpc.git
cd aurora-hpc/isambard-ai/era5-experiments
```

Get your API key from the Climate Data Store (see the page linked above).
Store it in the `cdsapi.config` file by running the following, replacing APIKEY with your actual API key.

```
printf "%s%s\n" "$(cat cdsapi.config.example)" "APIKEY" > cdsapi.config
```

## Set up conda environment

Prerequisites:

Installing mini-forge using the following instructions: https://docs.isambard.ac.uk/user-documentation/guides/python/#conda-installing-and-using-miniforge

```
sbatch batch-conda.sh
```

## Download the data

Note: In order to be able to download ERA5 data, you need to accept the terms of use in the dataset [page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download).

```
sbatch batch-download.sh
```

**TODO/FIX/NOTE**: Isambard-AI could error when trying to download the data with the `cdsapi` client for more than one day.

Error looks as follows:
```
requests.exceptions.SSLError: HTTPSConnectionPool(host='cds.climate.copernicus.eu', port=443): Max retries exceeded with url: /api/retrieve/v1/processes/reanalysis-era5-single-levels (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)')))
```

This is due to the fact that the `cdsapi` client is trying to make a request which needs to be prepared by the server and the preparation time depends on the amount of data requested. If the request takes too long, the server closes the connection and the client raises an error.

A workaround is to use a local machine to make the data request and wait until it is ready, only then try to download it on Isambard-AI.


## Perform the prediction

```
sbatch batch-runmodel.sh
```

## Fine-tuning the small model

For fine-tuning the same data download can be used.
You can then immediately perform finetuning with the small (debug) modeul on a 40 GiB A100 with the following.

```
sbatch batch-finetune-small.sh
```

## Fine-tuning the model

```
sbatch batch-finetune.sh
```

This should return an out of memory error.
