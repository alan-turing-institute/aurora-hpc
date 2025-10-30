# Aurora ERA5 Preduction example

https://microsoft.github.io/aurora/example_era5.html

## Set up

Clone the repository:
```bash
git clone --recursive https://github.com/alan-turing-institute/aurora-hpc.git
cd aurora-hpc/baskerville/era5-prediction
```

Get your API key from the Climate Data Store (see the page linked above).
Store it in the `cdsapi.config` file by running the following, replacing APIKEY with your actual API key.

```bash
printf "%s%s\n" "$(cat cdsapi.config.example)" "APIKEY" > cdsapi.config
```

## Interactive session

The instructions in the following sections explain how to run the experiments using queued tasks using `sbatch`.
However many of these can also be run within an interactive `srun` session, which can be convenient during development.
Setting up a session for use with the scripts can be done as follows.

```bash
srun --qos turing --account usjs9456-ati-test --time 1:00:00 \
    --nodes 1 --gpus 1 --cpus-per-gpu 36 --mem 0 --pty /bin/bash
. ./batch-srun.sh
```

This will set up modules, environment and virtual environment.
You can then run scripts directly, for example:

```bash
python download.py
```

## Download the data

This will download the data to the `aurora-hpc/downloads` directory.

```bash
sbatch batch-download.sh
```

## Perform the prediction

```bash
sbatch batch-runmodel.sh
```

## Display the resulting image

Assuming you have X-forwarding enabled on your Baskerville session you can display the resulting image on your local machine by running the following.

```bash
module load ImageMagick/7.1.0-37-GCCcore-11.3.0
magick display plots.pdf
```

## Fine-tuning the small model

For fine-tuning the same data download can be used.
You can then immediately perform finetuning with the small (debug) modeul on a 40 GiB A100 with the following.

```bash
sbatch batch-finetune-small.sh
```

## Fine-tuning the standard model

There are four versions of the fine tuning process for the standard model: DDP, FSDP, Aligned and a preliminary version.
The last of these is for historical interest and shows the development of the process, but won't run on Baskerville A100 with80 GiB of memory due to out of memory errors.
This preliminary version uses a simplified loss function rather than the loss function specified in the paper and which is likely to be the source of these errors.

To test out the different versious the following commands can be used:

```bash
sbatch batch-finetune-ddp.sh
sbatch batch-finetune-fsdp.sh
sbatch batch-finetune-aligned.sh
sbatch batch-finetune.sh
```
