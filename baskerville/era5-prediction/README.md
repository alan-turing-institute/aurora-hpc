# Aurora ERA5 Preduction example

https://microsoft.github.io/aurora/example_era5.html

## Set up

Clone the repository:
```
git clone --recursive https://github.com/alan-turing-institute/aurora-hpc.git
cd aurora-hpc/baskerville/era5-prediction
```

Get your API key from the Climate Data Store (see the page linked above).
Store it in the `cdsapi.config` file by running the following, replacing APIKEY with your actual API key.

```
printf "%s%s\n" "$(cat cdsapi.config.example)" "APIKEY" > cdsapi.config
```

## Download the data

```
sbatch batch-download.sh
```

## Perform the prediction

```
sbatch batch-runmodel.sh
```

## Display the resulting image

Assuming you have X-forwarding enabled on your Baskerville session you can display the resulting image on your local machine by running the following.

```
module load ImageMagick/7.1.0-37-GCCcore-11.3.0
magick display plots.pdf
```
