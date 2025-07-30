# Building and running podman-hpc container

Start by navigating to the container directory
```bash
cd aurora-hpc/container
```

## Build and run the image

### Build the image

Run the following command to build the image:

```bash
podman-hpc build -t aurora-hpc .
```

### Migrate the image

Migration is the process of moving the image to the shared filesystem.
This is needed to run on the compute nodes.

To migrate the image, run the following command:

```bash
podman-hpc migrate aurora-hpc:latest
```

### Run in an interactive shell

If instead you'd like to run in an interactive shell, you can use:

```bash
podman-hpc run -it --gpu --rm --name aurora-hpc \
  localhost/aurora-hpc:latest /bin/bash
```

If you'd like to do this using GPU, you will need to launch an interactive job first using srun:

```bash
srun --gres=gpu:1 -A <project> --time 1:00:00 --pty /bin/bash
```

Then you can run the podman-hpc command above, the `--gpu` flag will ensure that your requested GPUs are available to the container.

