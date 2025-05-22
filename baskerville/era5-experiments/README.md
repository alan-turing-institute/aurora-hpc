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

## Fine-tuning the small model

For fine-tuning the same data download can be used.
You can then immediately perform finetuning with the small (debug) modeul on a 40 GiB A100 with the following.

```
sbatch batch-finetune-small.sh
```

## Fine-tuning the standard model

Currently fine-tuning the standard model fails on an 80 GiB A100 GPU due to out-of-memory errors.
You can try this yourself with the following:

```
sbatch batch-finetune.sh
```

Alternatively to run the same fine-tuning code that works on DAWN, run the following:

```
sbatch batch-finetune-aligned.sh
```

The resulting errors looks like this:

```log
/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
loading model...
loading data...
batching...
preparing model...
performing forward pass...
calculating loss...
performing backward pass...
Traceback (most recent call last):
  File "/bask/projects/u/usjs9456-ati-test/ovau2564/aurora/aurora-hpc/baskerville/era5-prediction/finetune-fsdp.py", line 88, in <module>
    loss.backward()
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/_tensor.py", line 492, in backward
    torch.autograd.backward(
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/autograd/__init__.py", line 251, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/autograd/function.py", line 288, in apply
    return user_fn(self, *args)
           ^^^^^^^^^^^^^^^^^^^^
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 271, in backward
    outputs = ctx.run_function(*detached_inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py", line 153, in my_function
    return self._checkpoint_wrapped_module(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/projects/u/usjs9456-ati-test/ovau2564/aurora/aurora-hpc/aurora/aurora/model/swin3d.py", line 722, in forward
    x = blk(x, c, res, rollout_step)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/projects/u/usjs9456-ati-test/ovau2564/aurora/aurora-hpc/aurora/aurora/model/swin3d.py", line 486, in forward
    attn_windows = self.attn(x_windows, mask=attn_mask, rollout_step=rollout_step)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/apps/live/EL8-ice/software/PyTorch/2.1.2-foss-2023a-CUDA-12.1.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/bask/projects/u/usjs9456-ati-test/ovau2564/aurora/aurora-hpc/aurora/aurora/model/swin3d.py", line 161, in forward
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=attn_dropout)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 570.00 MiB. GPU 0 has a total capacty of 79.25 GiB of which 103.50 MiB is free. Including non-PyTorch memory, this process has 79.14 GiB memory in use. Of the allocated memory 76.38 GiB is allocated by PyTorch, and 2.24 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

