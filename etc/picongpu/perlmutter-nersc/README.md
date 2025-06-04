# PIConGPU on Perlmutter
## General Remarks
PIConGPU can be compiled on a login node, but remember to limit the number of used CPU cores. `pic-build -j 16` works fine; reduce the number if the compiler gets killed.

## Installing Dependencies

Before you begin, update the following in the `gpu.profile` file:
- Change the project number (`export proj` variable)
  ```bash
  export proj="m0000"
  ```
- Change path to PIConGPU repository (`PICSRC` variable)
  ```bash
  export PICSRC=$HOME/src/picongpu
  ```

First-Time Setup:
1. Set the `PIC_PROFILE` environment variable to the path of your `gpu.profile` file:  
   ```bash
   export PIC_PROFILE=/path/to/gpu.profile
   ```
2. Install missing dependencies from source with `dependencies_autoinstall.sh`, this can be done on a login node: 
   ```bash
   source dependencies_autoinstall.sh
   ```


Subsequent Use:

-  Simply source the `gpu.profile` each time before using PIConGPU:
   ```bash
   source gpu.profile
   ```


## Preemptive Jobs
When using preemptive queues, you should uncomment the `sleep 120` command at the end of the tpl file.

## Streaming
Note: This is quite an old example; some things may have changed. `gpu_stream.tpl` is an example streaming configuration. It needs to be adjusted (CPU distribution between PIConGPU and reader) to individual needs. The reader has to be provided as `input/bin/reader`. The minimal writer (PIConGPU)/reader JSON OpenPMD configuration is:
```json
{
  "adios2": {
    "engine": {
      "parameters": {
        "DataTransport": "rdma"
      }
    }
  }
}
```
Currently, this requires at least two nodes to ensure that the network interface is available.
