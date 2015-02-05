Server for PIConGPU 3D Live Simulation
======================================

Collects simulation streams from the cluster, provides access to the image
streams and forwards the camera controls back to the simulation.

### Prepare Environment

- install [rivlib](https://github.com/ComputationalRadiationPhysics/rivlib)
  (includes `vislib` and `thelib++`)
- set environment vars

```bash
export RIVLIB_ROOT=<PATH2RIVLIB>
export VISLIB_ROOT=$RIVLIB_ROOT
export THELIB_ROOT=$RIVLIB_ROOT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RIVLIB_ROOT/lib
```

### Configure and Build

```bash
make
```

### Run

```bash
./server --port 8100
```
