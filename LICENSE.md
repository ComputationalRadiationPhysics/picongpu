 PIConGPU - Licenses
================================================================================

**Copyright 2009-2014** Florian Berninger, Heiko Burau, Robert Dietrich, Wen Fu,
                        Anton Helm, Wolfgang Hoehnig, Axel Huebl, Remi Lehe,
                        Richard Pausch, Felix Schmitt, Benjamin Schneider,
                        Joseph Schuchart, Klaus Steiniger, Rene Widera

See [active team](README.md#active-team).

PIConGPU is a program collection containing the main simulation, independent
scripts and auxiliary libraries. If not stated otherwise explicitly, the
following licenses apply:

The **main simulation** is licensed under the **GPLv3+**. See
[COPYING](COPYING). If not stated otherwise explicitly, that affects:
 - `buildsystem`
 - `examples`
 - `doc`
 - `src/picongpu`
 - `src/tools` (without `splash2txt`)
 - `src/mpiInfo`
 
All **libraries** are *dual licensed* under the **GLPv3+ and LGPLv3+**. See
[COPYING](COPYING) and [COPYING.LESSER](COPYING.LESSER).
If not stated otherwise explicitly, that affects:
 - `src/libPMacc`
 - `src/tools/splash2txt`

**other licenses**:
 - `src/cuda_memtest`:
   CUDA MemTest is an *independent program* developed by the University
   Illinois, published under the *Illinois Open Source License*.
   Please refer to the file `src/cuda_memtest/README` for license information.
   We redistribute this modified version of CUDA MemTest under the same license
   [src/cuda_memtest/README](src/cuda_memtest/README).
