#!/bin/bash
./bin/picongpu -g 1 1 1 -d 1 1 1 --periodic 1 1 1 -s 100 --openPMD.period 1 --openPMD.file simOutput --openPMD.ext bp --openPMD.json '{ "adios2": { "engine": { "type": "file", "parameters": { "BufferGrowthFactor": "1.2", "InitialBufferSize": "2GB" } } } }' --Ar_macroParticlesCount.period 1 --eth_macroParticlesCount.period 1 --versionOnce 2>&1 | tee ../output.result
