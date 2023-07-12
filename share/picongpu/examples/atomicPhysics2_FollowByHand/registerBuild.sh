#!/bin/bash
pic-build -c "-Dalpaka_CUDA_SHOW_REGISTER=ON -DPMACC_BLOCKING_KERNEL=ON" 2>&1 | tee compile.result
