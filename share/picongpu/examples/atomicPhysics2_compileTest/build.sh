#!/bin/bash
pic-build -b serial -c "-DPIC_CI_COMPILE=ON" 2>&1 | tee compile.result
