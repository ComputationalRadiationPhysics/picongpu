#!/bin/bash
pic-build -b serial -c "-DPIC_CI_COMPILE=ON -DCMAKE_BUILD_TYPE=Debug" 2>&1 | tee compile.result
