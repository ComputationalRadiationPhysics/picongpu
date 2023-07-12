#!/bin/bash
./bin/picongpu -g 1 1 1 -d 1 1 1 --periodic 1 1 1 -s 2 2>&1 | tee ../output.result
