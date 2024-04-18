# Shadowgraphy Test

## Description
Simulate a Gaussian pulse propagating in z direction to quantify the Shadowgraphy plugin performance
the following values to the expectation value: energy in shadowgram, position of peak in shadowgram,
width of Gaussian pulse, bandwidth of all field components, and mean frequency of all field components.

## Python Environment
To run this test you need a python environment with:
- numpy
- scipy
- openpmd-api
- adios2
- blosc

## Executing the test
```bash
source ~/profiles/picongpu.profile
pic-create $PICSRC/share/picongpu/tests/Shadowgraphy/ PATH/TO/SHADOWGRAPHY/TEST
cd PATH/TO/SHADOWGRAPHY/TEST
./bin/ci.sh
```
