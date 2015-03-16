InSitu Client (Keyboard Edition)
================================

The keyboard/mouse controlled edition of the live visualization client.

### Prepare Environment

- install rivlib (and vislib/thelib)
- set environment vars

```bash
export RIVLIB_ROOT=<PATH2RIVLIB>
export VISLIB_ROOT=$RIVLIB_ROOT
export THELIB_ROOT=$RIVLIB_ROOT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RIVLIB_ROOT/lib
```

### Configure and Build

Use Qt4!

```bash
# sudo apt-get install qt4-default qt4-qmake
qmake-qt4 SimpleUIVisClient.pro
make
```

On *Hypnos*:
```bash
module load qt/4.8.2
which qmake # version 4 ok? path ok?
qmake SimpleUIVisClient.pro
make
```


### Run

```bash
./SimpleUIVisClient
```


### LICENSE

This tool is part of PIConGPU and licensed accordingly under GPLv3+.

It contains parts of the `PictureFlow` Qt widget from Ariya Hadayat under the
MIT license, see http://pictureflow.googlecode.com
