Change Log / Release Log for PIConGPU
================================================================

0.1.0
-----
**Date:** 2015-05-21

This is version `0.1.0` of PIConGPU, a *pre-beta* version.

Initial field ionization support was added, including the first model for BSI.
The code-base was substantially hardened, fixing several minor and major
issues. Especially, several restart related issues, an issue with 2D3V zigzack
current calculation and a memory issue with Jetson TK1 boards were fixed.
A work-around for a critical CUDA 6.5 compiler bug was applied to all affected
parts of the code.

### Changes to "Open Beta RC6"

**.param file changes:**
 See full syntax for each file at
 https://github.com/ComputationalRadiationPhysics/picongpu/tree/0.1.0/src/picongpu/include/simulation_defines/param
 - `componentsConfig.param` & `gasConfig.param` fix typo `gasHomogeneous` #577
 - `physicalConstants.param`: new variable `GAMMA_THRESH` #669
 - `speciesAttributes.param`: new identifier `boundElectrons` and
   new aliases `ionizer`, `atomicNumbers`
 - `ionizationEnergies.param`, `ionizerConfig.param`: added

**.unitless file changes:**
 See full syntax for each file at
 https://github.com/ComputationalRadiationPhysics/picongpu/tree/0.1.0/src/picongpu/include/simulation_defines/unitless
 - `gasConfig.unitless`: typo in `gasHomogeneous` #577
 - `speciesAttributes.unitless`: new unit for `boundElectrons` identifier
 - `speciesDefinition.unitless`: new traits `GetCharge`, `GetMass`,
   `GetChargeState` and added `ionizers`
 - `ionizerConfig.unitless`: added

**New Features:**
 - initial support for field ionization:
   - basic framework and BSI #595
   - attribute (constant flag) for proton and neutron number #687 #731
   - attribute `boundElectrons` #706
 - tools:
   - python scripts:
     - new reader for `SliceFieldPrinter` plugin #578
     - new analyzer tool for numerical heating #672 #692
   - `cuda_memtest`:
     - 32bit host system support (Jetson TK1) #583
     - works without `nvidia-smi`, `grep` or `gawk` - optional with NVML for
       GPU serial number detection (Jetson TK1) #626
   - `splash2txt`:
     - removed build option `S2T_RELEASE` and uses `CMAKE_BUILD_TYPE` #591
   - `tbg`:
     - allows for defaults for `-s`, `-t`, `-c` via env vars #613 #622
   - 3D live visualization: `server` tool that collects `clients` and
     simulations was published #641
 - new/updated particle traits and attributes:
   - `getCharge`, `getMass` #596
   - attributes are now automatically initialized to their generic
     defaults #607 #615
 - libPMacc:
   - machine-dependent `UInt` vector class is now split in explicit
     `UInt32` and `UInt64` classes #665
   - nvidia random number generators (RNG) refactored #711
 - plugins:
   - background fields do now affect plugins/outputs #600
   - `Radiation` uses/requires HDF5 output #419 #610 #628 #646 #716
   - `SliceFieldPrinter` supports `FieldJ`, output in one file,
     updated command-line syntax #548
   - `CountParticles`, `EnergyFields`, `EnergyParticles` support restarts
     without overwriting their previous output #636 #703

**Bug Fixes:**
 - CUDA 6.5: `int(bool)` casts were broken (affects plugins
   `BinEnergyParticles`, `PhaseSpace` and might had an effect on methods of the
   basic PIC cycle) #570 #651 #656 #657 #678 #680
 - the ZigZag current solver was broken for 2D3V if non-zero
   momentum-components in z direction were used (e.g. warm plasmas or
   purely transversal KHI) #823
 - host-device-shared memory (SoC) support was broken (Jetson TK1) #633
 - boost 1.56.0+ support via `Resolve<T>` trait #588 #593 #594
 - potential race condition in field update and pusher #604
 - using `--gridDist` could cause a segfault when adding additional arguments,
   e.g., in 2D3V setups #638
 - `MessageHeader` (used in `png` and 2D live visualization) leaked memory #683
 - restarts with HDF5:
   - static load-balancing via `--gridDist` in y-direction was broken #639
   - parallel setups with particle-empty GPUs hung with HDF5 #609 #611 #642
   - 2D3V field reads were broken (each field's z-component was not initialized
     with the checkpointed values again, e.g., `B_z`) #688 #689
   - loading more than 4 billion global particles was potentially broken #721
 - plugins:
   - `Visualization` (png & 2D live sim) memory bug in double precision runs #621
   - `ADIOS`
     - storing more than 4 billion particles was broken #666
     - default of `adios.aggregators` was broken (now = MPI_Size) #662
     - parallel setups with particle-empty GPUs did hang #661
   - `HDF5`/`ADIOS` output of grid-mapped particle energy for non-relativistic
     particles was zero #669
 - libPMacc:
   - CMake: path detection could fail #796 #808
   - `DeviceBuffer<*,DIM3>::getPointer()` was broken (does not affect
     PIConGPU) #647
   - empty super-cell memory foot print reduced #648
   - `float2int` return type should be int #623
   - CUDA 7:
     - cuSTL prefixed templates with `_` are not allowed; usage of static dim
       member #630
     - explicit call to `template`-ed `operator()` to avoid waring #750
     - `EnvironmentController` caused a warning about `extendend friend syntax` #644
   - multi-GPU nodes might fail to start up when not using `default` compute
     mode with CUDA 7 drivers #643

**Misc:**
 - HDF5 support requires libSplash 1.2.4+ #642 #715
 - various code clean-up for MSVC #563 #564 #566 #624 #625
 - plugins:
   - removed `LineSliceFields` #590
   - `png` plugin write speedup ~2.3x by increasing file size about 12% #698
 - updated contribution guidelines, install, cfg examples #601 #598 #617 #620
   #673 #700 #714
 - updated module examples and cfg files for:
   - lawrencium (LBL) #612
   - titan (ORNL) #618
   - hypnos (HZDR) #670
 - an `Empty` example was added, which defaults to the setup given by
   all `.param` files in default mode (a standard PIC cycle without lasers nor
   particles), see `src/picongpu/include/simulation_defines/` #634
 - some source files had wrong file permissions #668


Open Beta RC6
-------------
**Date:** 2014-11-25

This is the 6th release candidate, a *pre-beta* version.

Initial "multiple species" support was added for flexible particles,
but is yet still limited to two species.
The checkpoint system was refactored and unified, also incooperating
extreme high file I/O bandwidth with ADIOS 1.7+ support.
The JetsonTK1 development kit (32bit ARM host side) is now experimentally
supported by libPMacc/PIConGPU.
The *ZigZag* current deposition scheme was implemented providing
40% to 50% speedup over our optimized Esirkepov implementation.

### Changes to "Open Beta RC5"

**.param file changes:**
 - Restructured file output control (HDF5/ADIOS), new `fileOutput.param`
   [#495](https://github.com/ComputationalRadiationPhysics/picongpu/pull/495/files#diff-1)
 - `componentsConfig.param`: particle pushers and current solvers moved to new files:
   - `species.param`: general definitions to change all species at once (pusher, current solver)
   - `pusherConfig.param`: special tweaks for individual particle pushers,
                           forward declarations restructured
   - `particleConfig.param`: shapes moved to `species.param`,
                             still defines initial momentum/temperature
   - `speciesAttributes.param`: defines *unique* attributes that can
                                be used across all particle species
   - `speciesDefinition.param`: finally, assign common attributes from `speciesAttributes.param`
                                and methods from `species.param` to define individual species,
                                also defines a general compile time "list" of all available
                                species
 - `currentConfig.param`: removed (contained only forward declarations)
 - `particleDefinition.param`: removed, now in `speciesAttributes.param`
 - `laserConfig.param`: new polarization/focus sections for plane wave and wave-packet:
     `git diff --ignore-space-change  beta-rc5..beta-rc6 src/picongpu/include/simulation_defines/param/laserConfig.param`
 - `memory.param`: remove `TILE_` globals and define general `SuperCellSize` and `MappingDesc` instead #435

**.unitless file changes:**
 - `fileOutput.unitless`: restructured and moved to `fileOutput.param`
 - `checkpoint.unitless`: removed some includes
 - `currentConfig.unitless`: removed
 - `gasConfig.unitless`: calculate 3D gas density (per volume) and 2D surface charge density (per area) #445
 - `gridConfig.unitless`: include changed
 - `laserConfig.unitless`: added ellipsoid for wave packet
 - `physicalConstatns.unitless`: `GAS_DENSITY_NORMED` fixed for 2D #445
 - `pusherConfig.unitless`: restructured, according to `pusherConfig.param`
 - `memory.unitless`: removed #435
 - `particleDefinition.unitless`: removed
 - `speciesAttributes.unitless`: added, contains traits to access species attributes (e.g., position)
 - `speciesDefinition.unitless`: added, contains traits to access quasi-fixed attributes (e.g., charge/mass)

**New Features:**
 - ZigZag current deposition scheme #436 #476
 - initial multi/generic particle species support #457 #474 #516
 - plugins
   - BinEnergy supports clean restarts without loosing old files #540
   - phase space now works in 2D3V, with arbitrary super cells and
     with multiple species #463 #470 #480
   - radiation: 2D support #527 #530
 - tools
   - splash2txt now supports ADIOS files #531 #545
 - plane wave & wave packet lasers support user-defined polarization #534 #535
 - wave packet lasers can be ellipses #434 #446
 - central restart file to store available checkpoints #455
 - libPMacc
   - added `math::erf` #525
   - experimental 32bit host-side support (JetsonTK1 dev kits) #571
   - `CT::Vector` refactored and new methods added #473
   - cuSTL: better 2D container support #461

**Bug Fixes:**
 - esirkepov + CIC current deposition could cause a deadlock in some situations #475
 - initialization for `kernelSetDrift` was broken (traversal of frame lists, CUDA 5.5+) #538 #539
 - the particleToField deposition (e.g. in FieldTmp solvers for analysis)
   forgot a small fraction of the particle #559
 - libPMacc
   - no `static` keyword for non-storage class functions/members (CUDA 6.5+) #483 #484
   - fix a game-of-life compile error #550
   - ParticleBox `setAsLastFrame`/`setAsFirstFrame` race condition (PIConGPU was not affected) #514
 - tools
   - tbg caused errors on empty variables, tabs, ampersands, comments #485 #488 #528 #529
 - dt/CFL ratio in stdout corrected #512
 - 2D live view: fix out-of-mem access #439 #452

**Misc:**
 - updated module examples and cfg files for:
   - hypnos (HZDR) #573 #575
   - taurus (ZIH/TUDD) #558
   - titan (ORNL) #489 #490 #492
 - Esirkepov register usage (stack frames) reduced #533
 - plugins
   - EnergyFields output refactored and clarified #447 #502
   - warnings fixed #479
   - ADIOS
     - upgraded to 1.7+ support #450 #494
     - meta attributes synchronized with HDF5 output #499
 - tools
   - splash2txt updates
     - requires libSplash 1.2.3+ #565
     - handle exceptions more transparently #556
     - fix listing of data sets #549 #555
     - fix warnings #553
   - BinEnergyPlot: refactored #542
   - memtest: warnings fixed #521
   - pic2xdmf: refactor XDMF output format #503 #504 #505 #506 #507 #508 #509
   - paraview config updated for hypnos #493
 - compile suite
   - reduce verbosity #467
   - remove virtual machine and add access-control list #456 #465
 - upgraded to ADIOS 1.7+ support #450 #494
 - boost 1.55.0 / nvcc <6.5 work around only applied for affected versions #560
 - `boost::mkdir` is now used where necessary to increase portability #460
 - libPMacc
   - `ForEach` refactored #427
   - plugins: `notify()` is now called *before* `checkpoint()` and a getter
              method was added to retrieve the last call's time step #541
   - `DomainInfo` and `SubGrid` refactored and redefined #416 #537
   - event system overhead reduced by 3-5% #536
   - warnings fixed #487 #515
   - cudaSetDeviceFlags: uses `cudaDeviceScheduleSpin` now #481 #482
   - `__delete` makro used more consequently #443
   - static asserts refactored and messages added #437
 - coding style / white space cleanups #520 #522 #519
 - git / GitHub / documentation
   - pyc (compiled python files) are now ignored #526
   - pull requests: description is mandatory #524
 - mallocMC cmake `find_package` module added #468


Open Beta RC5
-------------
**Date:** 2014-06-04

This is the 5th release candidate, a *pre-beta* version.

We rebuild our complete plugin and restart scheme, most of these
changes are not backwards compatible and you will have to upgrade
to libSplash 1.2+ for HDF5 output (this just means: you can
not restart from a beta-rc4 checkpoint with this release).

HDF5 output with libSplash does not contain *ghost*/*guard* data
any more. These information are just necessary for checkpoints
(which are now separated from the regular output).

### Changes to "Open Beta RC4"

**.param file changes:**
 - Added selection of optional window functions in `radiationConfig.param`
   [#286](https://github.com/ComputationalRadiationPhysics/picongpu/pull/286/files#diff-1)
 - Added more window functions in `radiationConfig.param`
   [#320](https://github.com/ComputationalRadiationPhysics/picongpu/pull/320/files#diff-1)
 - removed double `#define __COHERENTINCOHERENTWEIGHTING__ 1` in some examples `radiationConfig.param`
   [#323](https://github.com/ComputationalRadiationPhysics/picongpu/pull/323/files)
 - new file: `seed.param` allows to vary the starting conditions of "identical" runs
   [#353](https://github.com/ComputationalRadiationPhysics/picongpu/pull/353)
 - Updated a huge amount of `.param` files to remove outdated comments
   [#384](https://github.com/ComputationalRadiationPhysics/picongpu/pull/384)
 - Update `gasConfig.param`/`gasConfig.unitless` and doc string in `componentsConfig.param`
   with new gasFromHdf5 profile
   [#280](https://github.com/ComputationalRadiationPhysics/picongpu/pull/280/files)

**.unitless file changes:**
 - update `fileOutput.unitless` and add new file `checkpoints.unitless`
   [#387](https://github.com/ComputationalRadiationPhysics/picongpu/pull/387/files)
 - update `fieldSolver.unitless`
   [#314](https://github.com/ComputationalRadiationPhysics/picongpu/pull/314/files#diff-5)
 - Update `radiationConfig.unitless`: adjust to new supercell size naming
   [#394](https://github.com/ComputationalRadiationPhysics/picongpu/pull/394/files#diff-61)
 - Corrected CFL criteria (to be less conservative) in `gridConfig.unitless`
   [#371](https://github.com/ComputationalRadiationPhysics/picongpu/pull/371/files#diff-1)

**New Features:**
 - Radiation plugin: add optional window functions to reduce ringing effects
   caused by sharp boundaries #286 #323 #320
 - load gas profiles from png #280
 - restart mechanism rebuild #326 #375 #358 #387 #376 #417
 - new unified naming scheme for domain and window sizes/offsets #128 #334 #396 #403 #413 #421
 - base seed for binary idential simulations now exposed in seed.param #351 #353
 - particle kernels without "early returns" #359 #360
 - lowered memory foot print during shiftParticles #367
 - ShiftCoordinateSystem refactored #414
 - tools:
   - tbg warns about broken line continuations in tpl files #259
   - new CMake modules for: ADIOS, libSplash, PNGwriter #271 #304 #307 #308 #406
   - pic2xdmf
     - supports information tags #290 #294
     - one xdmf for grids and one for particles #318 #345
   - Vampir and Score-P support updated/added #293 #291 #399 #422
   - ParaView remote server description for Hypnos (HZDR) added #355 #397
 - plugins
   - former name: "modules" #283
   - completely refactored #287 #336 #342 #344
   - restart capabilites added (partially) #315 #326 #425
   - new 2D phase space analysis added (for 3D sims and one species at a time) #347 #364 #391 #407
   - libSplash 1.2+ upgrade (incompatible output to previous versions) #388 #402
 - libPMacc
   - new Environment class provides all singletons #254 #276 #404 #405
   - new particle traits, methods and flags #279 #306 #311 #314 #312
   - cuSTL ForEach on 1-3D data sets #335
   - cuSTL twistVectorAxes refactored #370
   - NumberOfExchanges replaced numberOfNeighbors implementation #362
   - new math functions: tan, float2int_rd (host) #374 #410
   - CT::Vector now supports ::shrink #392

**Bug fixes:**
 - CUDA 5.5 and 6.0 support was broken #401
 - command line argument parser messages were broken #281 #270 #309
 - avoid deadlock in computeCurrent, remove early returns #359
 - particles that move in the absorbing GUARD are now cleaned up #363
 - CFL criteria fixed (the old one was too conservative) #165 #371 #379
 - non-GPU-aware (old-stable) MPI versions could malform host-side
   pinned/page-locked buffers for subsequent cudaMalloc/cudaFree calls
   (core routines not affected) #438
 - ADIOS
   - particle output was broken #296
   - CMake build was broken #260 #268
 - libSplash
   - output performance drastically improved #297
 - libPMacc
   - GameOfLife example was broken #295
   - log compile broken for high log level #372
   - global reduce did not work for references/const #448
   - cuSTL assign was broken for big data sets #431
   - cuSTL reduce minor memory leak fixed #433
 - compile suite updated and messages escaped #301 #385
 - plugins
   - BinEnergyParticles header corrected #317 #319
   - PNG undefined buffer values fixed #339
   - PNG in 2D did not ignore invalid slides #432
 - examples
   - Kelvin-Helmholtz example box size corrected #352
   - Bunch/SingleParticleRadiationWithLaser observation angle fixed #424

**Misc:**
 - more generic 2 vs 3D algorithms #255
 - experimental PGI support removed #257
 - gcc 4.3 support dropped #264
 - various gcc warnings fixed #266 #284
 - CMake 3.8.12-2 warnings fixed #366
 - picongpu.profile example added for
   - Titan (ORNL) #263
   - Hypnos (HZDR) #415
 - documentation updated #275 #337 #338 #357 #409
 - wiki started: plugins, developer hints, simulation control, examples #288 #321 #328
 - particle interfaces clened up #278
 - ParticleToGrid kernels refactored #329
 - slide log is now part of the SIMULATION_STATE level #354
 - additional NGP current implementation removed #429
 - libPMacc
   - GameOfLife example documented #305
   - compile time vector refactored #349
   - shortened compile time template error messages #277
   - cuSTL inline documentation added #365
   - compile time operators and ForEach refactored #380
   - TVec removed in preference of CT::Vector #394
 - new developers added #331 #373
 - Attribution text updated and BibTex added #428


Open Beta RC4
-------------
**Date:** 2014-03-07

This is the 4th release candidate, a *pre-beta* version.

### Changes to "Open Beta RC3"

**.param file changes:**
 - Removed unnesseary includes #234 from:
   `observer.hpp`, `physicalConstants.param`, `visColorScales.param`,
   `visualization.param`, `particleConfig.param`, `gasConfig.param`,
   `fieldBackground.param`, `particleDefinition.param`
   see the lines that should be removed in
   [#234](https://github.com/ComputationalRadiationPhysics/picongpu/pull/234/files)
 - Renamed `observer.hpp` -> `radiationObserver.param` #237 #241
   Changed variable name `N_theta` to `N_observer`
   https://github.com/ComputationalRadiationPhysics/picongpu/commit/9e487ec30ade10ece44fc19fd7a815b8dfe58f61#diff-9
 - Added background FieldJ (current) capability #245
   Add the following lines to your `fieldBackground.param`:
   https://github.com/ComputationalRadiationPhysics/picongpu/commit/7b22f37c6a58250d6623cfbc821c4f996145aad9#diff-1

**New Features:**
 - 2D support for basic PIC cycle #212
 - hdf5 output xdmf meta description added: ParaView/VisIt support #219
 - background current (FieldJ) can be added now #245

**Bug fixes:**
 - beta-rc3 was broken for some clusters due to an init bug #239
 - examples/WeibelTransverse 4 GPU example was broken #221
 - smooth script was broken for 1D fields #223
 - configure non-existing path did not throw an error #229
 - compile time vector "max" was broken #224
 - cuda_memtest did throw false negatives on hypnos #231 #236
 - plugin "png" did not compile for missing freetype #248

**Misc:**
 - documentation updates
   - radiation post processing scripts #222
   - more meta data in hdf5 output #216
   - tbg help extended and warnings to errors #226
   - doc/PARTICIPATE.md is now GitHub's CONTRIBUTING.md #247 #252
   - slurm interactive queue one-liner added #250
   - developers updated #251
 - clean up / refactoring
   - cell_size -> cellSize #227
   - typeCast -> precisionCast #228
   - param file includes (see above for details) #234
   - DataConnector interface redesign #218 #232
   - Esirkepov implementation "paper-like" #238


Open Beta RC3
-------------
**Date:** 2014-02-14

This is the third release candidate, a *pre-beta* version.

### Changes to "Open Beta RC2"

**.param and .cfg file changes:**
 - `componentsConfig.param`:
   - remove simDim defines #134 #137
     (example how to update your existing `componentsConfig.param`, see
     https://github.com/ComputationalRadiationPhysics/picongpu/commit/af1f20790ad2aa15e6fc2c9a51d8c870437a5fb7)
 - `dimension.param`: new file with simDim setting #134
   - only add this file to your example/test/config if you want to change it
     from the default value (3D)
 - `fieldConfig.param`: renamed to `fieldSolver.param` #131
 - `fieldBackground.param`: new file to add external background fields #131
 - cfg files cleaned up #153 #193

**New Features:**
 - background fields for E and B #131
 - write parallel hdf5 with libSplash 1.1 #141 #151 #156 #191 #196
 - new plugins
   - ADIOS output support #179 #196
   - makroParticleCounter/PerSuperCell #163
 - cuda_memtest can check mapped memory now #173
 - EnergyDensity works for 2-3D now #175
 - new type floatD_X shall be used for position types (2-3D) #184
 - libPMacc
   - new functors for multiplications and substractions #135
   - opened more interfaces to old functors #197
   - MappedMemoryBuffer added #169 #182
   - unary transformations can be performed on DataBox'es now,
     allowing for non-commutative operations in reduces #204

**Bug fixes:**
 - libPMacc
   - GridBuffer could deadlock if called uninitialized #149
   - TaskSetValue was broken for all arrays with x-size != n*256 #174
   - CUDA 6.0 runs crashed during cudaSetDeviceFlags #200
   - extern shared mem could not be used with templated types #199
 - tbg
   - clearify error message if the tpl file does not exist #130
 - HDF5Writer did not write ions any more #188
 - return type of failing Slurm runs fixed #198 #205
 - particles in-cell position fixed with cleaner algorithm #209

**Misc:**
 - documentation improved for
   - cuSTL #116
   - gasConfig.param describe slopes better (no syntax changes) #126
   - agreed on coding guide lines #155 #161 #140
   - example documentation started #160 #162 #176
   - taurus (slurm based HPC cluster) updates #206
 - IDE: ignore Code::Blocks files #125
 - Esirkepov performance improvement by 30% #139
 - MySimulation asserts refactored for nD #187
 - Fields.def with field forward declarations added,
   refactored to provide common ValueType #178
 - icc warnings in cuda_memcheck fixed #210
 - libPMacc
   - refactored math::vector to play with DataSpace #138 #147
   - addLicense script updated #167
   - MPI_CHECK writes to stderr now #168
   - TVec from/to CT::Int conversion #185
   - PositionFilter works for 2-3D now #189 #207
   - DeviceBuffer cudaPitchedPtr handling clean up #186
   - DataBoxDim1Access refactored #202


Open Beta RC2
-------------
**Date:** 2013-11-27

This is the second release candidate, a *pre-beta* version.

### Changes to "Open Beta RC1"

**.param file changes:**
 - `gasConfig.param`:
   - add gasFreeFormula #96
     (example how to update your existing `gasConfig.param`, see
     https://github.com/ComputationalRadiationPhysics/picongpu/pull/96/files#diff-1)
   - add inner radius to gasSphereFlanks #66
     (example how to update your existing `gasConfig.param`, see
     https://github.com/ComputationalRadiationPhysics/picongpu/pull/66/files#diff-0)

**New Features:**
 - A change log was introduced for master releases #93
 - new gas profile "gasFreeFormula" for user defined profiles #96
 - CMake (config) #79
   - checks for minimal required versions of dependent libraries #92
   - checks for libSplash version #85
   - update to v2.8.5+ #52
   - implicit plugin selection: enabled if found #52
   - throw more warnings #37
   - experimental support for icc 12.1 and PGI 13.6 #37
 - libPMacc
   - full rewrite of the way we build particle frames # 86
   - cuSTL: ForEach works on host 1D and 2D data now #44
   - math::pow added #54
   - compile time ForEach added #50
 - libSplash
   - dependency upgraded to beta (v1.0) release #80
   - type traits for types PIConGPU - libSplash #69
   - splash2txt update to beta interfaces #83
 - new particle to grid routines calculating the Larmor energy #68
 - dumping multiple FieldTmp to hdf5 now possible #50
 - new config for SLURM batch system (taurus) #39

**Bug fixes:**
 - libPMacc
   - cuSTL
     - assign was broken for deviceBuffers #103
     - lambda expressions were broken #42 #46 #100
     - icc support was broken #100 #101
     - views were broken #62
   - InheritGenerator and deselect: icc fix #101
   - VampirTrace (CUPTI) support: cudaDeviceReset added #90
   - GameOfLife example fixed #53 #55
   - warnings in __cudaKernel fixed #51
 - picongpu
   - removed all non-ascii chars from job scripts #95 #98
   - CMake
     - keep ptx code was broken #82
     - PGI: string compare broken #75
     - MPI: some libs require to load the C++ dependencies, too #64
     - removed deprecated variables #52
     - Threads: find package was missing #34
   - various libSplash bugs #78 #80 #84
   - current calculation speedup was broken #72
   - Cell2Particle functor missed to provide static methods #49
 - tools
   - compile: script uses -q now implicit for parallel (-j N) tests
   - plotDensity: update to new binary format #47
 - libraries
   - boost 1.55 work around, see trac #9392 (nvcc #391854)

**Misc:**
 - new reference: SC13 paper, Gordon Bell Finals #106
 - new flavoured logo for alpha
 - Compile Suite: GitHub integration #33 #35
 - dropped CUDA sm_13 support (now sm_20+ is required) #42


Open Beta RC1
-------------
**Date:** 2013-09-05 07:47:03 -0700

This is the first release candidate, a *pre-beta* version.
We tagged this state since we started to support `sm_20+` only.

### Changes to "Open Alpha"
n/a


Open Alpha
----------
**Date:** 2013-08-14 02:25:36 -0700

That's our our open alpha release.
The [alpha](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
release is **developer** and **power user** release only!
**Users** should wait for our
[beta](https://github.com/ComputationalRadiationPhysics/picongpu/issues?milestone=2)
release!
