Change Log / Release Log for PIConGPU
================================================================

Open Beta RC5
-------------
**Date:** 2014-05-28

This is the 5th release candidate, a *pre-beta* version.

We rebuild our complete plugin and restart scheme, most of these
changes are not backwards compatible and you will have to upgrade
to libSplash 1.2+ for HDF5 output (this just means: you can
not restart from a beta-rc4 checkpoint with this release).

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
   - new 2D phase space analysis added #347 #364 #391 #407
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
