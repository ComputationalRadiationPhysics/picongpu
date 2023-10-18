Changelog
=========

0.7.0
-----

**Date:** 2023-10-19

C++17, lasers via incidentField, DispersivePulse laser profile, sub-stepping field solver, and [PICMI](https://picmi-standard.github.io/#) support

With this release, the laser initialization switches to an incidentField approach, which allows easier initialization of arbitrary, Maxwell conform electromagnetic fields in a simulation. Old `laser.param` files won't work anymore. See the documentation!
This update of the laser implementation has further been used to add the new profile 'DispersivePulse' and to rename the existing laser profiles.
The new profile allows initialization of Gaussian laser pulses with arbitrary first, second, and third-order dispersion, while the renaming aims at expressing a laser's profile more precisely.

Furthermore, basic [PICMI](https://picmi-standard.github.io/#) support is added to PIConGPU, as well as the possibility of time sub-stepping for field solvers, and improved handling of currents in PML boundaries. Again, please see the documentation for details.

Larger code refactorings come with this release, too. To these belong a cleaning of the code base, in particular removal of unused plugins and CuSTL, performance improvements, and a switch to C++17 as the minimum required version.

As usual, this release features fixes of PIConGPU and extensions and clarifications of the documentation.

Thanks to Sergei Bastrakov, Finn-Ole Carstens, Alexander Debus, Fabia Dietrich,
Simeon Ehrig, Marco Garten, Bernhard Manfred Gruber, Axel Huebl, Jeffrey Kelling, 
Anton Lebedev,Brian Edward Marre, Paweł Ordyna, Richard Pausch, Franz Pöschel, 
Klaus Steiniger, Jan Stephan, Hannes Tröpgen, Mika Soren Voß, René Widera
for their contributions to this release!

**User Input Changes:**
- Remove deprecated laser profiles #4502
- Remove flylite #4360
- Remove FieldToParticleInterpolationNative #4369
- Remove SliceFieldPrinter plugin #4112
- Remove pusher `Axel` #4115
- Remove unused plugin Intensity #4361
- Remove PIConGPUs docker file #4374
- Remove bremsstrahlung.unitless #4494
- Remove deprecated laser profiles #4502
- Incident field laser profiles #4038 #4050 #4053 #4064 #4069 #4080 #4077 #4076 #4120 #4105 #4172 #4302 #4507 #3860
- Remove derived attribute/field MomentumComponent #4191
- Add a new functor for start position and weightings of macroparticles #3882
- Add a runtime option to set particle density file #3972
- Implement phase-coefficients in Gauss-Laguerre laser modes #3992
- Extend comments in density.param #4021
- Comments to declarations of density profile types #4024
- Add EZ current deposition type as alias to EmZ #4036
- Change way of controlling position of the Huygens surface #4056
- Specify BP5 default BufferChunkSize as 2GB #4127
- Unify default openPMD file extension in plugins #4155
- Add derived particle attributes for component of momentum and related density #4169
- Add custom normalization option to PNG output #4165
- Add a new functor to sample macroparticles in a cell #4111
- Auto adjust particle exchange buffers for double precision #4236
- Disallow binomial current interpolation together with current background #4252
- Add free temperature functor #4256
- Change user interface of free temperature functor to match free density functor #4273
- KHI memory species buffer scaling based on simulation volume #4422
- Refactor particle filters #4432
- Replace mpl by mp11 (part 1/2) #3834

**New Features:**
- PIC:
    - Add readthedocs description of TF/SF field generation #3877
    - Add a new functor for start position and weightings of macroparticles #3882
    - Optimize Jz calculation in EmZ 2d #3904
    - Arbitrary order incident field solver #3860 #4038 #4050 #4082 #4105
    - Add a runtime option to set particle density file #3972
    - fix warnings and check for warnings in CI #3878
    - Implement phase-coefficients in Gauss-Laguerre laser modes #3992
    - Add a check and warning printed when no volume inside the Huygens surface #4049
    - Add a warning for auto-adjusting domain to fit field absorber #4047
    - Change way of controlling position of the Huygens surface #4056
    - Check uniqueness of particle attributes #4078
    - Improve reporting of wrong MPI/devices configuration #4122
    - Check that averagely derived attributes are weighted #4154
    - Add derived particle attributes for component of momentum and related density #4169
    - Allow fractional indices in TWTSfast functors #4184
    - Calculate and output laser phase velocity #4174
    - Remove derived attribute/field MomentumComponent #4191
    - HIP: use compute current strategy `strategy::CachedSupercellsScaled<2>` #4218
    - Add a new functor to sample macroparticles in a cell #4111
    - Add time-substepping versions of existing field solvers #3804
    - Add time interpolation/extrapolation of J in substepping field solver  #4258
    - Improve treating currents in PML #4293
    - Add a new particle attribute for weighting damping factor  #4300
    - FLYonPIC atomic physics for PIConGPU #4134
    - fix warning in `Derivative.hpp` #4508
- PMacc:
    - allow time dependent checkpointing #4042
    - Clarify performance warning on send/receive buffer being too small #4171
    - provide a pmacc CMake target  #4257
    - PMacc tests: use the CI provided GPU architecture #4493
    - Replace mpl by mp11 (part 1/2) #3834
- plugins:
    - openPMD plugin: Set default backend depending on available options #4008
    - openPMD plugin: throw error when neither defining period or toml #4028
    - Clarify an inaccurate exception message in calorimeter plugin #4027
    - Specify BP5 default BufferChunkSize as 2GB #4127
    - Add custom normalization option to PNG output #4165
    - OpenPMD plugin data range selection #4096
    - Switch to .bp4 extension for ADIOS2 if available, print available extensions in help message #4234
- tools:
    - Crusher hipcc Profile: Fix export of env variables #3976
    - Crusher hipcc profile #3973
    - Fix EnergyHistogram and clarify usage #3970
    - Fix PICSRC in bash_picongpu.profile #3999
    - Enable --mpiDirect for crusher-ornl #4007
    - Add profiles for A100 GPUs on Taurus #3998
    - hzdr dev server profile #4086
    - crusher: add slurm fault tolerance #4230
    - `pic-configure` validate preset selection #4235
    - crusher tpl: avoid false error message #4261
    - Activate cuda_memtest on crusher #4216 
    - Add template for async jobs on Crusher #4413
    - Async template for Juwels Booster, Crusher and Summit #4431
    - Added script to edit copied PIConGPU checkpoints. #4520
    - gpu profile for dev-server #4183

**Bug Fixes:**
- PIC:
    - Fix missing euler number in ADK formula #4696
    - Fix incorrect assignment of Jz in 2d EmZ implementation #3892
    - Fix getExternalCellsTotal() to properly account for guard cells #4030
    - Fix incident field math around corners #4102
    - Fix index shift in 2d incident field kernel #4202
    - Fix fieldBackground constructors to be HINLINE #4013
    - Better separate CFL and pusher/current deposition requirements #4237
    - Skip J contribution to E for None field solver #4243
    - Properly account for current background when deciding to skip adding J #4247
    - Disallow binomial current interpolation together with current background #4252
    - Change TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE to always be 64-bit #4263
    - Update binary collisions, dynamic Coulomb log #4295
    - Remove explicit definition of RngType in filter::generic::FreeRng #4440
    - Fix particle position rounding issue #4463
- PMacc:
    - Fix copying of cuSTL device buffers #4029
    - Fix function `toTimeSlice` #4040
    - Fix implementation of pmacc::makeDeepCopy() and field background #4192
    - Fix soft restart #4034
    - Fix PMacc reduce #4328
- plugins:
    - Add missing communication in XrayScattering #3937
    - Fix rad restart #3961
    - Fix calculation of total current in the SumCurrents plugin for 2d case #4118
    - Fix radiation trees #4231
- tools:
    - Add missing double quotes in handleSlurmSignals #4161
    - Fix wrong memory requirment calculation in .tpl files #4308
    - Fix PICSRC in bash_picongpu.profile #3999
- Fix that PMacc depends on PIConGPU type definitions #3925
- HIP: allow full architecture definition #3941
- Make the span-based storeChunk API opt-in in various openPMD-based IO routines #3933
- Fix openPMD outputs negative particle patch offset #4037
- Fix CUDA container apt keys #4085
- Add HIP and mallocMC + HIP to PIConGPU version output #4139
- Do not forceinline recursive functions #4147
- PNG plugin: fix invalid memory access in 2D #4225
- Fix using uninitilized variable #4303
- Fix openPMD particles `positionOffset` **breaking change** #4283
- Fix precision cast is returning a reference #4337

**Misc:**
- Refactoring:
    - PIC:
        - Refactor absorbing and reflecting particle BC #3875
        - Move yee::StencilFunctor to fdtd::StencilFunctor #3894
        - Improve performance of MoveParticle #3908
        - Cleanup current deposition implementation #3944
        - Use `if constexpr()` to check for simulation/object dimension #4107
        - Avoid using macro define `SIMDIM` #4116
        - Switch to alpaka::Complex<> #4114
        - Add a helper type for unique typelist of enabled incident field profiles #4177
        - Cached shapes #4124
        - Use cached shapes for current deposition #4170
        - Add device side `ForEachParticle` #4173
        - Workaround wrong reported free memory #4219
        - Simplify implementation of incident field kernel #4226
        - Auto adjust particle exchange buffers for double precision #4236
        - Make adding J term part of field solver #4245
        - Refactor adding current density to electomagnetic field #4255
        - Add free temperature functor #4256
        - Add singleton-like access for AbsorberImpl  #4262
        - Simplify PmlImpl interface for getting update functors #4266
        - Remove Cell2Particle #4289
        - Remove getGlobalSuperCells #4325
        - Move include collision.unitless to the default loader #4326
        - Refactor lockstep programming #4321
        - Switch to std::optional in compute field value #4345
        - Use cupla/alpaka's min/max function #4350
        - Remove flylite #4360
        - Remove cuSTL #4363 #4398 #4402
        - Remove Bremsstrahlung and Synchrotron radition extension #4395
        - Remove unused vectors transition radiation #4427
        - Fix incident field calculation order #4451
        - Improve command line argument validation #4447
        - Change class RelayPoint to function #4464
        - OnePositionImpl refactoring #4465
        - Refactor particle pusher launcher #4471
        - Refactor differentiation trait #4473
        - Refactor GetChargeState and GetCharge #4474
        - Remove usage of `boost::result_of` #4505
    - PMacc:
        - Avoid overaligning particle frame data #3807
        - Remove `MappedBufferIntern` #4018
        - Use `if constexpr` in `lockstep::ForEach` #4095
        - Rework implementation of pmacc::Unique #4176
        - Remove HIP `atomicAdd(float)` emulation #4269
        - Split PMacc objects into hpp and cpp files #4260
        - DataBox remove method `reduceZ()` #4291
        - Refactor CountParticles and particle filters #4313
        - PMacc: refactor kernel en-queuing #4317
        - Use alpaka `mem_fence()` #4349
        - Introduce a constexpr value identifier #4344
        - Remove boost workaround #4373
        - Remove unused cuSTL classes #4415
        - Extend global reduce interface #4425
        - Remove cuSTL #4434
        - Cleanup PMacc's event system #4443
        - Remove buffer `*Intern` #4448
        - Topic lockstep for each refactoring #4460
        - PMacc: remove unused class math::Tuple #4461
        - PMacc: remove invokeIf #4467
        - PMacc: refactor meta::ForEach #4469
        - Refactor `TaskSetValue` #4470
        - Add HDINLINE to Selection methods #4500
        - PMacc: header include validation #3924
        - Remove dependency to `boost::math` #4503
        - PMacc: DataConnector change `get()` interface #4506
        - Refactor `math::Vector`, use unsigned dimension #4499
        - Remove `boost::filesystem usage` #4504
        - Remove particle merger from `TBG_macros.cfg` #4512
        - Rename `abs2` -> `l2norm2` and `abs` for  vectors to `l2norm` #4509
        - Replace mpl by mp11 (part 1.5/2) #4516
    - Plugins:
        - Add optionDefined() to plugins::multi::Option struct #3906
        - Read IO plugin configuration (openPMD plugin) from reading applications (TOML) #3720
        - openPMD-based plugins: choose default backend depending on what is available #4014
        - Remove cupla compatibility macro definition #4017
        - Refactor openPMD plugin #4054
        - Improve comments and naming in PNG output #4164
        - openPMD plugin: add IO file version #4287
        - openPMD plugin: Flush data to disk within a step #4002
        - openPMD plugin: JSON parameters for restarting #4329
        - Use cuplas/alpakas mapped memory #4348
        - Remove xray scattering #4352
        - Remove plugin: PositionsParticle #4371
        - Remove plugin: recource log #4372
        - Remove particle merging plugins #4388
        - Radiation plugin: Enable streaming #4387
        - Plugin radiation: use PMaccs forEachFrame #4377
        - Plugin emittance: remove cuSTL #4426
        - Code clean-up: avoid using new and plane pointers #4428
        - Charge conservation plugin: remove cuSTL #4424
        - Refactor particle filters #4432
        - Radiation plugin: refactor particle filter execution #4466
        - Radiation plugin: refactor getRadiationMask #4468
        - Plugin: CountMacroParticlesPerSupercell and ChargeConservation #4496
        - Plugin sum current refactoring #4497
        - Fix logging message in openPMD plugin #4044
    - Tools:
        - Update to cupla0.4.0 dev, cmake-modules and cuda_memtest #3868
        - cupla 0.4.0-dev + alpaka 0.8.0-dev #3915
        - Optimize CI #4437
        - Switch PICMI to incidentField #4459
        - Refactor `pic-create` #4515
    - Unroll interpolation #3859
    - Update to clang-format-12 #3826
    - PML: avoid `stack frames` in GPU kernel #3881
    - Version bump: openPMD-api from 0.12.0 to 0.14.3 #3913
    - Switch to C++17 #3949
    - Add combined derived attributes #3943
    - Remove pusher `Axel` #4115
    - Unify default openPMD file extension in plugins #4155
    - Improve comments and naming in PNG output in docs and examples #4168
    - Switch to std::void_t #4195
    - Change user interface of free temperature functor to match free density functor #4273
    - Change particle attribute for thermal boundaries to openPMD-conformant name #4278
    - PIConGPU use RPATH by default #4312
    - Remove FieldToParticleInterpolationNative #4369
    - Remove unused file `LinearInterpolationWithUpper` #4370
    - Switch to CI container version 3 #4378
    - Plugin: phase space remove cuSTL #4404
    - Clean Bunch example #4419
    - Example foilLCT: use incident field laser #4488
- Documentation:
    - Upgrade to Boost 1.66 #3939
    - Docs: Fix Stale LSF Links #3981
    - Expand documentation on running interactively #3967
    - Extend comments in density.param #4021
    - Add doxygen comments to declarations of density profile types #4024
    - Add a doc section on general usage of CPU systems #4039
    - Merge 0.6.0 changes back to the development branch #4079
    - Fix comments on axis definitions in incident field profiles #4083
    - Switch to alpaka 0.9.0 #4081
    - Fix a few instances of calling the Taflove-Hagness book just Taflove #4103
    - Fix some typos in the reference doc page #4117
    - Fix readthedocs example of context variable default initialization #4133
    - Fix incorrect doxygen includes in readthedocs #4132
    - Remove zlib from required dependencies in the docs #4128
    - Update documentation of dependencies #4136
    - Clarify description of numerical dispersion for Yee solver #4156
    - Remove field slice printer arguments from TBG_macros.cfg #4188
    - Documentation: mention `CMAKE_BUILD_TYPE` #4229
    - Fix copyright years ending in 2021 in some files #4242
    - Clarify docs on setting same sources for incident field on each side #4232
    - Clarify/fix comment for AOFDTD derivative functors #4246
    - Clarify approximation order of J in AOFDTD docs #4248
    - Add docs on substepping field solver #4259
    - Add Python code formatting guidelines to the docs #4277
    - Install dependencies for memoryPerDevice example #4327
    - Add papers Cristian Bontoiu #4445
- Fix warning: add explicit braces to avoid dangling else [-Wdangling-else] #3869
- Fix field absorber example that violated CFL for AOFDTD #3871
- Spack load: no more "-r" #3873
- CI: update to container version 2.0 #3934
- Add --ntasks-per-node and -np to .tpl files on Hemera #3940
- Adjust taurus k80 modules to work with c++17 #3962
- Don't install toml11 internally shipped library #3986
- Change the container URL of the GitLab CI to the new registry #4001
- Fix fieldBackground constructors to be HINLINE #4013
- Add ARM architecture templates #4033
- Add EZ current deposition type as alias to EmZ #4036
- Adjust laser in incident field example for a more realistic setup #3955
- System DICC: add a system description and a profile for the GPU queue #4015
- Remove example: ThermalTest #4100
- Add Thermal electron benchmark #4099
- CUDA: fix false warning #4113
- Bugfix JUWELS Booster profile #4151
- Add new benchmark TWEAC-FOM #4167
- Update to cupla with alpaka 1.0.0-dev and mallocMC #4166
- Extend TWEAC-FOM benchmark with incident field option for laser generation #4197
- Add componentwise incident field calculation option to TWEAC-FOM benchmark #4200
- Update cuda_memtest #4214
- Update mallocMC to fix possible out of memory access #4220
- CI: disable all optional dependencies for the Empty test case #4254
- Add and enable UCX_RC_TIMEOUT workaround #4288
- Add compilation of particle merger plugins to compile-time tests #4297
- Update to latest cupla dev branch #4309
- CI: add CUDA 11.3 - 11.4, HIP 4.5, 5.0 - 5.2 #4315
- Add new openPMD-api module for hemera kepler #4320
- Fix HIP CI tests #4353
- Remove PIConGPU install description using spack #4375
- KHI memory species buffer scaling based on simulation volume #4422
- Remove warm copper example #4433
- More variants: test KHI growth rate #4452
- Example bunch: use incident field laser #4449
- Remove old cluster profiles #4514
- Fix single particle test example #4482


0.6.0
-----

**Date:** 2021-12-21

C++14, New Solvers, I/O via openPMD API, HIP Support

This release switches to C++14 as minimum required version. Transition to C++17
is planned for upcoming releases.

We extended PIConGPU with a few new solvers. Binary collisions are now
available. We added arbitrary-order FDTD Maxwell's solver. All field solvers are
now compatible with perfectly matched layer absorber, which became default.
Yee solver now supports incident field generation using total field/scattered
field technique. We added Higuera-Cary particle pusher and improved
compatibility of pushers with probe species. Implementation of particle
boundaries was extended to support custom positions, reflecting and thermal
boundary kinds were added.

With this release, PIConGPU fully switches to openPMD API library for performing
I/O. The native HDF5 and ADIOS output plugins were replaced with a new openPMD
plugin. All other plugins were updated to use openPMD API. Plugins generally
support HDF5 and ADIOS2 backends of openPMD API, a user can choose file format
based on their installation of openPMD API. We also added new plugins for SAXS
and particle merging.

We added support for HIP as a computational backend. In particular, it allows
running on AMD GPUs. Several performance optimizations were added. Some functors
and plugins now have performance-influencing parameters exposed to a user.

The code was largely modernized and refactored, documentation was extended.

Thanks to Sergei Bastrakov, Kseniia Bastrakova, Brian Edward Marre,
Alexander Debus, Marco Garten, Bernhard Manfred Gruber, Axel Huebl,
Jakob Trojok, Jeffrey Kelling, Anton Lebedev, Felix Meyer, Paweł Ordyna,
Franz Poeschel, Lennert Sprenger, Klaus Steiniger, Manhui Wang,
Sebastian Starke, Maxence Thévenet, Richard Pausch, René Widera
for contributions to this release!

### Changes to "0.5.0"

**User Input Changes:**
 - Remove HDF5 I/O plugin (replaced with new openPMD plugin) #3361
 - Remove ADIOS I/O plugin (replaced with new openPMD plugin) #3691
 - Decouple field absorber selection from field solver #3635
 - Move all field absorber compile-time parameters to fieldAbsorber.param #3645
 - Change default field absorber to PML #3672
 - Change current interpolation from compile-time to command-line parameter #3552
 - Remove directional splitting field solver #3363
 - Remove compile-time movePoint value from grid.param #3793
 - Switch to cupla math functions #3245
 - Scale particle exchange buffer #3465
 - Update to new mallocMC version and parameters #3856

**New Features:**
 - PIC:
   - Add strategy parameter for current deposition algorithms #3221
   - Add Higuera-Cary particle pusher #3280 #3371
   - Add support for PML absorber with Lehe field solver #3301
   - Add support for Lehe field solver in 2D #3321
   - Add arbitrary-order FDTD field solver #3338
   - Add ionization current #3355
   - Add PML support to arbitrary-order FDTD field solver #3417
   - Faster TWTS laser implementation #3439
   - Parametrize FieldBackground for accuracy vs. memory #3527
   - Add a check for Debye length resolution #3446
   - Add user-defined iteration start pipeline
   - Add binary collisions #3416
   - Expose compute current worker multiplier  #3539
   - Add a new way to generate incident field using TF/SF #3592
   - Add support for device oversubscription #3632
   - Signal handling #3633
   - Replace FromHDF5Impl density with FromOpenPMDImpl #3655
   - Introduce filtered particleToGrid algorithm #3574
   - Record probe data with all pushers #3714
   - Compatibility check for laser and field solver #3734
   - Extend particles::Manipulate with area parameter #3747
   - Enable runtime kernel mapping #3750
   - Optimize particle pusher kernel #3775
   - Enable particle boundaries at custom positions #3763 #3776
   - Adjust physics log output #3825
   - Add a new particle manipulator combining total cell offset and RNG #3832
   - Add reflective particle boundary conditions #3806
   - Add thermal particle boundary conditions #3858
   - HIP support #3356 #3456 #3500
 - PMacc:
   - MPI direct support #3195
   - Improve output of kernel errors in PMacc with PMACC_BLOCKING_KERNEL=ON #3396
   - Optimize atomic functor for HIP #3457
   - Add device-side assert macro #3488
   - Add support for alpaka OpenMP target/OpenACC #3512
   - Additional debug info in guard check for supercell size #3267
   - Clarify error message and add comments in GameOfLife #3553
   - Print milliseconds in output #3606
   - Enable KernelShiftParticles to operate in guard area #3772
 - plugins:
   - Add a new openPMD plugin to replace the native HDF5 and ADIOS1 ones #2966
   - Add probabilistic particle merger plugin #3227
   - Add SAXS plugin using electron density #3134
   - Add work distribution parameter in radiation plugin #3354
   - Add field solver parameters attribute to output #3364
   - Update required openPMD version to 0.12.0 #3405
   - Use Streaming API in openPMD plugin #3485
   - Make radiation Amplitude class template #3519
   - Add compatibility to ISAAC GLM #3498
   - Extend JSON patterns in openPMD plugin #3513
   - Compatibility to new unified ISAAC naming scheme #3545
   - Use span-based storeChunk API in openPMD plugin #3609
   - Optimize radiation plugin math #3711
   - Deactivate support for writing to ADIOS1 via the openPMD plugin #3395
   - Avoid hardcoding of floating-point type in openPMD plugin #3759
   - Add checkpointing of internal RNG states #3758
   - Make the span-based storeChunk API opt-in in various openPMD-based IO routines #3933
 - tools:
   - Update to C++14 #3242
   - Add JetBrains project dir to .gitignore. #3265
   - Convert path to absolute inside tbg #3190
   - Add verbose output for openPMD-api #3281
   - Switch to openPMD API in `plot_chargeConservation_overTime.py` #3505
   - Switch to openPMD API in `plot_chargeConservation.py` #3504
   - Save the used cmakeFlags setup in output directory #3537
   - Add JUWELS Booster profile #3341
   - Check for doxygen style in CI #3629
   - Update Summit profile #3680
   - Remove old CI helper #3745
   - Remove uncrustify #3746
   - Add python env to gitignore #3805

**Bug Fixes:**
 - PIC:
   - Fix compilation with clang-cuda and Boost #3295
   - Modify FreeFormulaImpl density profile to evalute a functor in centers of cells #3415
   - Fix deprecation warnings #3467
   - Fix the scheme of updating convolutional B in PML by half time step #3475
   - Fix and modernize the generation scheme for MPI communication tags #3558
   - Fix TWTS implementations #3704
   - Fix domain adjuster for absorber in case there is a single domain along a direction #3760
   - Fix unreachable code warnings #3575
   - Fix incorrect assignment of Jz in 2d EmZ implementation #3893
   - Fix restarting with moving window #3902
   - Fix performance issue with HIP 4.3+ #3903
   - Fix using host function in device code #3911
 - PMacc:
   - Fix missing override for virtual functions #3315
   - Fix unsafe access to a vector in cuSTL MPI reduce #3332
   - Fix cuSTL CartBuffer dereferencing a null pointer when CUDA is enabled #3330
   - Remove usage of `atomicAddNoRet` for HIP 4.1 #3572
   - Fix compilation with icc #3628
   - Fix warning concerning used C++17 extension #3318
   - Fix unused variable warning #3803
 - plugins:
   - Fix checkpointing and output of PML fields for uneven domains #3276
   - Fix warnings in the openPMD plugin #3289
   - fix clang-cuda compile #3314
   - Fix shared memory size in the PhaseSpace plugin #3333
   - Fix observation direction precision to use float_X #3638
   - Fix crash in output after moving window stopped #3743
   - Fix openPMD warning in XrayScatteringWriter #3358
   - Fix warning due to missing virtual destructor #3499
   - Fix warning due to missing override #3855
   - Remove deprecated openPMD::AccessType and replace with openPMD::Access #3373
   - Fix processing of outgoing particles by multi plugins #3619
   - Fix getting unitSI for amplitude #3688
   - Fix outdated exception message for openPMD plugin with ADIOS1 #3730
   - Remove usused variable in ISAAC plugin #3756
   - Fix warning in radiation plugin #3771
   - Fix internal linkage of private JSON header in openPMD plugin #3863
   - Fix treatment of bool particle attributes in openPMD plugin #3890
   - Add missing communication in XrayScattering #3937
 - tools:
   - Fix plotting tool for numerical heating #3324
   - Fix typos in pic-create output #3435
   - No more "-r" in spack load #3873
   - Fix docker recipe #3921
 - Fix KHI for non-CUDA devices #3285
 - Fix clang10-cuda compile #3310
 - Fix segfault in thermal test #3517
 - Fix jump on uninitialized variable #3523
 - Fix EOF whitespace test #3555
 - Disable PMacc runtime tests for HIP in CI #3650

**Misc:**
 - refactoring:
   - PIC:
     - Use cupla instead of cuda prefix for function calls #3211
     - Abstract PML definitions from YeePML field solver #3283
     - Refactor implementations of derivatives and curls for fields #3309
     - Refactor Lehe solver to use the new derivative and curl functors #3317
     - Add pmacc functions to get basis unit vectors #3319
     - Refactor margin traits for field derivatives #3320
     - Add a new trait GetCellType of field solvers #3322
     - Remove outdated pmacc/nvidia/rng/* #3351
     - Add generic utilities to get absorber thickness #3348
     - Remove pmacc::memory::makeUnique #3353
     - Remove unused HasIonizersWithRNG and UsesRNG #3367
     - Cleanup particle shape structs #3376
     - Refactoring and cleanup of species.param #3431
     - Rename picongpu::MySimulation to picongpu::Simulation #3476
     - Avoid access to temporary variable #3508
     - Remove unused enum picongpu::FieldType #3556
     - Switch internal handling of current interpolation from compile-time to run-time #3551
     - Refactor and update comments of GetMargin #3564
     - Remove enum picongpu::CommunicationTag #3561
     - Add a version of GetTrait taking only field solver type as a parameter #3571
     - Remove interface `DataConnector::releaseData()` #3582
     - Remove unused directory pmacc/nvidia #3604
     - Rename picongpu::particles::CallFunctor to pmacc::functor::Call  #3608
     - Refactor common field absorber implementation #3611
     - Move PML implementation to a separate directory #3617
     - Move the basic implementation of FDTD field solver to the new eponymous template #3643
     - Use IUnary with filters in collisions and FieldTmp #3687
     - Refactor field update functor #3648
     - Replace leftover usage of boost/type_traits with std:: counterparts #3812
     - Relocate function to move particles #3802
     - Fix a typo in domain adjuster output #3851
     - Replace C-style stdlib includes with C++ counterparts #3852
     - Replace raw pointers used for memory management with smart pointers #3854
     - Refactor and document laser profiles #3798
     - Remove unused variable boundaryKind #3769
   - PMacc:
     - Fix pedantic warnings #3255
     - Refactor ConstVector #3274
     - Set default values for some PMacc game of life example arguments #3352
     - Refactor PMacc warp and atomic functions #3343
     - Change SharedMemAllocator function qualifiers from DEVICEONLY to DINLINE #3520
     - Delete unused file MultiGridBuffer.hpp #3565
     - Refactor lockstep::ForEach #3630
     - Refactor lockstep programming #3616
     - Change ExchangeTypeToRank from protected to public #3718
     - Clarify naming and comments for pmacc kernel mapping types #3765
     - Separate notification of plugins into a new function in SimulationHelper #3788
     - Add a factory and a factory function for pmacc::StrideMapping #3785
     - Replace custom compile-time string creation by BOOST_METAPARSE_STRING #3792
     - Refactor CachedBox #3813
     - Refactor and extend Vector #3817
     - Drop unused TwistedAxesNavigator #3821
     - Refactor DataBox #3820
     - Replace PitchedBox ctor with offset by DataBox.shift() #3828
     - Rename type to Reference in Cursor #3827
     - Fix a warning in MapperConcept #3777
     - Remove leftover mentions of ADIOS1 #3795
   - plugins:
     - Remove unused ParticlePatches class #3419
     - Switch to openPMD API in PhaseSpace plugin #3468
     - Switch to openPMD API in MacroParticleCounter plugin #3570
     - Switch to openPMD API in ParticleCalorimeter plugin #3560
     - Vector field vis vis compatility and benchmarking in ISAAC plugin #3719
     - Remove libSplash #3744
     - Remove unnecessary parameter in writeField function template of openPMD plugin #3767
     - Remove --openPMD.compression parameter #3764
     - Rename instance interface for multiplugins #3822
     - Use multidim access instead of two subscripts #3829
     - Use Iteration::close() in openPMD plugin #3408
   - tools:
     - Add cmake option to enforce dependencies #3586
     - Update to mallocMC 2.5.0crp-dev #3325
     - Clarify output concerning cuda_memtest not being available #3345
   - Reduce complexity of examples #3323
   - Avoid species dependency in field solver compile test #3430
   - Switch from Boost tests to Catch2 #3447
   - Introduce clang format #3440
   - Use ubuntu 20.04 with CUDA 11.2 in docker image #3502
   - Apply sorting of includes according to clang-format #3605
   - Remove leftover boost::shared_ptr #3800
   - Modernize according to clang-tidy #3801
   - Remove more old boost includes and replace with std:: counterparts #3819
   - Remove boost::result_of #3830
   - Remove macro __deleteArray()  #3839
 - documentation:
   - Fix a typo in the reference to [Pausch2018] #3214
   - Remove old HZDR systems #3246
   - Document C++14 requirement #3268
   - Add reference in LCT Example #3297
   - Fix minor issues with the sphinx rst formating #3290
   - Fix the name of probe fieldE and fieldB attribute in docs #3385
   - Improve clarity of comments concerning density calculation and profile interface #3414
   - Clarify comments for generating momentums based on temperature #3423
   - Replace travis badges with gitlab ones #3451
   - Update supported CUDA versions #3458
   - Juwels profile update #3359
   - Extend reminder to load environment in the docs with instructions for spack #3478
   - Fix typo in FieldAbsorberTest #3528
   - Improve documentation of clang-format usage #3522
   - Fix some outdated docs regarding output and checkpoint backends #3535
   - Update version to 0.6.0-dev #3542
   - Add installation instructions for ADIOS2 and HDF5 backends of openPMD API #3549
   - Add support for boost 1.74.0 #3583
   - Add brief user documentation for boundary conditions #3600
   - Update grid.param file doxygen string #3601
   - Add missing arbitrary order solver to the list of PML-enabled field solvers #3550
   - Update documentation of AOFDTD #3578
   - Update spack installation guide #3640
   - Add fieldAbsorber runtime parameter to TBG_macros.cfg #3646
   - Extend docs for the openPMD plugin regarding --openPMD.source #3667
   - Update documentation of memory calculator #3669
   - Summit template and documentation for asynchronous writing via ADIOS2 SST #3698
   - Add DGX (A100) profile on Cori #3694
   - Include clang-format in suggested contribution pipeline #3710
   - Extend TBG_macros and help string for changing field absorber #3736
   - Add link to conda picongpu-analysis-environment file to docs #3738
   - Add a brief doc page on ways of adding a laser to a simulation #3739
   - Update cupla to 0.3.0 release #3748
   - Update to malloc mc2.6.0crp-dev #3749
   - Mark openPMD backend libraries as optional dependencies in the docs #3752
   - Add a documentation page for developers that explains how to extend PIConGPU #3791
   - Extend doc section on particle filter workflows #3782
   - Add clarification on pluginUnload #3836
   - Add a readthedocs page on debugging #3848
   - Add a link to the domain definitions wiki in the particle global filter workflow #3849
   - add reference list #3273
   - Fix docker documentation #3544
   - Extend comments of CreateDensity #3837
   - Refactor and document laser profiles #3798
   - Mark CUDA 11.2 as supported version #3503
   - Document intention of Pointer #3831
   - Update ISAAC plugin documentation #3740
   - Add a doxygen warning to not remove gamma filter in transition radiation #3857
   - Extend user documentation with a warning about tools being Linux-only #3462
 - Update hemera modules and add openPMDapi module #3270
 - Separate project and account for Juwels profile #3369
 - Adjust tpl for juwels #3368
 - Delete superfluous `fieldSolver.param` in examples #3374
 - Build CI tests with all dependencies #3316
 - Update openPMD and ADIOS module #3393
 - Update summit profile to include openPMD #3384
 - Add both adios and adios2 module on hemera #3411
 - Use openPMD::getVersion() function #3427
 - Add SPEC benchmark example #3466
 - Update the FieldAbsorberTest example to match the Taflove book #3487
 - Merge mainline changes back into dev #3540
 - SPEC bechmark: add new configurations #3597
 - Profile for spock at ORNL #3627
 - CI: use container version 1.3 #3644
 - Update spock profile #3653
 - CI: use HIP 4.2 #3662
 - Compile for MI100 architecture only using spock in ORNL #3671
 - Fix compile warning using spock in ORNL #3670
 - Add radiation cases to SPEC benchmark #3683
 - Fix taurus-tud k80 profile #3729
 - Update spock profile after update to RHEL8 #3755
 - Allow interrupting job in CI #3761
 - Fix drift manipulator in SingleParticle example #3766
 - Compile on x86 runners in CI #3762
 - Update gitignore  for macOS system generated files #3779
 - Use stages for downstream pipe in CI #3773
 - Add Acceleration pusher to compile-time test suite #3844
 - Update mallocMC #3897


0.5.0
-----

**Date:** 2020-06-03

Perfectly Matched Layer (PML) and Bug Fixes

This release adds a new field absorber for the Yee solver, convolutional
perfectly matched layer (PML). Compared to the still supported exponential
dampling absorber, PML provides better absorption rate and much less spurious
reflections.

We added new plugins for computing emittance and transition radiation, particle
rendering with the ISAAC plugin, Python tools for reading and visualizing output
of a few plugins.

The release also adds a few quality-of-life features, including a new memory
calculator, better command-line experience with new options and bashcompletion,
improved error handling, cleanup of the example setups, and extensions to
documentation.

Thanks to Igor Andriyash, Sergei Bastrakov, Xeinia Bastrakova, Andrei Berceanu,
Finn-Ole Carstens, Alexander Debus, Jian Fuh Ong, Marco Garten, Axel Huebl,
Sophie Rudat (Koßagk), Anton Lebedev, Felix Meyer, Pawel Ordyna, Richard Pausch,
Franz Pöschel, Adam Simpson, Sebastian Starke, Klaus Steiniger, René Widera
for contributions to this release!

### Changes to "0.4.0"

**User Input Changes:**
 - Particle pusher acceleration #2731
 - stop moving window after N steps #2792
 - Remove unused ABSORBER_FADE_IN_STEPS from .param files in examples #2942
 - add namespace "radiation" around code related to radiation plugin #3004
 - Add a runtime parameter for window move point #3022
 - Ionization: add silicon to pre-defines #3078
 - Make dependency between boundElectrons and atomicNumbers more explicit #3076
 - openPMD: use particle id naming #3165
 - Docs: update `species.param` #2793 #2795

**New Features:**
 - PIC:
   - Particle pusher acceleration #2731
   - Stop moving window after N steps #2792
   - Auto domain adjustment #2840
   - Add a wrapper around main() to catch and report exceptions #2962
   - Absorber perfectly matched layer PML #2950 #2967
   - Make dependency between boundElectrons and atomicNumbers more explicit #3076
 - PMacc:
   - `ExchangeTypeNames` Verify Parameter for Access #2926
   - Name directions in species buffer warnings #2925
   - Add an implementation of exp for pmacc vectors #2956
   - SimulationFieldHelper: getter method to access cell description #2986
 - plugins:
   - PhaseSpaceData: allow multiple iterations #2754
   - Python MPL Visualizer: plot for several simulations #2762
   - Emittance Plugin #2588
   - DataReader: Emittance & PlotMPL: Emittance, SliceEmittance, EnergyWaterfall #2737
   - Isaac: updated for particle rendering #2940
   - Resource Monitor Plugin: Warnings #3013
   - Transition radiation plugin #3003
   - Add output and python module doc for radiation plugin #3052
   - Add reference to thesis for emittance plugin doc #3101
   - Plugins: ADIOS & PhaseSpace Wterminate #2817
   - Calorimeter Plugin: Document File Suffix #2800
   - Fix returning a stringstream by value #3251
 - tools:
   - Support alpaka accelerator `threads` #2701
   - Add getter for omega and n to python module #2776
   - Python Tools: Incorporate sim_time into readers and visualizers #2779
   - Add PIConGPU memory calculator #2806
   - Python visualizers as jupyter widgets #2691
   - pic-configure: add `--force/-f` option #2901
   - Correct target thickness in memory calculator #2873
   - CMake: Warning in 3.14+ Cache List #3008
   - Add an option to account for PML in the memory calculator #3029
   - Update profile hemera-hzdr: CMake version #3059
   - Travis CI: OSX sed Support #3073
   - CMake: mark cuda 10.2 as tested #3118
   - Avoid bash completion file path repetition #3136
   - Bashcompletion #3069
   - Jupyter widgets output capture #3149
   - Docs: Add ionization prediction plot #2870
   - pic-edit: clean cmake file cache if new param added #2904
   - CMake: Honor _ROOT Env Hints #2891
   - Slurm: Link stdout live #2839

**Bug Fixes:**
 - PIC:
   - fix EveryNthCellImpl #2768
   - Split `ParserGridDistribution` into `hpp/cpp` file #2899
   - Add missing inline qualifiers potentially causing multiple definitions #3006
   - fix wrong used method prefix #3114
   - fix wrong constructor call #3117
   - Fix calculation of omega_p for logging #3163
   - Fix laser bug in case focus position is at the init plane #2922
   - Fix binomial current interpolation #2838
   - Fix particle creation if density zero #2831
   - Avoid two slides #2774
   - Fix warning: comparison of unsigned integer #2987
 - PMacc:
   - Typo fix in Send/receive buffer warning #2924
   - Explicitly specify template argument for std::forward #2902
   - Fix signed int overflow in particle migration between supercells #2989
   - Boost 1.67.0+ Template Aliases #2908
   - Fix multiple definitions of PMacc identifiers and aliases #3036
   - Fix a compilation issue with ForEach lookup #2985
 - plugins:
   - Fix misspelled words in plugin documentation #2705
   - Fix particle merging #2753
   - OpenMPI: Use ROMIO for IO #2857
   - Radiation Plugin: fix bool conditions for hdf5 output #3021
   - CMake Modules: Update ADIOS FindModule #3116
   - ADIOS Particle Writer: Fix timeOffset #3120
   - openPMD: use particle id naming #3165
   - Include int16 and uint16 types as traits for ADIOS #2929
   - Fix observation direction of transition radiation plugin #3091
   - Fix doc transition radiation plugin #3089
   - Fix doc rad plugin units and factors #3113
   - Fix wrong underline in TransRad plugin doc #3102
   - Fix docs for radiation in 2D #2772
   - Fix radiation plugin misleading filename #3019
 - tools:
   - Update cuda_memtest: NVML Noise #2785
   - Dockerfile: No SSH Deamon & Keys, Fix Flex Build #2970
   - Fix hemera k80_restart.tpl #2938
   - Templates/profile for hemera k20 queue #2935
   - Splash2txt Build: Update deps #2914
   - splash2txt: fix file name trimming #2913
   - Fix compile splash2txt #2912
   - Docker CUDA Image: Hwloc Default #2906
   - Fix Python EnergyHistogramData: skip of first iteration #2799
 - Spack: Fix Compiler Docs #2997
 - Singularity: Workaround Chmod Issue, No UCX #3017
 - Fix examples particle filters #3065
 - Fix CUDA device selection #3084
 - Fix 8.cfg for Bremsstrahlung example #3097
 - Fix taurus profile #3152
 - Fix a typo in density ratio value of the KHI example #3162
 - Fix GCC constexpr lambda bug #3188
 - CFL Static Assert: new grid.param #2804
 - Fix missing exponent in fieldIonization.rst #2790
 - Spack: Improve Bootstrap #2773
 - Fix python requirements: remove sys and getopt #3172

**Misc:**
 - refactoring:
   - PIC:
     - Eliminate M_PI (again) #2833
     - Fix MappingDesc name hiding  #2835
     - More fixes for MSVC capturing constexpr in lambdas #2834
     - Core Particles: C++11 Using for Typedef #2859
     - Remove unused getCommTag() in FieldE, FieldB, FieldJ #2947
     - Add a using declaration for Difference type to yee::Curl #2955
     - Separate the code processing currents from MySimulation #2964
     - Add DataConnector::consume(), which shares and consumes the input #2951
     - Move picongpu/simulationControl to picongpu/simulation/control #2971
     - Separate the code processing particles from MySimulation #2974
     - Refactor cell types #2972
     - Rename `compileTime` into `meta` #2983
     - Move fields/FieldManipulator to fields/absorber/ExponentialDamping #2995
     - Add picongpu::particles::manipulate() as a high-level interface to particle manipulation #2993
     - `particles::forEach` #2991
     - Refactor and modernize implementation of fields #3005
     - Modernize ArgsParser::ArgsErrorCode #3023
     - Allow constructor for density free formular functor #3024
     - Reduce PML memory consumption #3122
     - Bremsstrahlung: use more constexpr #3176
     - Pass mapping description by value instead of pointer from simulation stages #3014
     - Add missing inline specifiers for functions defined in header files #3051
     - Remove ZigZag current deposition  #2837
     - Fix style issues with particlePusherAcceleration #2781
   - PMacc:
     - Supercell particle counter #2637
     - ForEachIdx::operator(): Use Universal Reference #2881
     - Remove duplicated definition of `BOOST_MPL_LIMIT_VECTOR_SIZE` #2883
     - Cleanup `pmacc/types.hpp` #2927
     - Add pmacc::memory::makeUnique similar to std::make_unique #2949
     - PMacc Vector: C++11 Using #2957
     - Remove pmacc::forward and pmacc::RefWrapper #2963
     - Add const getters to ParticleBox #2941
     - Remove unused pmacc::traits::GetEmptyDefaultConstructibleType #2976
     - Remove pmacc::traits::IsSameType which is no longer used #2979
     - Remove template parameter for initialization method of Pointer and FramePointer #2977
     - Remove pmacc::expressions which is no longer used #2978
     - Remove unused pmacc::IDataSorter #3030
     - Change PMACC_C_STRING to produce a static constexpr member #3050
     - Refactor internals of pmacc::traits::GetUniqueTypeId #3049
     - rename "counterParticles" to "numParticles" #3062
     - Make pmacc::DataSpace conversions explicit #3124
   - plugins:
     - Small update for python visualizers #2882
     - Add namespace "radiation" around code related to radiation plugin #3004
     - Remove unused includes of pthread #3040
     - SpeciesEligibleForSolver for radiation plugin #3061
     - ADIOS: Avoid unsafe temporary strings #2946
   - tools:
     - Update cuda_memtest: CMake CUDA_ROOT Env #2892
     - Update hemera tpl after SLURM update  #3123
   - Add pillow as dependency #3180
   - Params: remove `boost::vector<>` usage #2769
   - Use _X syntax in OnceIonized manipulator #2745
   - Add missing const to some GridController getters #3154
 - documentation:
   - Containers: Update 0.4.0 #2750
   - Merge 0.4.0 Changelog #2748
   - Update Readme & License: People #2749
   - Add .zenodo.json #2747
   - Fix species.param docu (in all examples too) #2795
   - Fix species.param example doc and grammar #2793
   - Further improve wording in docs #2710
   - MemoryCalculator: fix example output for documentation #2822
   - Manual: Plugin & Particle Sections, Map #2820
   - System: D.A.V.I.D.E #2821
   - License Header: Update 2019 #2845
   - Docs: Memory per Device Spelling #2868
   - CMake 3.11.0+ #2959
   - CUDA 9.0+, GCC 5.1+, Boost 1.65.1+ #2961
   - CMake: CUDA 9.0+ #2965
   - Docs: Update Sphinx #2969
   - CMake: CUDA 9.2-10.1, Boost <= 1.70.0 #2975
   - Badge: Commits Since Release & Good First #2980
   - Update info on maintainers in README.md #2984
   - Fix grammar in all `.profile.example` #2930
   - Docs: Dr.s #3009
   - Fix old file name in radiation doc #3018
   - System: ARIS #3039
   - fix typo in getNode and getDevice #3046
   - Window move point clean up #3045
   - Docs: Cori's KNL Nodes (NERSC) #3043
   - Fix various sphinx issues not related to doxygen #3056
   - Extend the particle merger plugin documentation #3057
   - Fix docs using outdated ManipulateDeriveSpecies #3068
   - Adjust cores per gpu on taurus after multicore update #3071
   - Docs: create conda env for building docs #3074
   - Docs: add missing checkpoint options #3080
   - Remove titan ornl setup and doc #3086
   - Summit: Profile & Templates #3007
   - Update URL to ADIOS #3099
   - License Header: Update 2020 #3138
   - Add PhD thesis reference in radiation plugin #3151
   - Spack: w/o Modules by Default #3182
   - Add a brief description of simulation output to basics #3183
   - Fix a typo in exchange communication tag status output #3141
   - Add a link to PoGit to the docs #3115
   - fix optional install instructions in the Summit profile #3094
   - Update the form factor documentation #3083
   - Docs: Add New References #3072
   - Add information about submit.cfg and submit.tpl files to docs. #3070
   - Fix style (underline length) in profile.rst #2936
   - Profiles: Section Title Length #2934
   - Contributor name typo in LICENSE.md #2880
   - Update modules and memory in gpu_picongpu.profile #2923
   - Add k80_picongpu.profile and k80.tpl #2919
   - Update taurus-tud profiles for the `ml` partition #2903
   - Hypnos: CMake 3.13.4 #2887
   - Docs: Install Blosc #2829
   - Docs: Source Intro Details #2828
   - Taurus Profile: Project #2819
   - Doc: Add System Links #2818
   - remove grep file redirect #2788
   - Correct jupyter widget example #3191
   - fix typo: `UNIT_LENGHT` to `UNIT_LENGTH` #3194
   - Change link to CRP group @ HZDR #2814
 - Examples: Unify .cfg #2826
 - Remove unused ABSORBER_FADE_IN_STEPS from .param files in examples #2942
 - Field absorber test example #2948
 - Singularity: Avoid Dotfiles in Home #2981
 - Boost: No std::auto_ptr #3012
 - Add YeePML to comments for field solver selection #3042
 - Add a runtime parameter for window move point #3022
 - Ionization: add silicon to pre-defines #3078
 - Add 1.cfg to Bremsstrahlung example #3098
 - Fix cmake flags for MSVS #3126
 - Fix missing override flags #3156
 - Fix warning #222-D: floating-point operation result is out of range #3170
 - Update alpaka to 0.4.0 and cupla to 0.2.0 #3175
 - Slurm update taurus: workdir to chdir #3181
 - Adjust profiles for taurus-tud #2990
 - Update mallocMC to 2.3.1crp #2893
 - Change imread import from scipy.misc to imageio #3192

0.4.3
-----

**Date:** 2019-02-14

System Updates and Bug Fixes

This release adds updates and new HPC system templates. Important bug
fixes include I/O work-arounds for issues in OpenMPI 2.0-4.0 (mainly
with HDF5), guards for particle creation with user-defined
profiles, a fixed binomial current smoothing, checks for the number
of devices in grid distributions and container (Docker & Singularity)
modernizations.

Thanks to Axel Huebl, Alexander Debus, Igor Andriyash, Marco Garten,
Sergei Bastrakov, Adam Simpson, Richard Pausch, Juncheng E,
Klaus Steiniger, and René Widera for contributions to this release!

### Changes to "0.4.2"

**Bug Fixes:**
- fix particle creation if density `<=` zero #2831
- fix binomial current interpolation #2838
- Docker & Singularity updates #2847
- OpenMPI: use ROMIO for IO #2841 #2857
- `--gridDist`: verify devices and blocks #2876
- Phase space plugin: unit of colorbar in 2D3V #2878

**Misc:**
- `ionizer.param`: fix typo in "Aluminium" #2865
- System Template Updates:
  - Add system links #2818
  - Taurus:
    - add project #2819
    - add Power9 V100 nodes #2856
  - add D.A.V.I.D.E (CINECA) #2821
  - add JURECA (JSC) #2869
  - add JUWELS (JSC) #2874
  - Hypnos (HZDR): CMake update #2887
  - Slurm systems: link `stdout` to `simOutput/output` #2839
- Docs:
  - Change link to CRP group @ HZDR #2814
  - `FreeRng.def`: typo in example usage #2825
  - More details on source builds #2828
  - Dependencies: Blosc install #2829
  - Ionization plot title linebreak #2867
- plugins:
  - ADIOS & phase space `-Wterminate` #2817
  - Radiation: update documented options #2842
- Update versions script: containers #2846
- pyflakes: `str`/`bytes`/`int` compares #2866
- Travis CI: Fix Spack CMake Install #2879
- Contributor name typo in `LICENSE.md` #2880
- Update mallocMC to 2.3.1crp #2893
- CMake: Honor `_ROOT` Env Hints #2891 #2892 #2893


0.4.2
-----
**Date:** 2018-11-19

CPU Plugin Performance

This release fixes a performance regression for energy histograms and phase
space plugins on CPU with our OpenMP backend on CPU. At least OpenMP 3.1 is
needed to benefit from this. Additionally, several small documentation issues
have been fixed and the energy histogram python tool forgot to return the first
iteration.

Thanks to Axel Huebl, René Widera, Sebastian Starke, and Marco Garten for
contributions to this release!

### Changes to "0.4.1"

**Bug Fixes:**
- Plugin performance regression:
  - Speed of plugins `EnergyHistogram` and `PhaseSpace` on CPU (`omp2b`) #2802
- Tools:
  - Python `EnergyHistogramData`: skip of first iteration #2799

**Misc:**
- update Alpaka to 0.3.5 to fix #2802
- Docs:
  - CFL Static Assert: new grid.param #2804
  - missing exponent in fieldIonization.rst #2790
  - remove grep file redirect #2788
  - Calorimeter Plugin: Document File Suffix #2800


0.4.1
-----
**Date:** 2018-11-06

Minor Bugs and Example Updates

This release fixes minor bugs found after the 0.4.0 release.
Some examples were slightly outdated in syntax, the new "probe particle"
`EveryNthCell` initialization functor was broken when not used with equal
spacing per dimension. In some rare cases, sliding could occur twice in
moving window simulations.

Thanks to Axel Huebl, René Widera, Richard Pausch and Andrei Berceanu for
contributions to this release!

### Changes to "0.4.0"

**Bug Fixes:**
- PIConGPU:
  - avoid sliding twice in some corner-cases #2774
  - EveryNthCell: broken if not used with same spacing #2768
  - broken compile with particle merging #2753
- Examples:
  - fix outdated derive species #2756
  - remove current deposition in bunch example #2758
  - fix 2D case of single electron init (via density) #2766
- Tools:
  - Python Regex: r Literals #2767
  - `cuda_memtest`: avoid noisy output if NVML is not found #2785

**Misc:**
- `.param` files: refactor `boost::vector<>` usage #2769
- Docs:
  - Spack: Improve Bootstrap #2773
  - Fix docs for radiation in 2D #2772
  - Containers: Update 0.4.0 #2750
  - Update Readme & License: People #2749
  - Add `.zenodo.json` #2747


0.4.0
-----
**Date:** 2018-10-19

CPU Support, Particle Filter, Probes & Merging

This release adds CPU support, making PIConGPU a many-core, single-source,
performance portable PIC code for all kinds of supercomputers.
We added particle filters to initialization routines and plugins, allowing
fine-grained in situ control of physical observables. All particle plugins
now support those filters and can be called multiple times with different
settings.

Particle probes and more particle initialization manipulators have been added.
A particle merging plugin has been added. The Thomas-Fermi model has been
improved, allowing to set empirical cut-offs. PIConGPU input and output
(plugins) received initial Python bindings for efficient control and analysis.

User input files have been dramatically simplified. For example, creating the
PIConGPU binary from input files for GPU or CPU is now as easy as
`pic-build -b cuda` or `pic-build -b omp2b` respectively.

Thanks to Axel Huebl, René Widera, Benjamin Worpitz, Sebastian Starke,
Marco Garten, Richard Pausch, Alexander Matthes, Sergei Bastrakov, Heiko Burau,
Alexander Debus, Ilja Göthel, Sophie Rudat, Jeffrey Kelling, Klaus Steiniger,
and Sebastian Hahn for contributing to this release!

### Changes to "0.3.0"

**User Input Changes:**
 - (re)move directory `simulation_defines/` #2331
 - add new param file `particleFilters.param` #2385
 - `components.param`: remove define `ENABLE_CURRENT` #2678
 - `laser.param`: refactor Laser Profiles to Functors #2587 #2652
 - `visualization.param`: renamed to `png.param` #2530
 - `speciesAttributes.param`: format #2087
 - `fieldSolver.param`: doxygen, refactored #2534 #2632
 - `mallocMC.param`: file doxygen #2594
 - `precision.param`: file doxygen #2593
 - `memory.param`:
   - `GUARD_SIZE` docs #2591
   - exchange buffer size per species #2290
   - guard size per dimension #2621
 - `density.param`:
   - Gaussian density #2214
   - Free density: fix `float_X` #2555
 - `ionizer.param`: fixed excess 5p shell entry in gold effective Z #2558
 - `seed.param`:
   - renamed to `random.param` #2605
   - expose random number method #2605
 - `isaac.param`: doxygen documentation #2260
 - `unit.param`:
   - doxygen documentation #2467
   - move conversion units #2457
   - earlier normalized speed of light in `physicalConstants.param` #2663
 - `float_X` constants to literals #2625
 - refactor particle manipulators #2125
 - new tools:
   - `pic-edit`: adjust `.param` files #2219
   - `pic-build`: combine pic-configure and make install #2204
 - `pic-configure`:
   - select CPU/GPU backend and architecture with `-b` #2243
   - default backend: CUDA #2248
 - `tbg`:
   - `.tpl` no `_profile` suffix #2244
   - refactor `.cfg` files: devices #2543
   - adjust LWFA setup for 8GPUs #2480
 - `SliceField` plugin: Option `.frequency` to `.period `#2034
 - particle filters:
   - add filter support to phase space plugin #2425
   - multi plugin energy histogram with filter #2424
   - add particle filter to `EnergyParticles` #2386
 - Default Inputs: C++11 `using` for `typedef` #2315
 - Examples: C++11 `using` for `typedef` #2314
 - Python: Parameter Ranges for Param Files (LWFA) #2289
 - `FieldTmp`: `SpeciesEligibleForSolver` Traits #2377
 - Particle Init Methods: Unify API & Docs #2442
 - get species by name #2464
 - remove template dimension from current interpolator's #2491
 - compile time string #2532

**New Features:**
 - PIC:
   - particle merging #1959
   - check cells needed for stencils #2257
   - exchange buffer size per species #2290
   - push with `currentStep` #2318
   - `InitController`: unphysical particles #2365
   - New Trait: `SpeciesEligibleForSolver` #2364
   - Add upper energy cut-off to ThomasFermi model #2330
   - Particle Pusher: Probe #2371
   - Add lower ion density cut-off to ThomasFermi model #2361
   - CT Factory: `GenerateSolversIfSpeciesEligible` #2380
   - add new param file `particleFilters.param` #2385
   - Probe Particle Usage #2384
   - Add lower electron temperature cut-off to ThomasFermi model #2376
   - new particle filters #2418 #2659 #2660 #2682
   - Derived Attribute: Bound Electron Density #2453
   - get species by name #2464
   - New Laser Profile: Exp. Ramps with Prepulse #2352
   - Manipulator: `UnboundElectronsTimesWeighting` #2398
   - Manipulator: `unary::FreeTotalCellOffset` #2498
   - expose random number method to the user #2605
   - seed generator for RNG #2607
   - FLYlite: initial interface & helper fields #2075
 - PMacc:
   - cupla compatible RNG #2226
   - generic `min()` and `max()` implementation #2173
   - Array: store elements without a default constructor #1973
   - add array to hold context variables #1978
   - add `ForEachIdx` #1977
   - add trait `GetNumWorker` #1985
   - add index pool #1958
   - Vector `float1_X` to `float_X` cast #2020
   - extend particle handle #2114
   - add worker config class #2116
   - add interfaces for functor and filter #2117
   - Add complex logarithm to math #2157
   - remove unused file `BitData.hpp` #2174
   - Add Bessel functions to math library #2156
   - Travis: Test PMacc Unit Tests #2207
   - rename CUDA index names in `ConcatListOfFrames` #2235
   - cuSTL `Foreach` with lockstep support #2233
   - Add complex `sin()` and `cos()` functions. #2298
   - Complex `BesselJ0` and `BesselJ1` functions #2161
   - CUDA9 default constructor warnings #2347
   - New Trait: HasIdentifiers #2363
   - RNG with reduced state #2410
   - PMacc RNG 64bit support #2451
   - PhaseSpace: add lockstep support #2454
   - signed and unsigned comparison #2509
   - add a workaround for MSVC bug with capturing `constexpr` #2522
   - compile time string #2532
   - `Vector`: add method `remove<...>()` #2602
   - add support for more cpu alpaka accelerators #2603 #2701
   - Vector `sumOfComponents` #2609
   - `math::CT::max` improvement #2612
 - plugins:
   - ADIOS: allow usage with accelerator `omp2b` #2236
   - ISAAC:
     - alpaka support #2268 #2349
     - require version 1.4.0+ #2630
   - `InSituVolumeRenderer`: removed (use ISAAC instead) #2238
   - HDF5: Allow Unphysical Particle Dump #2366
   - `SpeciesEligibleForSolver` Traits #2367
   - PNG:
     - lockstep kernel refactoring `Visualisation.hpp` #2225
     - require PNGwriter version 0.7.0+ #2468
   - `ParticleCalorimeter`:
     - add particle filter #2569
     - fix usage of uninitialized variable #2320
   - Python:
     - Energy Histogram Reader #2209 #2658
     - Phase Space Reader #2334 #2634 #2679
     - Move SliceField Module & add Python3 support #2354 #2718
     - Multi-Iteration Energy Histogram #2508
     - MPL Visualization modules #2484 #2728
     - migrated documentation to Sphinx manual #2172 #2726 #2738
     - shorter python imports for postprocessing tools #2727
     - fix energy histogram deprecation warning #2729
     - `data`: base class for readers #2730
     - `param_parser` for JSON parameter files #2719
 - tools:
   - Tool: New Version #2080
   - Changelog & Left-Overs from 0.3.0 #2120
   - TBG: Check Modified Input #2123
   - Hypnos (HZDR) templates:
     - `mpiexec` and `LD_LIBRARY_PATH` #2149
     - K20 restart #2627
     - restart `.tpl` files: new `checkpoints.period` syntax #2650
   - Travis: Enforce PEP8 #2145
   - New Tool: pic-build #2204
   - Docker:
     - `Dockerfile` introduced #2115 #2286
     - `spack clean` & `load` #2208
     - update ISAAC client URL #2565
   - add HZDR cluster `hydra` #2242
   - pic-configure: default backend CUDA #2248
   - New Tool: pic-edit #2219
   - FoilLCT: Plot Densities #2259
   - tbg: Add `-f` | `--force` #2266
   - Improved the cpuNumaStarter.sh script to support not using all hw threads #2269
   - Removed libm dependency for Intel compiler... #2278
   - CMake: Same Boost Min for Tools #2293
   - HZDR tpl: killall return #2295
   - PMacc: Set CPU Architecture #2296
   - ThermalTest: Flake Dispersion #2297
   - Python: Parameter Ranges for Param Files (LWFA) #2289
   - LWFA: GUI .cfg & Additional Parameters #2336
   - Move mpiInfo to new location #2355
   - bracket test for external libraries includes #2399
   - Clang-Tidy #2303
   - tbg -f: mkdir -p submitAction #2413
   - Fix initial setting of Parameter values #2422
   - Move TBG to bin/ #2537
   - Tools: Move pic-* to bin/ #2539
   - Simpler Python Parameter class #2550

**Bug Fixes:**
 - PIC:
   - fix restart with background fields enabled #2113
   - wrong border with current background field #2326
   - remove usage of pure `float` with `float_X` #2606
   - fix stencil conditions #2613
   - fix that guard size must be one #2614
   - fix dead code #2301
   - fix memory leaks #2669
 - PMacc:
   - event system:
     - fix illegal memory access #2151
     - fix possible deadlock in blocking MPI ops #2683
   - cuSTL:
     - missing `#include` in `ForEach` #2406
     - `HostBuffer` 1D Support #2657
   - fix warning concerning forward declarations of `pmacc::detail::Environment` #2489
   - `pmacc::math::Size_t<0>::create()` in Visual Studio #2513
   - fix V100 deadlock #2600
   - fix missing include #2608
   - fix gameOfLife #2700
   - Boost template aliases: fix older CUDA workaround #2706
 - plugins:
   - energy fields: fix reduce #2112
   - background fields: fix restart `GUARD` #2139
   - Phase Space:
     - fix weighted particles #2428
     - fix momentum meta information #2651
   - ADIOS:
     - fix 1 particle dumps #2437
     - fix zero size transform writes #2561
     - remove `adios_set_max_buffer_size` #2670
     - require 1.13.1+ #2583
   - IO fields as source #2461
   - ISAAC: fix gcc compile #2680
   - Calorimeter: Validate minEnergy #2512
 - tools:
   - fix possible linker error #2107
   - cmakeFlags: Escape Lists #2183
   - splash2txt: C++98 #2136
   - png2gas: C++98 #2162
   - tbg env variables escape \ and & #2262
   - XDMF Scripts: Fix Replacements & Offset #2309
   - pic-configure: cmakeFlags return code #2323
   - tbg: fix wrong quoting of `'` #2419
   - CMake in-source builds: too strict #2407
 - `--help` to stdout #2148
 - Density: Param Gaussian Density #2214
 - Fixed excess 5p shell entry in gold effective Z #2558
 - Hypnos: Zlib #2570
 - Limit Supported GCC with nvcc 8.0-9.1 #2628
 - Syntax Highlighting: Fix RTD Theme #2596
 - remove extra typename in documentation of manipulators #2044

**Misc:**
 - new example: Foil (LCT) TNSA #2008
 - adjust LWFA setup for 8 GPUs #2480
 - `picongpu --version` #2147
 - add internal Alpaka & cupla #2179 #2345
 - add alpaka dependency #2205 #2328 #2346 #2590 #2501 #2626 #2648 #2684 #2717
 - Update mallocMC to `2.3.0crp` #2350 #2629
 - cuda_memtest:
   - update #2356 #2724
   - usage on hypnos #2722
 - Examples:
   - remove unused loaders #2247
   - update `species.param` #2474
 - Bunch: no `precision.param` #2329
 - Travis:
   - stages #2341
   - static code analysis #2404
 - Visual Studio: ERROR macro defined in `wingdi.h` #2503
 - Compile Suite: update plugins #2595
 - refactoring:
   - PIC:
     - `const` POD Default Constructor #2300
     - `FieldE`: Fix Unreachable Code Warning #2332
     - Yee solver lockstep refactoring #2027
     - lockstep refactoring of `KernelComputeCurrent` #2025
     - `FieldJ` bash/insert lockstep refactoring #2054
     - lockstep refactoring of `KernelFillGridWithParticles` #2059
     - lockstep refactoring `KernelLaserE` #2056
     - lockstep refactoring of `KernelBinEnergyParticles` #2067
     - remove empty `init()` methods #2082
     - remove `ParticlesBuffer::createParticleBuffer()` #2081
     - remove init method in `FieldE` and `FieldB` #2088
     - move folder `fields/tasks` to libPMacc #2090
     - add `AddExchangeToBorder`, `CopyGuardToExchange` #2091
     - lockstep refactoring of `KernelDeriveParticles` #2097
     - lockstep refactoring of `ThreadCollective` #2101
     - lockstep refactoring of `KernelMoveAndMarkParticles` #2104
     - Esirkepov: reorder code order #2121
     - refactor particle manipulators #2125
     - Restructure Repository Structure  #2135
     - lockstep refactoring `KernelManipulateAllParticles` #2140
     - remove all lambda expressions. #2150
     - remove usage of native CUDA function prefix #2153
     - use `nvidia::atomicAdd` instead of our old wrapper #2152
     - lockstep refactoring `KernelAbsorbBorder` #2160
     - functor interface refactoring #2167
     - lockstep kernel refactoring `KernelAddCurrentToEMF` #2170
     - lockstep kernel refactoring `KernelComputeSupercells` #2171
     - lockstep kernel refactoring `CopySpecies` #2177
     - Marriage of PIConGPU and cupla/alpaka #2178
     - Ionization: make use of generalized particle creation #2189
     - use fast `atomicAllExch` in `KernelFillGridWithParticles` #2230
     - enable ionization for CPU backend #2234
     - ionization: speedup particle creation #2258
     - lockstep kernel refactoring `KernelCellwiseOperation` #2246
     - optimize particle shape implementation #2275
     - improve speed to calculate number of ppc #2274
     - refactor `picongpu::particles::startPosition` #2168
     - Particle Pusher: Clean-Up Interface #2359
     - create separate plugin for checkpointing #2362
     - Start Pos: OnePosition w/o Weighting #2378
     - rename filter: `IsHandleValid` -> `All` #2381
     - FieldTmp: `SpeciesEligibleForSolver` Traits #2377
     - use lower case begin for filter names #2389
     - refactor PMacc functor interface #2395
     - PIConGPU: C++11 `using` #2402
     - refactor particle manipulators/filter/startPosition #2408
     - rename `GuardHandlerCallPlugins` #2441
     - activate synchrotron for CPU back-end #2284
     - `DifferenceToLower/Upper` forward declaration #2478
     - Replace usage of M_PI in picongpu with Pi #2492
     - remove template dimension from current interpolator's #2491
     - Fix issues with name hiding in Particles #2506
     - refactor: field solvers #2534
     - optimize stride size for update `FieldJ` #2615
     - guard size per dimension #2621
     - Lasers: `float_X` Constants to Literals #2624
     - `float_X`: C++11 Literal #2622
     - log: per "device" instead of "GPU" #2662 #2677
     - earlier normalized speed of light #2663
     - fix GCC 7 fallthrough warning #2665 #2671
     - `png.unitless`: static asserts `clang` compatible #2676
     - remove define `ENABLE_CURRENT` #2678
   - PMacc:
     - refactor `ThreadCollective` #2021
     - refactor reduce #2015
     - lock step kernel `KernelShiftParticles` #2014
     - lockstep refactoring of `KernelCountParticles` #2061
     - lockstep refactoring `KernelFillGapsLastFrame` #2055
     - lockstep refactoring of `KernelFillGaps` #2083
     - lockstep refactoring of `KernelDeleteParticles` #2084
     - lockstep refactoring of `KernelInsertParticles` #2089
     - lockstep refactoring of `KernelBashParticles` #2086
     - call `KernelFillGaps*` from device #2098
     - lockstep refactoring of `KernelSetValue` #2099
     - Game of Life lockstep refactoring #2142
     - `HostDeviceBuffer` rename conflicting type defines #2154
     - use c++11 move semantic in cuSTL #2155
     - lockstep kernel refactoring `SplitIntoListOfFrames` #2163
     - lockstep kernel refactoring `Reduce` #2169
     - enable cuSTL CartBuffer on CPU #2271
     - allow update of a particle handle #2382
     - add support for particle filters #2397
     - RNG: Normal distribution #2415
     - RNG: use non generic place holder #2440
     - extended period syntax #2452
     - Fix buffer cursor dim #2488
     - Get rid of `<sys/time.h>` #2495
     - Add a workaround for `PMACC_STRUCT` to work in Visual Studio #2502
     - Fix type of index in OpenMP-parallelized loop #2505
     - add support for CUDA9 `__shfl_snyc, __ballot_sync` #2348
     - Partially replace compound literals in PMacc #2494
     - fix type cast in `pmacc::exec::KernelStarter::operator()` #2518
     - remove modulo in 1D to ND index transformation #2542
     - Add Missing Namespaces #2579
     - Tests: Add Missing Namespaces #2580
     - refactor RNG method interface #2604
     - eliminate `M_PI` from PMacc #2486
     - remove empty last frame #2649
     - no `throw` in destructors #2666
     - check minimum GCC & Clang versions #2675
   - plugins:
     - SliceField Plugin: Option .frequency to .period #2034
     - change notifyFrequency(s) to notifyPeriod #2039
     - lockstep refactoring `KernelEnergyParticles` #2164
     - remove `LiveViewPlugin`  #2237
     - Png Plugin: Boost to std Thread #2197
     - lockstep kernel refactoring `KernelRadiationParticles` #2240
     - generic multi plugin #2375
     - add particle filter to `EnergyParticles` #2386
     - PluginController: Eligible Species #2368
     - IO with filtered particles #2403
     - multi plugin energy histogram with filter #2424
     - lockstep kernel refactoring `ParticleCalorimeter` #2291
     - Splash: 1.7.0 #2520
     - multi plugin `ParticleCalorimeter` #2563
     - Radiation Plugin: Namespace #2576
     - Misc Plugins: Namespace #2578
     - EnergyHistogram: Remove Detector Filter #2465
     - ISAAC: unify the usage of period #2455
     - add filter support to phase space plugin #2425
     - Resource Plugin: `fix boost::core::swap` #2721
   - tools:
     - Python: Fix Scripts PEP8 #2028
     - Prepare for Python Modules #2058
     - pic-compile: fix internal typo #2186
     - Tools: All C++11 #2194
     - CMake: Use Imported Targets Zlib, Boost #2193
     - Python Tools: Move lib to / #2217
     - pic-configure: backend #2243
     - tbg: Fix existing-folder error message to stderr #2288
     - Docs: Fix Flake8 Errors #2340
     - Group parameters in LWFA example #2417
     - Python Tools (PS, Histo): Filter Aware #2431
     - Clearer conversion functions for Parameter values between UI scale and internal scale #2432
     - tbg:
       - add content of -o arg to env #2499
       - better handling of missing egetopt error message #2712
   - Format speciesAttributes.param #2087
   - Reduce # photons in Bremsstrahlung example #1979
   - TBG: .tpl no `_profile` suffix #2244
   - Default Inputs: C++11 Using for Typedef #2315
   - Examples: C++11 Using for Typedef #2314
   - LWFA Example: Restore a0=8.0 #2324
   - add support for CUDA9 `__shfl_snyc` #2333
   - add support for CUDA10 #2732
   - Update cuda_memtest: no cuBLAS #2401
   - Examples: Init of Particles per Cell #2412
   - Travis: Image Updates #2435
   - Particle Init Methods: Unify API & Docs #2442
   - PIConGPU use tiny RNG #2447
   - move conversion units to `unit.param` #2457
   - (Re)Move simulation_defines/ #2331
   - CMake: Project Vars & Fix Memtest #2538
   - Refactor .cfg files: devices #2543
   - Free Density: Fix float_X #2555
   - Boost: Format String Version #2566
   - Refactor Laser Profiles to Functors #2587
   - Params: float_X Constants to Literals #2625
 - documentation:
   - new subtitle #2734
   - Lockstep Programming Model #2026 #2064
   - `IdxConfig` append documentation #2022
   - `multiMask`: Refactor Documentation #2119
   - `CtxArray` #2390
   - Update openPMD Post-Processing #2322 #2733
   - Checkpoints Backends #2387
   - Plugins:
     - HDF5: fix links, lists & MPI hints #2313 #2711
     - typo in libSplash install #2735
     - External dependencies #2175
     - Multi & CPU #2423
     - Update PS & Energy Histo #2427
     - Memory Complexity #2434
   - Image Particle Calorimeter #2470
   - Update EnergyFields #2559
   - Note on Energy Reduce #2584
   - ADIOS: More Transport & Compression Doc #2640
   - ADIOS Metafile #2633
   - radiation parameters #1986
   - CPU Compile #2185
   - `pic-configure` help #2191
   - Python yt 3.4 #2273
   - Namespace `ComputeGridValuePerFrame` #2567
   - Document ionization param files for issue #1982 #1983
   - Remove ToDo from `ionizationEnergies.param` #1989
   - Parameter Order in Manual #1991
   - Sphinx:
     - Document Laser Cutoff #2000
     - Move Author Macros #2005
     - PDF Radiation  #2184
     - Changelog in Manual #2527
   - PBS usage example #2006
   - add missing linestyle to ionization plot for documentation #2032
   - fix unit ionization rate plot #2033
   - fix mathmode issue in ionization plot #2036
   - fix spelling of guard #2644
   - param: extended description #2041
   - fix typos found in param files and associated files #2047
   - Link New Coding Style #2074
   - Install: Rsync Missing #2079
   - Dev Version: 0.4.0-dev #2085
   - Fix typo in ADK documentation #2096
   - Profile Preparations #2095
   - SuperConfig: Header Fix #2108
   - Extended $SCRATCH Info #2093
   - Doxygen: Fix Headers #2118
   - Doxygen: How to Build HTML #2134
   - Badge: Docs #2144
   - CMake 3.7.0 #2181
   - Boost (1.62.0-) 1.65.1 - 1.68.0 #2182 #2707 #2713
   - Bash Subshells: `cmd` to $(cmd) #2187
   - Boost Transient Deps: date_time, chrono, atomic #2195
   - Install Docs: CUDA is optional #2199
   - Fix broken links #2200
   - PIConGPU Logo: More Platforms #2190
   - Repo Structure #2218
   - Document KNL GCC -march #2252
   - Streamline Install #2256
   - Added doxygen documentation for isaac.param file #2260
   - License Docs: Update #2282
   - Heiko to Former Members #2294
   - Added an example profile and tpl file for taurus' KNL #2270
   - Profile: Draco (MPCDF) #2308
   - $PIC_EXAMPLES #2327
   - Profiles for Titan & Taurus #2201
   - Taurus:
     - CUDA 8.0.61 #2337
     - Link KNL Profile #2339
     - SCS5 Update #2667
   - Move ParaView Profile #2353
   - Spack: Own GitHub Org #2358
   - LWFA Example: Improve Ranges #2360
   - fix spelling mistake in checkpoint #2372
   - Spack Install: Clarify #2373 #2720
   - Probe Pusher #2379
   - CI/Deps: CUDA 8.0 #2420
   - Piz Daint (CSCS):
     - Update Profiles #2306 #2655
     - ADIOS Build #2343
     - ADIOS 1.13.0 #2416
     - Update CMake #2436
     - Module Update #2536
     - avoid `pmi_alps` warnings #2581
   - Hypnos (HZDR): New Modules #2521 #2661
   - Hypnos: PNGwriter 0.6.0 #2166
   - Hypnos & Taurus: Profile Examples Per Queue #2249
   - Hemera: tbg templates #2723
   - Community Map #2445
   - License Header: Update 2018 #2448
   - Docker: Nvidia-Docker 2.0 #2462 #2557
   - Hide Double ToC #2463
   - Param Docs: Title Only #2466
   - New Developers #2487
   - Fix Docs: `FreeTotalCellOffset` Filter #2493
   - Stream-line Intro #2519
   - Fix HDF5 Release Link #2544
   - Minor Formatting #2553
   - PIC Model #2560
   - Doxygen: Publish As Well  #2575
   - Limit Filters to Eligible Species #2574
   - Doxygen: Less XML #2641
   - NVCC 8.0 GCC <= 5.3 && 9.0/9.1: GCC <= 5.5 #2639
   - typo: element-wise #2638
   - fieldSolver.param doxygen #2632
   - `memory.param`: `GUARD_SIZE` docs #2591
   - changelog script updated to python3 #2646
 - not yet supported on CPU (Alpaka): #2180
   - core:
     - Bremsstrahlung
   - plugins:
     - PositionsParticles
     - ChargeConservation
     - ParticleMerging
     - count per supercell (macro particles)
     - field intensity


0.3.2
-----
**Date:** 2018-02-16

Phase Space Momentum, ADIOS One-Particle Dumps & Field Names

This release fixes a bug in the phase space plugin which derived a too-low
momentum bin for particles below the typical weighting (and too-high for above
it). ADIOS dumps crashed on one-particle dumps and in the name of on-the-fly
particle-derived fields species name and field name were in the wrong order.
The plugins libSplash (1.6.0) and PNGwriter (0.6.0) need exact versions,
later releases will require a newer version of PIConGPU.

### Changes to "0.3.1"

**Bug Fixes:**
 - PIConGPU:
   - wrong border with current background field #2326
 - libPMacc:
   - cuSTL: missing include in `ForEach` #2406
   - warning concerning forward declarations of `pmacc::detail::Environment` #2489
   - `pmacc::math::Size_t<0>::create()` in Visual Studio #2513
 - plugins:
   - phase space plugin: weighted particles' momentum #2428
   - calorimeter: validate `minEnergy` #2512
   - ADIOS:
     - one-particle dumps #2437
     - `FieldTmp`: derived field name #2461
   - exact versions of libSplash 1.6.0 & PNGwriter 0.6.0
 - tools:
   - tbg: wrong quoting of `'` #2419
   - CMake: false-positive on in-source build check #2407
   - pic-configure: cmakeFlags return code #2323

**Misc:**
 - Hypnos (HZDR): new modules #2521 #2524

Thanks to Axel Huebl, René Widera, Sergei Bastrakov and Sebastian Hahn
for contributing to this release!


0.3.1
-----
**Date:** 2017-10-20

Field Energy Plugin, Gaussian Density Profile and Restarts

This release fixes the energy field plugin diagnostics and the "downramp"
parameter of the pre-defined Gaussian density profile. Restarts with enabled
background fields were fixed. Numerous improvements to our build system were
added to deal more gracefully with co-existing system-wide default libraries.
A stability issue due to an illegal memory access in the PMacc event system
was fixed.

### Changes to "0.3.0"

**.param file changes:**
 - `density.param`: in `Gaussian` profile, the parameter `gasSigmaRight`
   was not properly honored but `gasCenterRight` was taken instead  #2214
 - `fieldBackground.param`: remove micro meters usage in default file #2138

**Bug Fixes:**
 - PIConGPU:
   - `gasSigmaRight` of `Gaussian` density profile was broken since
     0.2.0 release #2214
   - restart with enabled background fields #2113 #2139
   - KHI example: missing constexpr in input #2309
 - libPMacc:
   - event system: illegal memory access #2151
 - plugins:
   - energy field reduce #2112
 - tools:
   - CMake:
     - Boost dependency:
       - same minimal version for tools #2293
       - transient dependenciens: `date_time`, `chrono`, `atomic` #2195
     - use targets of boost & zlib #2193 #2292
     - possible linker error #2107
   - XDMF script: positionOffset for openPMD #2309
   - cmakeFlags: escape lists #2183
   - tbg:
     - `--help` exit with 0 return code #2213
     - env variables: proper handling of \ and & #2262

**Misc:**
 - PIConGPU: `--help` to stdout #2148
 - tools: all to C++11 #2194
 - documentation:
   - Hypnos .tpl files: remove passing `LD_LIBRARY_PATH` to avoid warning #2149
   - fix plasma frequency and remove German comment #2110
   - remove micro meters usage in default background field #2138
   - README: update links of docs badge #2144

Thanks to Axel Huebl, Richard Pausch and René Widera for contributions to this
release!


0.3.0
-----
**Date:** 2017-06-16

C++11: Bremsstrahlung, EmZ, Thomas-Fermi, Improved Lasers

This is the first release of PIConGPU requiring C++11. We added a
newly developed current solver (EmZ), support for the generation of
Bremsstrahlung, Thomas-Fermi Ionization, Laguerre-modes in the
Gaussian-Beam laser, in-simulation plane for laser initialization,
new plugins for in situ visualization (ISAAC), a generalized particle
calorimeter and a GPU resource monitor. Initial support for clang
(host and device) has been added and our documentation has been
streamlined to use Sphinx from now on.

### Changes to "0.2.0"

**.param & .unitless file changes:**
 - use C++11 `constexpr` where possible and update arrays #1799 #1909
 - use C++11 `using` instead of `typedef`
 - removed `Config` suffix in file names #1965
 - `gasConfig` is now `density`
 - `speciesDefinition`:
   - simplified `Particles<>` interface #1711 #1942
   - `ionizer< ... >` became a sequence of `ionizers< ... >` #1999
 - `radiation`: replace `#defines` with clean C++ #1877 #1930 #1931 #1937

**Basic Usage:**

We renamed the default tools to create, setup and build a simulation.
Please make sure to update your `picongpu.profile` with the latest
syntax (e.g. new entries in `PATH`) and use from now on:
 - `$PICSRC/createParameterSet` -> `pic-create`
 - `$PICSRC/configure` -> `pic-configure`
 - `$PICSRC/compile` -> `pic-compile`

See the *Installation* and *Usage* chapters in our new documentation on
  https://picongpu.readthedocs.io
for detailed instructions.

**New Features:**
 - PIConGPU:
   - laser:
     - allow to define the initialization plane #1796
     - add transverse Laguerre-modes to standard Gaussian Beam #1580
   - ionization:
     - Thomas-Fermi impact ionization model #1754 #2003 #2007 #2037 #2046
     - Z_eff, energies, isotope: Ag, He, C, O, Al, Cu #1804 #1860
     - BSI models restructured #2013
     - multiple ionization algorithms can be applied per species,
       e.g. cut-off barrier suppression ionization (BSI),
       probabilistic field ionization (ADK) and collisional ionization #1999
   - Add EmZ current deposition solver #1582
   - FieldTmp:
     - Multiple slots #1703
     - Gather support to fill GUARD #2009
   - Particle `StartPosition`: `OnePosition` #1753
   - Add Bremsstrahlung #1504
   - Add kinetic energy algorithm #1744
   - Added species manipulators:
     - `CopyAttribute` #1861
     - `FreeRngImpl` #1866
   - Clang compatible static assert usage #1911
   - Use `PMACC_ASSERT` and `PMACC_VERIFY` #1662
 - PMacc:
   - Improve PMacc testsystem #1589
   - Add test for IdProvider #1590
   - Specialize HasFlag and GetFlagType for Particle #1604
   - Add generic atomicAdd #1606
   - Add tests for all RNG generators #1494
   - Extent function `twistVectorFieldAxes<>()` #1568
   - Expression validation/assertion #1578
   - Use PMacc assert and verify #1661
   - GetNComponents: improve error message #1670
   - Define `MakeSeq_t` #1708
   - Add `Array<>` with static size #1725
   - Add shared memory allocator #1726
   - Explicit cast `blockIdx` and `threadIdx` to `dim3` #1742
   - CMake: allow definition of multiple architectures #1729
   - Add trait `FilterByIdentifier` #1859
   - Add CompileTime Accessor: Type #1998
 - plugins:
   - HDF5/ADIOS:
     - MacroParticleCounter #1788
     - Restart: Allow disabling of moving window #1668
     - FieldTmp: MidCurrentDensityComponent #1561
   - Radiation:
     - Add pow compile time using c++11 #1653
     - Add radiation form factor for spherical Gaussian charge distribution #1641
   - Calorimeter: generalize (charged & uncharged) #1746
   - PNG: help message if dependency is not compiled #1702
   - Added:
     - In situ: ISAAC Plugin #1474 #1630
     - Resource log plugin #1457
 - tools:
   - Add a tpl file for k80 hypnos that automatically restarts #1567
   - Python3 compatibility for plotNumericalHeating #1747
   - Tpl: Variable Profile #1975
   - Plot heating & charge conservation: file export #1637
 - Support for clang as host && device compiler #1933

**Bug Fixes:**
 - PIConGPU:
   - 3D3V: missing absorber in z #2042
   - Add missing minus sign wavepacket laser transversal #1722
   - `RatioWeighting` (`DensityWeighting`) manipulator #1759
   - `MovingWindow`: `slide_point` now can be set to zero. #1783
   - `boundElectrons`: non-weighted attribute #1808
   - Verify number of ionization energy levels == proton number #1809
   - Ionization:
     - charge of ionized ions #1844
     - ADK: fix effective principal quantum number `nEff` #2011
   - Particle manipulators: position offset #1852
 - PMacc:
   - Avoid CUDA local memory usage of `Particle<>` #1579
   - Event system deadlock on `MPI_Barrier` #1659
   - ICC: `AllCombinations` #1646
   - Device selection: guard valid range #1665
   - `MapTuple`: broken compile with icc #1648
   - Missing '%%' to use ptx special register #1737
   - `ConstVector`: check arguments init full length #1803
   - `CudaEvent`: cyclic include #1836
   - Add missing `HDINLINE` #1825
   - Remove `BOOST_BIND_NO_PLACEHOLDERS` #1849
   - Remove CUDA native static shared memory #1929
 - plugins:
   - Write openPMD meta data without species #1718
   - openPMD: iterationFormat only Basename #1751
   - ADIOS trait for `bool` #1756
   - Adjust `radAmplitude` python module after openPMD changes #1885
   - HDF5/ADIOS: ill-placed helper `#include` #1846
   - `#include`: never inside namespace #1835
 - work-around for bug in boost 1.64.0 (odeint) + CUDA NVCC 7.5 & 8.0 #2053 #2076

**Misc:**
 - refactoring:
   - PIConGPU:
     - Switch to C++11 only #1649
     - Begin kernel names with upper case letter #1691
     - Maxwell solver, use curl instance #1714
     - Lehe solver: optimize performance #1715
     - Simplify species definition #1711
     - Add missing `math::` namespace to `tan()` #1740
     - Remove usage of pmacc and boost auto #1743
     - Add missing `typename`s #1741
     - Change ternary if operator to `if` condition #1748
     - Remove usage of `BOOST_AUTO` and `PMACC_AUTO` #1749
     - mallocMC: organize setting #1779
     - `ParticlesBase` allocate member memory #1791
     - `Particle` constructor interface  #1792
     - Species can omit a current solver #1794
     - Use constexpr for arrays in  `gridConfig.param` #1799
     - Update mallocMC #1798
     - `DataConnector`: `#includes` #1800
     - Improve Esirkepov speed #1797
     - Ionization Methods: Const-Ness #1824
     - Missing/wrong includes #1858
     - Move functor `Manipulate` to separate file #1863
     - Manipulator `FreeImpl` #1815
     - Ionization: clean up params #1855
     - MySimulation: remove particleStorage #1881
     - New `DataConnector` for fields (& species) #1887 #2045
     - Radiation filter functor: remove macros #1877
     - Topic use remove shared keyword #1727
     - Remove define `ENABLE_RADIATION` #1931
     - Optimize `AssignedTrilinearInterpolation` #1936
     - `Particles<>` interface  #1942
     - Param/Unitless files: remove "config" suffix #1965
     - Kernels: Refactor Functions to Functors #1669
     - Gamma calculation #1857
     - Include order in defaut loader #1864
     - Remove `ENABLE_ELECTRONS/IONS` #1935
     - Add `Line<>` default constructor #1588
   - PMacc:
     - Particles exchange: avoid message spamming #1581
     - Change minimum CMake version #1591
     - CMake: handle PMacc as separate library #1692
     - ForEach: remove boost preprocessor #1719
     - Refactor `InheritLinearly` #1647
     - Add missing `HDINLINE` prefix #1739
     - Refactor .h files to .hpp files #1785
     - Log: make events own level #1812
     - float to int cast warnings #1819
     - DataSpaceOperations: Simplify Formula #1805
     - DataConnector: Shared Pointer Storage #1801
     - Refactor `MPIReduce` #1888
     - Environment refactoring #1890
     - Refactor `MallocMCBuffer` share #1964
     - Rename `typedef`s inside `ParticleBuffer` #1577
     - Add typedefs for `Host`/`DeviceBuffer` #1595
     - DeviceBufferIntern: fix shadowed member variable #2051
   - plugins:
     - Source files: remove non-ASCII chars #1684
     - replace old analyzer naming #1924
     - Radiation:
       - Remove Nyquist limit switch #1930
       - Remove precompiler flag for form factor #1937
     - compile-time warning in 2D live plugin #2063
   - tools:
     - Automatically restart from ADIOS output #1882
     - Workflow: rename tools to set up a sim #1971
     - Check if binary `cuda_memtest` exists #1897
   - C++11 constexpr: remove boost macros #1655
   - Cleanup: remove EOL white spaces #1682
   - .cfg files: remove EOL white spaces #1690
   - Style: more EOL #1695
   - Test: remove more EOL white spaces #1685
   - Style: replace all tabs with spaces #1698
   - Pre-compiler spaces #1693
   - Param: Type List Syntax #1709
   - Refactor Density Profiles #1762
   - Bunch Example: Add Single e- Setup #1755
   - Use Travis `TRAVIS_PULL_REQUEST_SLUG` #1773
   - ManipulateDeriveSpecies: Refactor Functors & Tests #1761
   - Source Files: Move to Headers #1781
   - Single Particle Tests: Use Standard MySimulation #1716
   - Replace NULL with C++11 nullptr #1790
 - documentation:
   - Wrong comment random->quiet #1633
   - Remove `sm_20` Comments #1664
   - Empty Example & `TBG_macros.cfg` #1724
   - License Header: Update 2017 #1733
   - speciesInitialization: remove extra typename in doc #2044
   - INSTALL.md:
     - List Spack Packages #1764
     - Update Hypnos Example #1807
     - grammar error #1941
   - TBG: Outdated Header #1806
   - Wrong sign of `delta_angle` in radiation observer direction #1811
   - Hypnos: Use CMake 3.7 #1823
   - Piz Daint: Update example environment #2030
   - Doxygen:
     - Warnings Radiation #1840
     - Warnings Ionization #1839
     - Warnings PMacc #1838
     - Warnings Core #1837
     - Floating Docstrings #1856
     - Update `struct.hpp` #1879
     - Update FieldTmp Operations #1789
     - File Comments in Ionization #1842
     - Copyright Header is no Doxygen #1841
   - Sphinx:
     - Introduce Sphinx + Breathe + Doxygen #1843
     - PDF, Link rst/md, png #1944 #1948
     - Examples #1851 #1870 #1878
     - Models, PostProcessing #1921 #1923
     - PMacc Kernel Start #1920
     - Local Build Instructions #1922
     - Python Tutorials #1872
     - Core Param Files #1869
     - Important Classes #1871
     - .md files, tbg, profiles #1883
     - `ForEach` & Identifier #1889
     - References & Citation #1895
     - Slurm #1896 #1952
     - Restructure Install Instructions #1943
     - Start a User Workflows Section #1955
   - ReadTheDocs:
     - Build PDF & EPUB #1947
     - remove linenumbers #1974
   - Changelog & Version 0.2.3 (master) #1847
   - Comments and definition of `radiationObserver` default setup #1829
   - Typos plot radiation tool #1853
   - doc/ -> docs/ #1862
   - Particles Init & Manipulators #1880
   - INSTALL: Remove gimli #1884
   - BibTex: Change ShortHand #1902
   - Rename `slide_point` to `movePoint` #1917
   - Shared memory allocator documenation #1928
   - Add documentation on slurm job control #1945
   - Typos, modules #1949
   - Mention current solver `EmZ` and compile tests #1966
 - Remove assert.hpp in radiation plugin #1667
 - Checker script for `__global__` keyword #1672
 - Compile suite: GCC 4.9.4 chain #1689
 - Add TSC and PCS rad form factor shapes #1671
 - Add amend option for tee in k80 autorestart tpl #1681
 - Test: EOL and suggest solution #1696
 - Test: check & remove pre-compiler spaces #1694
 - Test: check & remove tabs #1697
 - Travis: check PR destination #1732
 - Travis: simple style checks #1675
 - PositionFilter: remove (virtual) Destructor #1778
 - Remove namespace workaround #1640
 - Add Bremsstrahlung example #1818
 - WarmCopper example: FLYlite benchmark #1821
 - Add compile tests for radiation methods #1932
 - Add visual studio code files to gitignore #1946
 - Remove old QT in situ volume visualization #1735

Thanks to Axel Huebl, René Widera, Alexander Matthes, Richard Pausch,
Alexander Grund, Heiko Burau, Marco Garten, Alexander Debus,
Erik Zenker, Bifeng Lei and Klaus Steiniger for contributions to this
release!


0.2.5
-----
**Date:** 2017-05-27

Absorber in z in 3D3V, effective charge in ADK ionization

The absorbing boundary conditions for fields in 3D3V simulations were
not enabled in z direction. This caused unintended reflections of
electro-magnetic fields in z since the 0.1.0 (beta) release.
ADK ionization was fixed to the correct charge state (principal
quantum number) which caused wrong ionization rates for all
elements but Hydrogen.

### Changes to "0.2.5"

**Bug Fixes:**
 - ADK ionization: effective principal quantum number nEff #2011
 - 3D3V: missing absorber in z #2042

**Misc:**
 - compile-time warning in 2D live plugin #2063
 - DeviceBufferIntern: fix shadowed member variable #2051
 - speciesInitialization: remove extra typename in doc #2044

Thanks to Marco Garten, Richard Pausch, René Widera and Axel Huebl
for spotting the issues and providing fixes!


0.2.4
-----
**Date:** 2017-03-06

Charge of Bound Electrons, openPMD Axis Range, Manipulate by Position

This release fixes a severe bug overestimating the charge of ions
when used with the `boundElectrons` attribute for field ionization.
For HDF5 & ADIOS output, the openPMD axis annotation for fields in
simulations with non-cubic cells or moving window was interchanged.
Assigning particle manipulators within a position selection was
rounded to the closest supercell (`IfRelativeGlobalPositionImpl`).

### Changes to "0.2.3"

**Bug Fixes:**
 - ionization: charge of ions with `boundElectrons` attribute #1844
 - particle manipulators: position offset, e.g. in
   `IfRelativeGlobalPositionImpl` rounded to supercell #1852 #1910
 - PMacc:
   - remove `BOOST_BIND_NO_PLACEHOLDERS` #1849
   - add missing `HDINLINE` #1825
   - `CudaEvent`: cyclic include #1836
 - plugins:
   - std includes: never inside namespaces #1835
   - HDF5/ADIOS openPMD:
     - GridSpacing, GlobalOffset #1900
     - ill-places helper includes #1846

Thanks to Axel Huebl, René Widera, Thomas Kluge, Richard Pausch and
Rémi Lehe for spotting the issues and providing fixes!


0.2.3
-----
**Date:** 2017-02-14

Energy Density, Ionization NaNs and openPMD

This release fixes energy density output, minor openPMD issues,
corrects a broken species manipulator to derive density weighted
particle distributions, fixes a rounding issue in ionization
routines that can cause simulation corruption for very small
particle weightings and allows the moving window to start
immediately with timestep zero. For ionization input, we now
verify that the number of arguments in the input table matches
the ion species' proton number.

### Changes to "0.2.2"

**Bug Fixes:**
 - openPMD:
   - iterationFormat only basename #1751
   - ADIOS trait for bool #1756
   - boundElectrons: non-weighted attribute #1808
 - RatioWeighting (DensityWeighting) manipulator #1759
 - MovingWindow: slide_point now can be set to zero #1783
 - energy density #1750 #1744 (partial)
 - possible NAN momenta in ionization #1817
 - `tbg` bash templates were outdated/broken #1831

**Misc:**
 - ConstVector:
   - check arguments init full length #1803
   - float to int cast warnings #1819
 - verify number of ionization energy levels == proton number #1809

Thanks to Axel Huebl, René Widera, Richard Pausch, Alexander Debus,
Marco Garten, Heiko Burau and Thomas Kluge for spotting the issues
and providing fixes!


0.2.2
-----
**Date:** 2017-01-04

Laser wavepacket, vacuum openPMD & icc

This release fixes a broken laser profile (wavepacket), allows to use
icc as the host compiler, fixes a bug when writing openPMD files in
simulations without particle species ("vacuum") and a problem with
GPU device selection on shared node usage via `CUDA_VISIBLE_DEVICES`.

### Changes to "0.2.1"

**Bug Fixes:**
 - add missing minus sign wavepacket laser transversal #1722
 - write openPMD meta data without species #1718
 - device selection: guard valid range #1665
 - PMacc icc compatibility:
   - `MapTuple` #1648
   - `AllCombinations` #1646

**Misc:**
 - refactor `InheritLinearly` #1647

Thanks to René Widera and Richard Pausch for spotting the issues and
providing fixes!


0.2.1
-----
**Date:** 2016-11-29

QED synchrotron photon & fix potential deadlock in checkpoints

This releases fixes a potential deadlock encountered during checkpoints and
initialization. Furthermore, we forgot to highlight that the 0.2.0 release
also included a QED synchrotron emission scheme (based on the review in
A. Gonoskov et al., PRE 92, 2015).

### Changes to "0.2.0"

**Bug Fixes:**
 - potential event system deadlock init/checkpoints #1659

Thank you to René Widera for spotting & fixing and Heiko Burau for the QED
synchrotron photon emission implementation!


0.2.0 "Beta"
------------
**Date:** 2016-11-24

Beta release: full multiple species support & openPMD

This release of PIConGPU, providing "beta" status for users, implements full
multi-species support for an arbitrary number of particle species and refactors
our main I/O to be formatted as openPMD (see http://openPMD.org).
Several major features have been implemented and stabilized,
highlights include refactored ADIOS support (including checkpoints), a
classical radiation reaction pusher (based on the work of M. Vranic/IST),
parallel particle-IDs, generalized on-the-fly particle creation, advanced field
ionization schemes and unification of plugin and file names.

This is our last C++98 compatible release (for CUDA 5.5-7.0). Upcoming releases
will be C++11 only (CUDA 7.5+), which is already supported in this release,
too.

Thank you to Axel Huebl, René Widera, Alexander Grund, Richard Pausch,
Heiko Burau, Alexander Debus, Marco Garten, Benjamin Worpitz, Erik Zenker,
Frank Winkler, Carlchristian Eckert, Stefan Tietze, Benjamin Schneider,
Maximilian Knespel and Michael Bussmann for contributions to this release!

### Changes to "0.1.0"

Input file changes: the generalized versions of input files are as always in
`src/picongpu/include/simulation_defines/`.

**.param file changes:**
 - all `const` parameters are now `BOOST_CONSTEXPR_OR_CONST`
 - add pusher with radiation reaction (Reduced Landau Lifshitz) #1216
 - add manipulator for setting `boundElectrons<>` attribute #768
 - add `PMACC_CONST_VECTOR` for ionization energies #768 #1022
 - `ionizationEnergies.param` #865
 - `speciesAttributes.param`: add ionization model `ADK` (Ammosov-Delone-Krainov) for lin. pol. and circ. pol cases #922 #1541
 - `speciesAttributes.param`: rename `BSI` to `BSIHydrogenLike`, add `BSIStarkShifted` and `BSIEffectiveZ` #1423
 - `laserConfig.param`: documentation fixed and clearified #1043 #1232 #1312 #1477
 - `speciesAttributes.param`: new required traits for for each attribute #1483
 - `species*.param`: refactor species mass/charge definition (relatve to base mass/charge) #948
 - `seed.param`: added for random number generator seeds #951
 - remove use of native `double` and `float` #984 #991
 - `speciesConstants.param`: move magic gamma cutoff value from radition plugin here #713
 - remove invalid `typename` #926 #944

**.unitless file changes:**
 - add pusher with radiation reaction (Reduced Landau Lifshitz) #1216
 - pusher traits simplified #1515
 - fieldSolver: numericalCellType is now a namespace not a class #1319
 - remove usage of native `double` and `float` #983 #991
 - remove invalid `typename` #926
 - add new param file: `synchrotronPhotons.param` #1354
 - improve the CFL condition depending on dimension in KHI example #774
 - add laserPolynom as option to `componentsConfig.param` #772

**tbg: template syntax**

Please be aware that templates (`.tpl`) used by `tbg` for job submission
changed slightly. Simply use the new system-wise templates from
`src/picongpu/submit/`. #695 #1609 #1618

Due to unifications in our command line options (plugins) and multi-species
support, please update your `.cfg` files with the new namings. Please visit
`doc/TBG_macros.cfg` and our wiki for examples.

**New Features:**
 - description of 2D3V simulations is now scaled to a user-defined "dZ" depth
   looking like a one-z-cell 3D simulation #249 #1569 #1601
 - current interpolation/smoothing added #888
 - add synchrotron radiation of photons from QED- and classical spectrum #1354 #1299 #1398
 - species attributes:
   - particle ids for tracking #1410
   - self-describing units and dimensionality #1261
   - add trait `GetDensityRatio`, add attribute `densityRatio`
   - current solver is now a optinal for a species #1228
   - interpolation is now a optional attribute for a species #1229
   - particle pusher is now a optional attribute for a species #1226
   - add species shape piecewise biqudratic spline `P4S` #781
 - species initialization:
   - add general particle creation module #1353
   - new manipulators to clone electrons from ions #1018
   - add manipulator to change the in cell position after gas creation #947 #959
   - documentation #961
 - species pushers:
   - enable the way for substepping particle pushers as RLL
     - add pusher with radiation reaction (Reduced Landau Lifshitz) #1216
     - enable substepping in pushers #1201 #1215  #1339 #1210 #1202 #1221
     - add Runge Kutta solver #1177
     - enable use of macro-particle weighting in pushers #1213
   - support 2D for all pushers #1126
 - refactor gas profile definitions #730 #980 #1265
 - extend `FieldToParticleInterpolation` to 1D- and 2D-valued fields #1452
 - command line options:
   - parameter validation #863
   - support for `--softRestarts <n>` to loop simulations #1305
   - a simulation `--author` can be specified (I/O, etc.) #1296 #1297
   - calling `./picongpu` without arguments triggers `--help` #1294
 - FieldTmp:
   - scalar fields renamed #1259 #1387 #1523
   - momentum over component #1481
 - new traits:
   - `GetStringProperties` for all solvers and species flags #1514 #1519
   - `MacroWeighted` and `WeightingPower` #1445
 - speedup current deposition solver ZigZag #927
 - speedup particle operations with collective atomics #1016
 - refactor particle update call #1377
 - enable 2D for single particle test #1203
 - laser implementations:
   - add phase to all laser implementations #708
   - add in-plane polarization to TWTS laser #852
   - refactor specific float use in laser polynom #782
   - refactored TWTS laser #704
 - checkpoints: now self-test if any errors occured before them #897
 - plugins:
   - add 2D support for SliceFieldPrinter plugin #845
   - notify plugins on particles leaving simulation #1394
   - png: threaded, less memory hungry in 2D3V, with author information #995 #1076 #1086 #1251 #1281 #1292 #1298 #1311 #1464 #1465
   - openPMD support in I/O
     - HDF5 and ADIOS plugin refactored #1427 #1428 #1430 #1478 #1517 #1520 #1522 #1529
     - more helpers added #1321 #1323 #1518
     - both write now in a sub-directory in simOutput: h5/ and bp/ #1530
     - getUnit and getUnitDimension in all fields & attributes #1429
   - ADIOS:
     - prepare particles on host side befor dumping #907
     - speedup with `OpenMP` #908
     - options to control striping & meta file creation #1062
     - update to 1.10.0+ #1063 #1557
     - checkpoints & restarts implemented #679 #828 #900
   - speedup radioation #996
   - add charge conservation plugin #790
   - add calorimeter plugin #1376
   - radiation:
     - ease restart on command line #866
     - output is now openPMD compatible #737 #1053
     - enable compression for hdf5 output #803
     - refactor specific float use #778
     - refactor radiation window function for 2D/3D #799
 - tools:
   - add error when trying to compile picongpu with CUDA 7.5 w/o C++11 #1384
   - add tool to load hdf5 radiation data into python #1332
   - add uncrustify tool (format the code) #767
   - live visualisation client: set fps panal always visible #1240
   - tbg:
     - simplify usage of `-p|--project` #1267
     - transfers UNIX-permisions from `*.tpl` to submit.start #1140
   - new charge conservation tools #1102, #1118, #1132, #1178
   - improve heating tool to support unfinished and single simulations #729
   - support for python3 #1134
   - improve graphics of numerical heating tool #742
   - speed up sliceFieldReader.py #1399
 - ionization models:
   - add possibility for starting simulation with neutral atoms #768
   - generalize BSI: rename BSI to BSIHydrogenLike, add BSIStarkShifted and BSIEffectiveZ #1423
   - add ADK (Ammosov-Delone-Krainov) for lin. pol. and circ. pol cases #922 #1490 #1541 #1542
   - add Keldysh #1543
   - make use of faster RNG for Monte-Carlo with ionization #1542 #1543
 - support radiation + ionization in LWFA example #868
 - PMacc:
   - running with synchronized (blocking) kernels now adds more useful output #725
   - add RNGProvider for persistent PRNG states #1236, #1493
   - add `MRG32k3a` RNG generator #1487
   - move readCheckpointMasterFile to PMacc #1498
   - unify cuda error printing #1484
   - add particle ID provider #1409 #1373
   - split off HostDeviceBuffer from GridBuffer  #1370
   - add a policy to GetKeyFromAlias #1252
   - Add border mapping #1133, #1169 #1224
   - make cuSTL gather accept CartBuffers and handle pitches #1196
   - add reference accessors to complex type #1198
   - add more rounding functions #1099
   - add conversion operator from `uint3` to `Dataspace` #1145
   - add more specializations to `GetMPI_StructAsArray` #1088
   - implement cartBuffer conversion for HostBuffer #1092
   - add a policy for async communication #1079
   - add policies for handling particles in guard cells #1077
   - support more types in atomicAddInc and warpBroadcast #1078
   - calculate better seeds #1040 #1046
   - move MallocMCBuffer to PMacc #1034
   - move TypeToPointerPair to PMacc #1033
   - add 1D, 2D and 3D linear interpolation cursor #1217 #1448
   - add method 'getPluginFromType()' to `PluginConnector` #1393
   - math:
     - add `abs`, `asin`, `acos`, `atan`, `log10`, `fmod`, `modf`, `floor` to algorithms::math #837 #1218 #1334 #1362 #1363 #1374 #1473
     - `precisionCast<>` for `PMacc::math::Vector<>` #746
     - support for `boost::mpl::integral_c<>` in `math::CT::Vector<>` #802
     - add complex support #664
   - add `cuSTL/MapTo1DNavigator` #940
   - add 2D support for cuSTL::algorithm::mpi::Gather #844
   - names for exchanges #1511
   - rename EnvMemoryInfo to MemoryInfo #1301
   - mallocMC (*Memory Allocator for Many Core Architectures*) #640 #747 #903 #977  #1171 #1148
     - remove `HeapDataBox`, `RingDataBox`, `HeapBuffer`, `RingBuffer` #640
     - out of heap memory detection #756
     - support to read mallocMC heap on host side #905
   - add multi species support for plugins #794
   - add traits:
     - `GetDataBoxType` #728
     - `FilterByFlag` #1219
     - `GetUniqueTypeId` #957 #962
     - `GetDefaultConstructibleType` #1045
     - `GetInitializedInstance` #1447
     - `ResolveAliasFromSpecies` #1451
     - `GetStringProperties` #1507
   - add pointer class for particles `FramePointer` #1055
   - independent sizes on device for `GridBuffer<>::addExchange`
   - `Communicator`: query periodic directions #1510
   - add host side support for kernel index mapper #902
   - optimize size of particle frame for border frames #949
   - add pre-processor macro for struct generation #972
   - add warp collective atomic function #1013
   - speedup particle operations with collective atomics #1014
   - add support to `deselect` unknown attributes in a particle #1524
   - add `boost.test` #1245
     - test for `HostBufferIntern` #1258
     - test for `setValue()` #1268
  - add resource monitor #1456
  - add MSVC compatibility #816 #821 #931
  - `const` box'es return `const pointer` #945
  - refactor host/device identifier #946

**Bug Fixes:**
 - laser implementations:
   - make math calls more robust & portable #1160
   - amplitude of Gaussian beam in 2D3V simulations #1052 #1090
   - avoid non zero E-field integral in plane wave #851
   - fix length setup of plane wave laser #881
   - few-cycle wavepacket #875
   - fix documentaion of `a_0` conversation #1043
 - FieldTmp Lamor power calculation #1287
 - field solver:
   - stricter condition checks #880
   - 2D3V `NoSolver` did not compile #1073
   - more experimental methods for DS #894
   - experimental: possible out of memory access in directional splitting #890
 - moving window moved not exactly with c #1273 #1337 #1549
 - 2D3V: possible race conditions for very small, non-default super-cells in current deposition (`StrideMapping`) #1405
 - experimental: 2D3V zigzag current deposition fix for `v_z != 0` #823
 - vaccuum: division by zero in `Quiet` particle start #1527
 - remove variable length arrays #932
 - gas (density) profiles:
   - gasFreeFormula #988 #899
   - gaussianCloud #807 #1136 #1265
 - C++ should catch by const reference #1295
 - fix possible underflow on low memory situations #1188
 - C++11 compatibility: use `BOOST_STATIC_CONSTEXPR` where possible #1165
 - avoid CUDA 6.5 int(bool) cast bug #680
 - PMacc detection in CMake #808
 - PMacc:
   - EventPool could run out of free events, potential deadlock #1631
   - Particle<>: avoid using CUDA lmem #1579
   - possible deadlock in event system could freeze simulation #1326
   - HostBuffer includes & constructor #1255 #1596
   - const references in Foreach #1593
   - initialize pointers with NULL before cudaMalloc #1180
   - report device properties of correct GPU #1115
   - rename `types.h` to `pmacc_types.hpp`  #1367
   - add missing const for getter in GridLayout  #1492
   - Cuda event fix to avoid deadlock #1485
   - use Host DataBox in Hostbuffer #1467
   - allow 1D in CommunicatorMPI #1412
   - use better type for params in vector  #1223
   - use correct sqrt function for abs(Vector) #1461
   - fix `CMAKE_PREFIX_PATH`s #1391, #1390
   - remove unnecessary floating point ops from reduce #1212
   - set pointers to NULL before calling cudaMalloc #1180
   - do not allocate memory if not gather root #1181
   - load plugins in registered order #1174
   - C++11 compatibility: use `BOOST_STATIC_CONSTEXPR` where possible #1176 #1175
   - fix usage of `boost::result_of` #1151
   - use correct device number #1115
   - fix vector shrink function #1113
   - split EventSystem.hpp into hpp and tpp #1068
   - fix move operators of CartBuffer #1091
   - missing includes in MapTuple #627
   - GoL example: fix offset #1023
   - remove deprecated throw declarations #1000
   - cuSTL:
     - `cudaPitchedPtr.xsize` used wrong #1234
     - gather for supporting static load balancing #1244
     - reduce #936
     - throw exception on cuda error #1235
     - `DeviceBuffer` assign operator #1375, #1308, #1463, #1435, #1401, #1220, #1197
     - Host/DeviceBuffers: Contructors (Pointers) #1094
     - let kernel/runtime/Foreach compute best BlockDim #1309
   - compile with CUDA 7.0 #748
   - device selection with `process exclusive` enabled #757
   - `math::Vector<>` assignment #806
   - `math::Vector<>` copy constructor #872
   - operator[] in `ConstVector` #981
   - empty `AllCombinations<...>` #1230
   - racecondition in `kernelShiftParticles` #1049
   - warning in `FieldManipulator` #1254
   - memory pitch bug in `MultiBox` and `PitchedBox` #1096
   - `math::abs()` for the type `double` #1470
   - invalid kernel call in `kernelSetValue<>` #1407
   - data alignment for kernel parameter #1566
   - `rsqrt` usage on host #967
   - invalid namespace qualifier #968
   - missing namespace prefix #971
 - plugins:
   - radiation:
     - enable multi species for radiation plugin #1454
     - compile issues with math in radiation #1552
     - documentation of radiation observer setup #1422
     - gamma filter in radiation plugin #1421
     - improve vector type name encapsuling #998
     - saveguard  restart #716
   - CUDA 7.0+ warning in `PhaseSpace` #750
   - racecondition in `ConcatListOfFrames` #1278
   - illegal memory acces in `Visualisation` #1526
   - HDF5 restart: particle offset overflow fixed #721
 - tools:
   - mpiInfo: add missing include #786
   - actually exit when pression no in compilesuite #1411
   - fix incorrect mangling of params  #1385
   - remove deprecated throw declarations #1003
   - make tool python3 compatible #1416
   - trace generating tool #1264
   - png2gas memory leak fixed #1222
   - tbg:
     - quoting interpretation #801
     - variable assignments stay in `.start` files #695 #1609
     - multiple variable use in one line possible #699 #1610
     - failing assignments at template evaluation time keep vars undefined #1611
   - heating tool supports multi species #729
   - fix numerical heating tool normalization #825
   - fix logic behind fill color of numerical heating tool #779
 - libSplash minimum version check #1284

**Misc:**
 - 2D3V simulations are now honoring the cell "depth" in z to make
   density interpretations easier #1569
 - update documentation for dependencies and installation #1556, 1557, #1559, #1127
 - refactor usage of several math functions #1462, #1468
 - FieldJ interface clear() replaced with an explicit assign(x) #1335
 - templates for known systems updated:
   - renaming directories into "cluster-insitutition"
   - tbg copies cmakeFlags now #1101
   - tbg aborts if mkdir fails #797
   - `*tpl` & `*.profile.example` files updated
   - system updates: #937 #1266 #1297 #1329 #1364 #1426 #1512 #1443 #1493
     - Lawrencium (LBNL)
     - Titan/Rhea (ORNL)
     - Piz Daint (CSCS)
     - Taurus (TUD) #1081 #1130  #1114 #1116 #1111 #1137
 - replace deprecated CUDA calls #758
 - remove support for CUDA devices with `sm_10`, `sm_11`, `sm_12` and `sm_13` #813
 - remove unused/unsupported/broken plugins #773 843
   - IntensityPlugin, LiveViewPlugin(2D), SumCurrents, divJ #843
 - refactor `value_identifier` #964
 - remove native type `double` and `float` #985 #990
 - remove `__startAtomicTransaction()` #1233
 - remove `__syncthreads()` after shared memory allocation #1082
 - refactor `ParticleBox` interface #1243
 - rotating root in `GatherSlice` (reduce load of master node) #992
 - reduce `GatherSlice` memory footprint #1282
 - remove `None` type of ionize, pusher #1238 #1227
 - remove math function implementations from `Vector.hpp`
 - remove unused defines #921
 - remove deprecated thow declaration #918
 - remove invalid `typename` #917 #933
 - rename particle algorithms from `...clone...` to `...derive...`  #1525
 - remove math functions from Vector.hpp  #1472
 - raditation plugin remove `unint` with `uint32_t` #1007
 - GoL example: CMake modernized #1138
 - INSTALL.md
   - moved from `/doc/` to `/`
   - now in root of the repo #1521
   - add environment variable `$PICHOME` #1162
   - more portable #1164
   - arch linux instructions #1065
 - refactor ionization towards independence from `Particle` class #874
 - update submit templates for hypnos #860 #861 #862
 - doxygen config and code modernized #1371 #1388
 - cleanup of stdlib includes #1342 #1346 #1347 #1348 #1368 #1389
 - boost 1.60.0 only builds in C++11 mode #1315 #1324 #1325
 - update minimal CMake version to 3.1.0 #1289
 - simplify HostMemAssigner #1320
 - add asserts to cuSTL containers #1248
 - rename TwistVectorAxes -> TwistComponents (cuSTL) #893
 - add more robust namespace qualifiers #839 #969 #847 #974
 - cleanup code #885 #814 #815 #915 #920 #1027 #1011 #1009
 - correct spelling #934 #938 #941
 - add compile test for ALL pushers #1205
 - tools:
   - adjust executable rights and shebang #1110 #1107 #1104 #1085 #1143
   - live visualization client added #681 #835 #1408
 - CMake
   - modernized #1139
   - only allow out-of-source builds #1119
   - cleanup score-p section #1413
   - add `OpenMP` support #904
 - shipped third party updates:
   - restructured #717
   - `cuda_memtest` #770 #1159
   - CMake modules #1087 #1310 #1533
 - removed several `-Wshadow` warnings #1039 #1061 #1070 #1071


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
 - PMacc:
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
 - PMacc:
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
   - `png` plugin write speedup 2.3x by increasing file size about 12% #698
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
supported by PMacc/PIConGPU.
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
 - PMacc
   - added `math::erf` #525
   - experimental 32bit host-side support (JetsonTK1 dev kits) #571
   - `CT::Vector` refactored and new methods added #473
   - cuSTL: better 2D container support #461

**Bug Fixes:**
 - esirkepov + CIC current deposition could cause a deadlock in some situations #475
 - initialization for `kernelSetDrift` was broken (traversal of frame lists, CUDA 5.5+) #538 #539
 - the particleToField deposition (e.g. in FieldTmp solvers for analysis)
   forgot a small fraction of the particle #559
 - PMacc
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
 - PMacc
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
 - PMacc
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
 - PMacc
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
 - PMacc
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
 - PMacc
   - new functors for multiplications and substractions #135
   - opened more interfaces to old functors #197
   - MappedMemoryBuffer added #169 #182
   - unary transformations can be performed on DataBox'es now,
     allowing for non-commutative operations in reduces #204

**Bug fixes:**
 - PMacc
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
 - PMacc
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
 - PMacc
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
 - PMacc
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
