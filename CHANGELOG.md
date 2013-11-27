Change Log / Release Log for PIConGPU
================================================================

Open Beta RC2
-------------
**Date:** 2013-11-27

This is the second release candidate, a *pre-beta* version.

### Changes to "Open Beta RC1"

**.param file changes:**
 - gasConfig.param:
   - add gasFreeFormula #96
     (example how to update your existing gasConfig.param, see
     [https://github.com/ComputationalRadiationPhysics/picongpu/pull/96/files#diff-1])
   - add inner radius to gasSphereFlanks #66
     (example how to update your existing gasConfig.param, see
     [https://github.com/ComputationalRadiationPhysics/picongpu/pull/66/files#diff-0])

**New Features:**
 - A change log was introduced for master releases #93
 - new gas profile "gasFreeFormula" for user defined profiles #96
 - CMake (config) #79
   - checks for minimal required versions of dependent libraries #92
   - checks for libSplash version #85
   - update to v2.8.5+ #52
   - implizit plugin selection: enabled if found #52
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
   - Vampir support: cudaDeviceReset added #90
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
 - dropped sm_13 support (now sm_20+ is required) #42


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
