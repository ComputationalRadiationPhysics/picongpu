/* Copyright 2023 Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    ifdef __SYCL_DEVICE_ONLY__

#        if(__SYCL_TARGET_INTEL_GPU_BDW__) || /* Broadwell Intel graphics architecture */                             \
            (__SYCL_TARGET_INTEL_GPU_SKL__) || /* Skylake Intel graphics architecture */                              \
            (__SYCL_TARGET_INTEL_GPU_KBL__) || /* Kaby Lake Intel graphics architecture */                            \
            (__SYCL_TARGET_INTEL_GPU_CFL__) || /* Coffee Lake Intel graphics architecture */                          \
            (__SYCL_TARGET_INTEL_GPU_APL__) || /* Apollo Lake Intel graphics architecture */                          \
            (__SYCL_TARGET_INTEL_GPU_GLK__) || /* Gemini Lake Intel graphics architecture */                          \
            (__SYCL_TARGET_INTEL_GPU_WHL__) || /* Whiskey Lake Intel graphics architecture */                         \
            (__SYCL_TARGET_INTEL_GPU_AML__) || /* Amber Lake Intel graphics architecture */                           \
            (__SYCL_TARGET_INTEL_GPU_CML__) || /* Comet Lake Intel graphics architecture */                           \
            (__SYCL_TARGET_INTEL_GPU_ICLLP__) || /* Ice Lake Intel graphics architecture */                           \
            (__SYCL_TARGET_INTEL_GPU_EHL__) || /* Elkhart Lake or Jasper Lake Intel graphics architecture */          \
            (__SYCL_TARGET_INTEL_GPU_TGLLP__) || /* Tiger Lake Intel graphics architecture */                         \
            (__SYCL_TARGET_INTEL_GPU_RKL__) || /* Rocket Lake Intel graphics architecture */                          \
            (__SYCL_TARGET_INTEL_GPU_ADL_S__) || /* Alder Lake S or Raptor Lake S Intel graphics architecture */      \
            (__SYCL_TARGET_INTEL_GPU_ADL_P__) || /* Alder Lake P Intel graphics architecture */                       \
            (__SYCL_TARGET_INTEL_GPU_ADL_N__) || /* Alder Lake N Intel graphics architecture */                       \
            (__SYCL_TARGET_INTEL_GPU_DG1__) || /* DG1 Intel graphics architecture */                                  \
            (__SYCL_TARGET_INTEL_GPU_ACM_G10__) || /* Alchemist G10 Intel graphics architecture */                    \
            (__SYCL_TARGET_INTEL_GPU_ACM_G11__) || /* Alchemist G11 Intel graphics architecture */                    \
            (__SYCL_TARGET_INTEL_GPU_ACM_G12__) || /* Alchemist G12 Intel graphics architecture */                    \
            (__SYCL_TARGET_INTEL_GPU_MTL_U__) || /* Meteor Lake U/S or Arrow Lake U/S Intel graphics architecture */  \
            (__SYCL_TARGET_INTEL_GPU_MTL_H__) || /* Meteor Lake H Intel graphics architecture */                      \
            (__SYCL_TARGET_INTEL_GPU_ARL_H__) || /* Arrow Lake H Intel graphics architecture */                       \
            (__SYCL_TARGET_INTEL_GPU_BMG_G21__) || /* Battlemage G21 Intel graphics architecture */                   \
            (__SYCL_TARGET_INTEL_GPU_LNL_M__) /* Lunar Lake Intel graphics architecture */

#            define SYCL_SUBGROUP_SIZE (8 | 16 | 32)

#        elif(__SYCL_TARGET_INTEL_GPU_PVC__) || /* Ponte Vecchio Intel graphics architecture */                       \
            (__SYCL_TARGET_INTEL_GPU_PVC_VG__) /* Ponte Vecchio VG Intel graphics architecture */

#            define SYCL_SUBGROUP_SIZE (16 | 32)

#        elif(__SYCL_TARGET_INTEL_X86_64__) /* generate code ahead of time for x86_64 CPUs */

#            define SYCL_SUBGROUP_SIZE (4 | 8 | 16 | 32 | 64)

#        elif(__SYCL_TARGET_NVIDIA_GPU_SM50__) || /* NVIDIA Maxwell architecture (compute capability 5.0) */          \
            (__SYCL_TARGET_NVIDIA_GPU_SM52__) || /* NVIDIA Maxwell architecture (compute capability 5.2) */           \
            (__SYCL_TARGET_NVIDIA_GPU_SM53__) || /* NVIDIA Jetson TX1 / Nano (compute capability 5.3) */              \
            (__SYCL_TARGET_NVIDIA_GPU_SM60__) || /* NVIDIA Pascal architecture (compute capability 6.0) */            \
            (__SYCL_TARGET_NVIDIA_GPU_SM61__) || /* NVIDIA Pascal architecture (compute capability 6.1) */            \
            (__SYCL_TARGET_NVIDIA_GPU_SM62__) || /* NVIDIA Jetson TX2 (compute capability 6.2) */                     \
            (__SYCL_TARGET_NVIDIA_GPU_SM70__) || /* NVIDIA Volta architecture (compute capability 7.0) */             \
            (__SYCL_TARGET_NVIDIA_GPU_SM72__) || /* NVIDIA Jetson AGX (compute capability 7.2) */                     \
            (__SYCL_TARGET_NVIDIA_GPU_SM75__) || /* NVIDIA Turing architecture (compute capability 7.5) */            \
            (__SYCL_TARGET_NVIDIA_GPU_SM80__) || /* NVIDIA Ampere architecture (compute capability 8.0) */            \
            (__SYCL_TARGET_NVIDIA_GPU_SM86__) || /* NVIDIA Ampere architecture (compute capability 8.6) */            \
            (__SYCL_TARGET_NVIDIA_GPU_SM87__) || /* NVIDIA Jetson/Drive AGX Orin (compute capability 8.7) */          \
            (__SYCL_TARGET_NVIDIA_GPU_SM89__) || /* NVIDIA Ada Lovelace arch. (compute capability 8.9) */             \
            (__SYCL_TARGET_NVIDIA_GPU_SM90__) /* NVIDIA Hopper architecture (compute capability 9.0) */

#            define SYCL_SUBGROUP_SIZE (32)

#        elif(__SYCL_TARGET_AMD_GPU_GFX700__) || /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */                 \
            (__SYCL_TARGET_AMD_GPU_GFX701__) || /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */                  \
            (__SYCL_TARGET_AMD_GPU_GFX702__) || /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */                  \
            (__SYCL_TARGET_AMD_GPU_GFX801__) || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */             \
            (__SYCL_TARGET_AMD_GPU_GFX802__) || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */             \
            (__SYCL_TARGET_AMD_GPU_GFX803__) || /* AMD GCN 4.0 Arctic Islands architecture (gfx 8.0) */               \
            (__SYCL_TARGET_AMD_GPU_GFX805__) || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */             \
            (__SYCL_TARGET_AMD_GPU_GFX810__) || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.1) */             \
            (__SYCL_TARGET_AMD_GPU_GFX900__) || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                         \
            (__SYCL_TARGET_AMD_GPU_GFX902__) || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                         \
            (__SYCL_TARGET_AMD_GPU_GFX904__) || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                         \
            (__SYCL_TARGET_AMD_GPU_GFX906__) || /* AMD GCN 5.1 Vega II architecture (gfx 9.0) */                      \
            (__SYCL_TARGET_AMD_GPU_GFX908__) || /* AMD CDNA 1.0 Arcturus architecture (gfx 9.0) */                    \
            (__SYCL_TARGET_AMD_GPU_GFX909__) || /* AMD GCN 5.0 Raven 2 architecture (gfx 9.0) */                      \
            (__SYCL_TARGET_AMD_GPU_GFX90A__) || /* AMD CDNA 2.0 Aldebaran architecture (gfx 9.0) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX90C__) || /* AMD GCN 5.1 Renoir architecture (gfx 9.0) */                       \
            (__SYCL_TARGET_AMD_GPU_GFX940__) || /* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */               \
            (__SYCL_TARGET_AMD_GPU_GFX941__) || /* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */               \
            (__SYCL_TARGET_AMD_GPU_GFX942__) /* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */

#            define SYCL_SUBGROUP_SIZE (64)

#        elif(__SYCL_TARGET_AMD_GPU_GFX1010__) || /* AMD RDNA 1.0 Navi 10 architecture (gfx 10.1) */                  \
            (__SYCL_TARGET_AMD_GPU_GFX1011__) || /* AMD RDNA 1.0 Navi 12 architecture (gfx 10.1) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1012__) || /* AMD RDNA 1.0 Navi 14 architecture (gfx 10.1) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1013__) || /* AMD RDNA 2.0 Oberon architecture (gfx 10.1) */                    \
            (__SYCL_TARGET_AMD_GPU_GFX1030__) || /* AMD RDNA 2.0 Navi 21 architecture (gfx 10.3) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1031__) || /* AMD RDNA 2.0 Navi 22 architecture (gfx 10.3) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1032__) || /* AMD RDNA 2.0 Navi 23 architecture (gfx 10.3) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1033__) || /* AMD RDNA 2.0 Van Gogh architecture (gfx 10.3) */                  \
            (__SYCL_TARGET_AMD_GPU_GFX1034__) || /* AMD RDNA 2.0 Navi 24 architecture (gfx 10.3) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1035__) || /* AMD RDNA 2.0 Rembrandt Mobile architecture (gfx 10.3) */          \
            (__SYCL_TARGET_AMD_GPU_GFX1036__) || /* AMD RDNA 2.0 Raphael architecture (gfx 10.3) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1100__) || /* AMD RDNA 3.0 Navi 31 architecture (gfx 11.0) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1101__) || /* AMD RDNA 3.0 Navi 32 architecture (gfx 11.0) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1102__) || /* AMD RDNA 3.0 Navi 33 architecture (gfx 11.0) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1103__) || /* AMD RDNA 3.0 Phoenix mobile architecture (gfx 11.0) */            \
            (__SYCL_TARGET_AMD_GPU_GFX1150__) || /* AMD RDNA 3.5 Strix Point architecture (gfx 11.5) */               \
            (__SYCL_TARGET_AMD_GPU_GFX1151__) || /* AMD RDNA 3.5 Strix Halo architecture (gfx 11.5) */                \
            (__SYCL_TARGET_AMD_GPU_GFX1200__) || /* AMD RDNA 4.0 Navi 44 architecture (gfx 12.0) */                   \
            (__SYCL_TARGET_AMD_GPU_GFX1201__) /* AMD RDNA 4.0 Navi 48 architecture (gfx 12.0) */

// starting from gfx10, HIP supports only wavefront size 32
#            define SYCL_SUBGROUP_SIZE (32)

#        else // __SYCL_TARGET_*

#            define SYCL_SUBGROUP_SIZE (0) /* unknown target */

#        endif // __SYCL_TARGET_*

#    else

#        define SYCL_SUBGROUP_SIZE (0) /* host compilation */

#    endif // __SYCL_DEVICE_ONLY__

#endif // ALPAKA_ACC_SYCL_ENABLED
