/* Copyright 2023 Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    ifdef __SYCL_DEVICE_ONLY__

#        if defined(__SYCL_TARGET_INTEL_GPU_BDW__) || /* Broadwell Intel graphics architecture */                     \
            defined(__SYCL_TARGET_INTEL_GPU_SKL__) || /* Skylake Intel graphics architecture */                       \
            defined(__SYCL_TARGET_INTEL_GPU_KBL__) || /* Kaby Lake Intel graphics architecture */                     \
            defined(__SYCL_TARGET_INTEL_GPU_CFL__) || /* Coffee Lake Intel graphics architecture */                   \
            defined(__SYCL_TARGET_INTEL_GPU_APL__) || /* Apollo Lake Intel graphics architecture */                   \
            defined(__SYCL_TARGET_INTEL_GPU_GLK__) || /* Gemini Lake Intel graphics architecture */                   \
            defined(__SYCL_TARGET_INTEL_GPU_WHL__) || /* Whiskey Lake Intel graphics architecture */                  \
            defined(__SYCL_TARGET_INTEL_GPU_AML__) || /* Amber Lake Intel graphics architecture */                    \
            defined(__SYCL_TARGET_INTEL_GPU_CML__) || /* Comet Lake Intel graphics architecture */                    \
            defined(__SYCL_TARGET_INTEL_GPU_ICLLP__) || /* Ice Lake Intel graphics architecture */                    \
            defined(__SYCL_TARGET_INTEL_GPU_TGLLP__) || /* Tiger Lake Intel graphics architecture */                  \
            defined(__SYCL_TARGET_INTEL_GPU_RKL__) || /* Rocket Lake Intel graphics architecture */                   \
            defined(__SYCL_TARGET_INTEL_GPU_ADL_S__) || /* Alder Lake S Intel graphics architecture */                \
            defined(__SYCL_TARGET_INTEL_GPU_RPL_S__) || /* Raptor Lake Intel graphics architecture */                 \
            defined(__SYCL_TARGET_INTEL_GPU_ADL_P__) || /* Alder Lake P Intel graphics architecture */                \
            defined(__SYCL_TARGET_INTEL_GPU_ADL_N__) || /* Alder Lake N Intel graphics architecture */                \
            defined(__SYCL_TARGET_INTEL_GPU_DG1__) || /* DG1 Intel graphics architecture */                           \
            defined(__SYCL_TARGET_INTEL_GPU_ACM_G10__) || /* Alchemist G10 Intel graphics architecture */             \
            defined(__SYCL_TARGET_INTEL_GPU_ACM_G11__) || /* Alchemist G11 Intel graphics architecture */             \
            defined(__SYCL_TARGET_INTEL_GPU_ACM_G12__) /* Alchemist G12 Intel graphics architecture */

#            define SYCL_SUBGROUP_SIZE (8 | 16 | 32)

#        elif defined(__SYCL_TARGET_INTEL_GPU_PVC__) /* Ponte Vecchio Intel graphics architecture */

#            define SYCL_SUBGROUP_SIZE (16 | 32)

#        elif defined(__SYCL_TARGET_INTEL_X86_64__) /* generate code ahead of time for x86_64 CPUs */

#            define SYCL_SUBGROUP_SIZE (4 | 8 | 16 | 32 | 64)

#        elif defined(__SYCL_TARGET_NVIDIA_GPU_SM_50__) || /* NVIDIA Maxwell architecture (compute capability 5.0) */ \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_52__) || /* NVIDIA Maxwell architecture (compute capability 5.2) */   \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_53__) || /* NVIDIA Jetson TX1 / Nano (compute capability 5.3) */      \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_60__) || /* NVIDIA Pascal architecture (compute capability 6.0) */    \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_61__) || /* NVIDIA Pascal architecture (compute capability 6.1) */    \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_62__) || /* NVIDIA Jetson TX2 (compute capability 6.2) */             \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_70__) || /* NVIDIA Volta architecture (compute capability 7.0) */     \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_72__) || /* NVIDIA Jetson AGX (compute capability 7.2) */             \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_75__) || /* NVIDIA Turing architecture (compute capability 7.5) */    \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_80__) || /* NVIDIA Ampere architecture (compute capability 8.0) */    \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_86__) || /* NVIDIA Ampere architecture (compute capability 8.6) */    \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_87__) || /* NVIDIA Jetson/Drive AGX Orin (compute capability 8.7) */  \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_89__) || /* NVIDIA Ada Lovelace arch. (compute capability 8.9) */     \
            defined(__SYCL_TARGET_NVIDIA_GPU_SM_90__) /* NVIDIA Hopper architecture (compute capability 9.0) */

#            define SYCL_SUBGROUP_SIZE (32)

#        elif defined(__SYCL_TARGET_AMD_GPU_GFX700__) || /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */         \
            defined(__SYCL_TARGET_AMD_GPU_GFX701__) || /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */           \
            defined(__SYCL_TARGET_AMD_GPU_GFX702__) || /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */           \
            defined(__SYCL_TARGET_AMD_GPU_GFX801__) || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */      \
            defined(__SYCL_TARGET_AMD_GPU_GFX802__) || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */      \
            defined(__SYCL_TARGET_AMD_GPU_GFX803__) || /* AMD GCN 4.0 Arctic Islands architecture (gfx 8.0) */        \
            defined(__SYCL_TARGET_AMD_GPU_GFX805__) || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */      \
            defined(__SYCL_TARGET_AMD_GPU_GFX810__) || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.1) */      \
            defined(__SYCL_TARGET_AMD_GPU_GFX900__) || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                  \
            defined(__SYCL_TARGET_AMD_GPU_GFX902__) || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                  \
            defined(__SYCL_TARGET_AMD_GPU_GFX904__) || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                  \
            defined(__SYCL_TARGET_AMD_GPU_GFX906__) || /* AMD GCN 5.1 Vega II architecture (gfx 9.0) */               \
            defined(__SYCL_TARGET_AMD_GPU_GFX908__) || /* AMD CDNA 1.0 Arcturus architecture (gfx 9.0) */             \
            defined(__SYCL_TARGET_AMD_GPU_GFX90A__) /* AMD CDNA 2.0 Aldebaran architecture (gfx 9.0) */

#            define SYCL_SUBGROUP_SIZE (64)

#        elif defined(__SYCL_TARGET_AMD_GPU_GFX1010__) || /* AMD RDNA 1.0 Navi 10 architecture (gfx 10.1) */          \
            defined(__SYCL_TARGET_AMD_GPU_GFX1011__) || /* AMD RDNA 1.0 Navi 12 architecture (gfx 10.1) */            \
            defined(__SYCL_TARGET_AMD_GPU_GFX1012__) || /* AMD RDNA 1.0 Navi 14 architecture (gfx 10.1) */            \
            defined(__SYCL_TARGET_AMD_GPU_GFX1013__) || /* AMD RDNA 2.0 Oberon architecture (gfx 10.1) */             \
            defined(__SYCL_TARGET_AMD_GPU_GFX1030__) || /* AMD RDNA 2.0 Navi 21 architecture (gfx 10.3) */            \
            defined(__SYCL_TARGET_AMD_GPU_GFX1031__) || /* AMD RDNA 2.0 Navi 22 architecture (gfx 10.3) */            \
            defined(__SYCL_TARGET_AMD_GPU_GFX1032__) || /* AMD RDNA 2.0 Navi 23 architecture (gfx 10.3) */            \
            defined(__SYCL_TARGET_AMD_GPU_GFX1034__) /* AMD RDNA 2.0 Navi 24 architecture (gfx 10.3) */

#            define SYCL_SUBGROUP_SIZE (32 | 64)

#        else // __SYCL_TARGET_*

#            define SYCL_SUBGROUP_SIZE (0) /* unknown target */

#        endif // __SYCL_TARGET_*

#    else

#        define SYCL_SUBGROUP_SIZE (0) /* host compilation */

#    endif // __SYCL_DEVICE_ONLY__

#endif // ALPAKA_ACC_SYCL_ENABLED
