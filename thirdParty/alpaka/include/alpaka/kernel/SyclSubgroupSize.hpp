/* Copyright 2023 Andrea Bocci, Aurora Perego, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#ifdef ALPAKA_ACC_SYCL_ENABLED

/* Device target macros are generated from clang offload architecture names. Source links:
 * - clang offload architecture wrapper:
 *   https://github.com/llvm/llvm-project/blob/llvmorg-24-init/clang/include/clang/Basic/OffloadArch.h
 * - Intel SYCL offload architecture list:
 *   https://github.com/intel/llvm/blob/sycl/clang/include/clang/Basic/OffloadArch.h
 * - NVIDIA target parser list:
 *   https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/TargetParser/NVPTXTargetParser.def
 * - AMDGPU processor table:
 *   https://llvm.org/docs/AMDGPUUsage.html#processors
 */

#    ifdef __SYCL_DEVICE_ONLY__

#        if /* Broadwell Intel graphics architecture */                                                               \
            (defined(__SYCL_TARGET_INTEL_GPU_BDW__) && __SYCL_TARGET_INTEL_GPU_BDW__)                                 \
            || /* Skylake Intel graphics architecture */                                                              \
            (defined(__SYCL_TARGET_INTEL_GPU_SKL__) && __SYCL_TARGET_INTEL_GPU_SKL__)                                 \
            || /* Kaby Lake Intel graphics architecture */                                                            \
            (defined(__SYCL_TARGET_INTEL_GPU_KBL__) && __SYCL_TARGET_INTEL_GPU_KBL__)                                 \
            || /* Coffee Lake Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_CFL__) && __SYCL_TARGET_INTEL_GPU_CFL__)                                 \
            || /* Apollo Lake Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_APL__) && __SYCL_TARGET_INTEL_GPU_APL__)                                 \
            || /* Broxton Intel graphics architecture */                                                              \
            (defined(__SYCL_TARGET_INTEL_GPU_BXT__) && __SYCL_TARGET_INTEL_GPU_BXT__)                                 \
            || /* Gemini Lake Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_GLK__) && __SYCL_TARGET_INTEL_GPU_GLK__)                                 \
            || /* Whiskey Lake Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_WHL__) && __SYCL_TARGET_INTEL_GPU_WHL__)                                 \
            || /* Amber Lake Intel graphics architecture */                                                           \
            (defined(__SYCL_TARGET_INTEL_GPU_AML__) && __SYCL_TARGET_INTEL_GPU_AML__)                                 \
            || /* Comet Lake Intel graphics architecture */                                                           \
            (defined(__SYCL_TARGET_INTEL_GPU_CML__) && __SYCL_TARGET_INTEL_GPU_CML__)                                 \
            || /* Ice Lake LP Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_ICLLP__) && __SYCL_TARGET_INTEL_GPU_ICLLP__)                             \
            || /* Ice Lake Intel graphics architecture */                                                             \
            (defined(__SYCL_TARGET_INTEL_GPU_ICL__) && __SYCL_TARGET_INTEL_GPU_ICL__)                                 \
            || /* Elkhart Lake or Jasper Lake Intel graphics architecture */                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_EHL__) && __SYCL_TARGET_INTEL_GPU_EHL__)                                 \
            || /* Jasper Lake Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_JSL__) && __SYCL_TARGET_INTEL_GPU_JSL__)                                 \
            || /* Tiger Lake LP Intel graphics architecture */                                                        \
            (defined(__SYCL_TARGET_INTEL_GPU_TGLLP__) && __SYCL_TARGET_INTEL_GPU_TGLLP__)                             \
            || /* Tiger Lake Intel graphics architecture */                                                           \
            (defined(__SYCL_TARGET_INTEL_GPU_TGL__) && __SYCL_TARGET_INTEL_GPU_TGL__)                                 \
            || /* Rocket Lake Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_RKL__) && __SYCL_TARGET_INTEL_GPU_RKL__)                                 \
            || /* Alder Lake S or Raptor Lake S Intel graphics architecture */                                        \
            (defined(__SYCL_TARGET_INTEL_GPU_ADL_S__) && __SYCL_TARGET_INTEL_GPU_ADL_S__)                             \
            || /* Raptor Lake S Intel graphics architecture */                                                        \
            (defined(__SYCL_TARGET_INTEL_GPU_RPL_S__) && __SYCL_TARGET_INTEL_GPU_RPL_S__)                             \
            || /* Alder Lake P Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_ADL_P__) && __SYCL_TARGET_INTEL_GPU_ADL_P__)                             \
            || /* Alder Lake N Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_ADL_N__) && __SYCL_TARGET_INTEL_GPU_ADL_N__)                             \
            || /* DG1 Intel graphics architecture */                                                                  \
            (defined(__SYCL_TARGET_INTEL_GPU_DG1__) && __SYCL_TARGET_INTEL_GPU_DG1__)                                 \
            || /* DG2 Intel graphics architecture */                                                                  \
            (defined(__SYCL_TARGET_INTEL_GPU_DG2__) && __SYCL_TARGET_INTEL_GPU_DG2__)                                 \
            || /* Alchemist G10 Intel graphics architecture */                                                        \
            (defined(__SYCL_TARGET_INTEL_GPU_ACM_G10__) && __SYCL_TARGET_INTEL_GPU_ACM_G10__)                         \
            || /* DG2 G10 Intel graphics architecture */                                                              \
            (defined(__SYCL_TARGET_INTEL_GPU_DG2_G10__) && __SYCL_TARGET_INTEL_GPU_DG2_G10__)                         \
            || /* Alchemist G11 Intel graphics architecture */                                                        \
            (defined(__SYCL_TARGET_INTEL_GPU_ACM_G11__) && __SYCL_TARGET_INTEL_GPU_ACM_G11__)                         \
            || /* DG2 G11 Intel graphics architecture */                                                              \
            (defined(__SYCL_TARGET_INTEL_GPU_DG2_G11__) && __SYCL_TARGET_INTEL_GPU_DG2_G11__)                         \
            || /* Alchemist G12 Intel graphics architecture */                                                        \
            (defined(__SYCL_TARGET_INTEL_GPU_ACM_G12__) && __SYCL_TARGET_INTEL_GPU_ACM_G12__)                         \
            || /* DG2 G12 Intel graphics architecture */                                                              \
            (defined(__SYCL_TARGET_INTEL_GPU_DG2_G12__) && __SYCL_TARGET_INTEL_GPU_DG2_G12__)                         \
            || /* Meteor Lake Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_MTL__) && __SYCL_TARGET_INTEL_GPU_MTL__)                                 \
            || /* Meteor Lake U/S or Arrow Lake U/S Intel graphics architecture */                                    \
            (defined(__SYCL_TARGET_INTEL_GPU_MTL_U__) && __SYCL_TARGET_INTEL_GPU_MTL_U__)                             \
            || /* Meteor Lake S Intel graphics architecture */                                                        \
            (defined(__SYCL_TARGET_INTEL_GPU_MTL_S__) && __SYCL_TARGET_INTEL_GPU_MTL_S__)                             \
            || /* Arrow Lake U Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_ARL_U__) && __SYCL_TARGET_INTEL_GPU_ARL_U__)                             \
            || /* Arrow Lake S Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_ARL_S__) && __SYCL_TARGET_INTEL_GPU_ARL_S__)                             \
            || /* Meteor Lake H Intel graphics architecture */                                                        \
            (defined(__SYCL_TARGET_INTEL_GPU_MTL_H__) && __SYCL_TARGET_INTEL_GPU_MTL_H__)                             \
            || /* Arrow Lake H Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_ARL_H__) && __SYCL_TARGET_INTEL_GPU_ARL_H__)                             \
            || /* Battlemage Intel graphics architecture */                                                           \
            (defined(__SYCL_TARGET_INTEL_GPU_BMG__) && __SYCL_TARGET_INTEL_GPU_BMG__)                                 \
            || /* Battlemage G21 Intel graphics architecture */                                                       \
            (defined(__SYCL_TARGET_INTEL_GPU_BMG_G21__) && __SYCL_TARGET_INTEL_GPU_BMG_G21__)                         \
            || /* Battlemage G31 Intel graphics architecture */                                                       \
            (defined(__SYCL_TARGET_INTEL_GPU_BMG_G31__) && __SYCL_TARGET_INTEL_GPU_BMG_G31__)                         \
            || /* Panther Lake Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_PTL__) && __SYCL_TARGET_INTEL_GPU_PTL__)                                 \
            || /* Lunar Lake Intel graphics architecture */                                                           \
            (defined(__SYCL_TARGET_INTEL_GPU_LNL_M__) && __SYCL_TARGET_INTEL_GPU_LNL_M__)                             \
            || /* Panther Lake H Intel graphics architecture */                                                       \
            (defined(__SYCL_TARGET_INTEL_GPU_PTL_H__) && __SYCL_TARGET_INTEL_GPU_PTL_H__)                             \
            || /* Panther Lake U Intel graphics architecture */                                                       \
            (defined(__SYCL_TARGET_INTEL_GPU_PTL_U__) && __SYCL_TARGET_INTEL_GPU_PTL_U__)                             \
            || /* Wildcat Lake Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_WCL__) && __SYCL_TARGET_INTEL_GPU_WCL__)                                 \
            || /* Nova Lake S Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_NVL_S__) && __SYCL_TARGET_INTEL_GPU_NVL_S__)                             \
            || /* Nova Lake U Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_NVL_U__) && __SYCL_TARGET_INTEL_GPU_NVL_U__)                             \
            || /* Nova Lake P Intel graphics architecture */                                                          \
            (defined(__SYCL_TARGET_INTEL_GPU_NVL_P__) && __SYCL_TARGET_INTEL_GPU_NVL_P__)                             \
            || /* Crescent Island Intel graphics architecture */                                                      \
            (defined(__SYCL_TARGET_INTEL_GPU_CRI__) && __SYCL_TARGET_INTEL_GPU_CRI__)

#            define SYCL_SUBGROUP_SIZE (8 | 16 | 32)

#        elif /* Ponte Vecchio Intel graphics architecture */                                                         \
            (defined(__SYCL_TARGET_INTEL_GPU_PVC__) && __SYCL_TARGET_INTEL_GPU_PVC__)                                 \
            || /* Ponte Vecchio VG Intel graphics architecture */                                                     \
            (defined(__SYCL_TARGET_INTEL_GPU_PVC_VG__) && __SYCL_TARGET_INTEL_GPU_PVC_VG__)

#            define SYCL_SUBGROUP_SIZE (16 | 32)

#        elif(/* generate code ahead of time for x86_64 CPUs */                                                       \
              defined(__SYCL_TARGET_INTEL_X86_64__) && __SYCL_TARGET_INTEL_X86_64__)

#            define SYCL_SUBGROUP_SIZE (4 | 8 | 16 | 32 | 64)

#        elif /* NVIDIA Maxwell architecture (compute capability 5.0) */                                              \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_50__) && __SYCL_TARGET_NVIDIA_GPU_SM_50__)                           \
            || /* NVIDIA Maxwell architecture (compute capability 5.2) */                                             \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_52__) && __SYCL_TARGET_NVIDIA_GPU_SM_52__)                           \
            || /* NVIDIA Jetson TX1 / Nano (compute capability 5.3) */                                                \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_53__) && __SYCL_TARGET_NVIDIA_GPU_SM_53__)                           \
            || /* NVIDIA Pascal architecture (compute capability 6.0) */                                              \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_60__) && __SYCL_TARGET_NVIDIA_GPU_SM_60__)                           \
            || /* NVIDIA Pascal architecture (compute capability 6.1) */                                              \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_61__) && __SYCL_TARGET_NVIDIA_GPU_SM_61__)                           \
            || /* NVIDIA Jetson TX2 (compute capability 6.2) */                                                       \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_62__) && __SYCL_TARGET_NVIDIA_GPU_SM_62__)                           \
            || /* NVIDIA Volta architecture (compute capability 7.0) */                                               \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_70__) && __SYCL_TARGET_NVIDIA_GPU_SM_70__)                           \
            || /* NVIDIA Jetson AGX (compute capability 7.2) */                                                       \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_72__) && __SYCL_TARGET_NVIDIA_GPU_SM_72__)                           \
            || /* NVIDIA Turing architecture (compute capability 7.5) */                                              \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_75__) && __SYCL_TARGET_NVIDIA_GPU_SM_75__)                           \
            || /* NVIDIA Ampere architecture (compute capability 8.0) */                                              \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_80__) && __SYCL_TARGET_NVIDIA_GPU_SM_80__)                           \
            || /* NVIDIA Ampere architecture (compute capability 8.6) */                                              \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_86__) && __SYCL_TARGET_NVIDIA_GPU_SM_86__)                           \
            || /* NVIDIA Jetson/Drive AGX Orin (compute capability 8.7) */                                            \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_87__) && __SYCL_TARGET_NVIDIA_GPU_SM_87__)                           \
            || /* NVIDIA architecture (compute capability 8.8) */                                                     \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_88__) && __SYCL_TARGET_NVIDIA_GPU_SM_88__)                           \
            || /* NVIDIA Ada Lovelace arch. (compute capability 8.9) */                                               \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_89__) && __SYCL_TARGET_NVIDIA_GPU_SM_89__)                           \
            || /* NVIDIA Hopper architecture (compute capability 9.0) */                                              \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_90__) && __SYCL_TARGET_NVIDIA_GPU_SM_90__)                           \
            || /*NVIDIA Hopper architecture variant(compute capability 9.0a) */                                       \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_90a__) && __SYCL_TARGET_NVIDIA_GPU_SM_90a__)                         \
            || /* NVIDIA Blackwell architecture (compute capability 10.0) */                                          \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_100__) && __SYCL_TARGET_NVIDIA_GPU_SM_100__)                         \
            || /* NVIDIA Blackwell architecture variant (compute capability 10.0a) */                                 \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_100a__) && __SYCL_TARGET_NVIDIA_GPU_SM_100a__)                       \
            || /* NVIDIA Blackwell architecture family-specific variant (compute capability 10.0f) */                 \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_100f__) && __SYCL_TARGET_NVIDIA_GPU_SM_100f__)                       \
            || /* NVIDIA Blackwell Next architecture (compute capability 10.1) */                                     \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_101__) && __SYCL_TARGET_NVIDIA_GPU_SM_101__)                         \
            || /* NVIDIA Blackwell Next architecture variant (compute capability 10.1a) */                            \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_101a__) && __SYCL_TARGET_NVIDIA_GPU_SM_101a__)                       \
            || /* NVIDIA Blackwell Next architecture family-specific variant (compute capability 10.1f) */            \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_101f__) && __SYCL_TARGET_NVIDIA_GPU_SM_101f__)                       \
            || /* NVIDIA Next-generation architecture (compute capability 10.3) */                                    \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_103__) && __SYCL_TARGET_NVIDIA_GPU_SM_103__)                         \
            || /* NVIDIA Next-generation architecture variant (compute capability 10.3a) */                           \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_103a__) && __SYCL_TARGET_NVIDIA_GPU_SM_103a__)                       \
            || /* NVIDIA Next-generation architecture family-specific variant (compute capability 10.3f) */           \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_103f__) && __SYCL_TARGET_NVIDIA_GPU_SM_103f__)                       \
            || /* NVIDIA Next-generation architecture (compute capability 10.7) */                                    \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_107__) && __SYCL_TARGET_NVIDIA_GPU_SM_107__)                         \
            || /* NVIDIA Next-generation architecture variant (compute capability 10.7a) */                           \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_107a__) && __SYCL_TARGET_NVIDIA_GPU_SM_107a__)                       \
            || /* NVIDIA Next-generation architecture family-specific variant (compute capability 10.7f) */           \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_107f__) && __SYCL_TARGET_NVIDIA_GPU_SM_107f__)                       \
            || /* NVIDIA Next-generation architecture (compute capability 11.0) */                                    \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_110__) && __SYCL_TARGET_NVIDIA_GPU_SM_110__)                         \
            || /* NVIDIA Next-generation architecture variant (compute capability 11.0a) */                           \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_110a__) && __SYCL_TARGET_NVIDIA_GPU_SM_110a__)                       \
            || /* NVIDIA Next-generation architecture family-specific variant (compute capability 11.0f) */           \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_110f__) && __SYCL_TARGET_NVIDIA_GPU_SM_110f__)                       \
            || /* NVIDIA Future architecture (compute capability 12.0) */                                             \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_120__) && __SYCL_TARGET_NVIDIA_GPU_SM_120__)                         \
            || /* NVIDIA Future architecture variant (compute capability 12.0a) */                                    \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_120a__) && __SYCL_TARGET_NVIDIA_GPU_SM_120a__)                       \
            || /* NVIDIA Future architecture family-specific variant (compute capability 12.0f) */                    \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_120f__) && __SYCL_TARGET_NVIDIA_GPU_SM_120f__)                       \
            || /* NVIDIA Future architecture (compute capability 12.1) */                                             \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_121__) && __SYCL_TARGET_NVIDIA_GPU_SM_121__)                         \
            || /* NVIDIA Future architecture variant (compute capability 12.1a) */                                    \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_121a__) && __SYCL_TARGET_NVIDIA_GPU_SM_121a__)                       \
            || /* NVIDIA Future architecture family-specific variant (compute capability 12.1f) */                    \
            (defined(__SYCL_TARGET_NVIDIA_GPU_SM_121f__) && __SYCL_TARGET_NVIDIA_GPU_SM_121f__)

#            define SYCL_SUBGROUP_SIZE (32) /* CUDA supports warp size 32 */

#        elif /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */                                                    \
            (defined(__SYCL_TARGET_AMD_GPU_GFX700__) && __SYCL_TARGET_AMD_GPU_GFX700__)                               \
            || /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */                                                   \
            (defined(__SYCL_TARGET_AMD_GPU_GFX701__) && __SYCL_TARGET_AMD_GPU_GFX701__)                               \
            || /* AMD GCN 2.0 Sea Islands architecture (gfx 7.0) */                                                   \
            (defined(__SYCL_TARGET_AMD_GPU_GFX702__) && __SYCL_TARGET_AMD_GPU_GFX702__)                               \
            || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */                                              \
            (defined(__SYCL_TARGET_AMD_GPU_GFX801__) && __SYCL_TARGET_AMD_GPU_GFX801__)                               \
            || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */                                              \
            (defined(__SYCL_TARGET_AMD_GPU_GFX802__) && __SYCL_TARGET_AMD_GPU_GFX802__)                               \
            || /* AMD GCN 4.0 Arctic Islands architecture (gfx 8.0) */                                                \
            (defined(__SYCL_TARGET_AMD_GPU_GFX803__) && __SYCL_TARGET_AMD_GPU_GFX803__)                               \
            || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.0) */                                              \
            (defined(__SYCL_TARGET_AMD_GPU_GFX805__) && __SYCL_TARGET_AMD_GPU_GFX805__)                               \
            || /* AMD GCN 3.0 Volcanic Islands architecture (gfx 8.1) */                                              \
            (defined(__SYCL_TARGET_AMD_GPU_GFX810__) && __SYCL_TARGET_AMD_GPU_GFX810__)                               \
            || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                                                          \
            (defined(__SYCL_TARGET_AMD_GPU_GFX900__) && __SYCL_TARGET_AMD_GPU_GFX900__)                               \
            || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                                                          \
            (defined(__SYCL_TARGET_AMD_GPU_GFX902__) && __SYCL_TARGET_AMD_GPU_GFX902__)                               \
            || /* AMD GCN 5.0 Vega architecture (gfx 9.0) */                                                          \
            (defined(__SYCL_TARGET_AMD_GPU_GFX904__) && __SYCL_TARGET_AMD_GPU_GFX904__)                               \
            || /* AMD GCN 5.1 Vega II architecture (gfx 9.0) */                                                       \
            (defined(__SYCL_TARGET_AMD_GPU_GFX906__) && __SYCL_TARGET_AMD_GPU_GFX906__)                               \
            || /* AMD CDNA 1.0 Arcturus architecture (gfx 9.0) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX908__) && __SYCL_TARGET_AMD_GPU_GFX908__)                               \
            || /* AMD GCN 5.0 Raven 2 architecture (gfx 9.0) */                                                       \
            (defined(__SYCL_TARGET_AMD_GPU_GFX909__) && __SYCL_TARGET_AMD_GPU_GFX909__)                               \
            || /* AMD CDNA 2.0 Aldebaran architecture (gfx 9.0) */                                                    \
            (defined(__SYCL_TARGET_AMD_GPU_GFX90A__) && __SYCL_TARGET_AMD_GPU_GFX90A__)                               \
            || /* AMD GCN 5.1 Renoir architecture (gfx 9.0) */                                                        \
            (defined(__SYCL_TARGET_AMD_GPU_GFX90C__) && __SYCL_TARGET_AMD_GPU_GFX90C__)                               \
            || /* AMD CDNA 3.x generic architecture (gfx 9.4) */                                                      \
            (defined(__SYCL_TARGET_AMD_GPU_GFX9_4_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX9_4_GENERIC__)               \
            || /* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */                                                \
            (defined(__SYCL_TARGET_AMD_GPU_GFX940__) && __SYCL_TARGET_AMD_GPU_GFX940__)                               \
            || /* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */                                                \
            (defined(__SYCL_TARGET_AMD_GPU_GFX941__) && __SYCL_TARGET_AMD_GPU_GFX941__)                               \
            || /* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */                                                \
            (defined(__SYCL_TARGET_AMD_GPU_GFX942__) && __SYCL_TARGET_AMD_GPU_GFX942__)                               \
            || /* AMD CDNA 4.0 derivative architecture (gfx 9.5) */                                                   \
            (defined(__SYCL_TARGET_AMD_GPU_GFX950__) && __SYCL_TARGET_AMD_GPU_GFX950__)                               \
            || /* AMD GCN 5.x generic architecture (gfx 9.x) */                                                       \
            (defined(__SYCL_TARGET_AMD_GPU_GFX9_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX9_GENERIC__)

#            define SYCL_SUBGROUP_SIZE (64) /* up to gfx9, HIP supports wavefront size 64 */

#        elif /* AMD RDNA 1.0 Navi 10 architecture (gfx 10.1) */                                                      \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1010__) && __SYCL_TARGET_AMD_GPU_GFX1010__)                             \
            || /* AMD RDNA 1.0 Navi 12 architecture (gfx 10.1) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1011__) && __SYCL_TARGET_AMD_GPU_GFX1011__)                             \
            || /* AMD RDNA 1.0 Navi 14 architecture (gfx 10.1) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1012__) && __SYCL_TARGET_AMD_GPU_GFX1012__)                             \
            || /* AMD RDNA 2.0 Oberon architecture (gfx 10.1) */                                                      \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1013__) && __SYCL_TARGET_AMD_GPU_GFX1013__)                             \
            || /* AMD RDNA 1.x generic architecture (gfx 10.1) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX10_1_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX10_1_GENERIC__)             \
            || /* AMD RDNA 2.0 Navi 21 architecture (gfx 10.3) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1030__) && __SYCL_TARGET_AMD_GPU_GFX1030__)                             \
            || /* AMD RDNA 2.0 Navi 22 architecture (gfx 10.3) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1031__) && __SYCL_TARGET_AMD_GPU_GFX1031__)                             \
            || /* AMD RDNA 2.0 Navi 23 architecture (gfx 10.3) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1032__) && __SYCL_TARGET_AMD_GPU_GFX1032__)                             \
            || /* AMD RDNA 2.0 Van Gogh architecture (gfx 10.3) */                                                    \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1033__) && __SYCL_TARGET_AMD_GPU_GFX1033__)                             \
            || /* AMD RDNA 2.0 Navi 24 architecture (gfx 10.3) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1034__) && __SYCL_TARGET_AMD_GPU_GFX1034__)                             \
            || /* AMD RDNA 2.0 Rembrandt Mobile architecture (gfx 10.3) */                                            \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1035__) && __SYCL_TARGET_AMD_GPU_GFX1035__)                             \
            || /* AMD RDNA 2.0 Raphael architecture (gfx 10.3) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1036__) && __SYCL_TARGET_AMD_GPU_GFX1036__)                             \
            || /* AMD RDNA 2.x generic architecture (gfx 10.3) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX10_3_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX10_3_GENERIC__)             \
            || /* AMD RDNA 3.0 Navi 31 architecture (gfx 11.0) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1100__) && __SYCL_TARGET_AMD_GPU_GFX1100__)                             \
            || /* AMD RDNA 3.0 Navi 32 architecture (gfx 11.0) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1101__) && __SYCL_TARGET_AMD_GPU_GFX1101__)                             \
            || /* AMD RDNA 3.0 Navi 33 architecture (gfx 11.0) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1102__) && __SYCL_TARGET_AMD_GPU_GFX1102__)                             \
            || /* AMD RDNA 3.0 Phoenix mobile architecture (gfx 11.0) */                                              \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1103__) && __SYCL_TARGET_AMD_GPU_GFX1103__)                             \
            || /* AMD RDNA 3.x generic architecture (gfx 11.x) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX11_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX11_GENERIC__)                 \
            || /* AMD RDNA 3.5 Strix Point architecture (gfx 11.5) */                                                 \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1150__) && __SYCL_TARGET_AMD_GPU_GFX1150__)                             \
            || /* AMD RDNA 3.5 Strix Halo architecture (gfx 11.5) */                                                  \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1151__) && __SYCL_TARGET_AMD_GPU_GFX1151__)                             \
            || /* AMD RDNA 3.5 Krackan Point architecture (gfx 11.5) */                                               \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1152__) && __SYCL_TARGET_AMD_GPU_GFX1152__)                             \
            || /* AMD RDNA 3.5 derivative architecture (gfx 11.5) */                                                  \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1153__) && __SYCL_TARGET_AMD_GPU_GFX1153__)                             \
            || /* AMD RDNA 3.5 derivative architecture (gfx 11.5) */                                                  \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1154__) && __SYCL_TARGET_AMD_GPU_GFX1154__)                             \
            || /* AMD RDNA 4m generic architecture (gfx 11.7) */                                                      \
            (defined(__SYCL_TARGET_AMD_GPU_GFX11_7_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX11_7_GENERIC__)             \
            || /* AMD RDNA 4m derivative architecture (gfx 11.7) */                                                   \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1170__) && __SYCL_TARGET_AMD_GPU_GFX1170__)                             \
            || /* AMD RDNA 4m derivative architecture (gfx 11.7) */                                                   \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1171__) && __SYCL_TARGET_AMD_GPU_GFX1171__)                             \
            || /* AMD RDNA 4m derivative architecture (gfx 11.7) */                                                   \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1172__) && __SYCL_TARGET_AMD_GPU_GFX1172__)                             \
            || /* AMD RDNA 4.0 Navi 44 architecture (gfx 12.0) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1200__) && __SYCL_TARGET_AMD_GPU_GFX1200__)                             \
            || /* AMD RDNA 4.0 Navi 48 architecture (gfx 12.0) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1201__) && __SYCL_TARGET_AMD_GPU_GFX1201__)                             \
            || /* AMD RDNA 4.x generic architecture (gfx 12.x) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX12_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX12_GENERIC__)                 \
            || /* AMD RDNA 4.5 generic architecture (gfx 12.5) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX12_5_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX12_5_GENERIC__)             \
            || /* AMD RDNA 4.5 derivative architecture (gfx 12.5) */                                                  \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1250__) && __SYCL_TARGET_AMD_GPU_GFX1250__)                             \
            || /* AMD RDNA 4.5 derivative architecture (gfx 12.5) */                                                  \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1251__) && __SYCL_TARGET_AMD_GPU_GFX1251__)                             \
            || /* AMD RDNA 5.0 generic architecture (gfx 13.x) */                                                     \
            (defined(__SYCL_TARGET_AMD_GPU_GFX13_GENERIC__) && __SYCL_TARGET_AMD_GPU_GFX13_GENERIC__)                 \
            || /* AMD RDNA 5.0 derivative architecture (gfx 13.1) */                                                  \
            (defined(__SYCL_TARGET_AMD_GPU_GFX1310__) && __SYCL_TARGET_AMD_GPU_GFX1310__)

#            define SYCL_SUBGROUP_SIZE (32) /* starting from gfx10, HIP supports wavefront size 32 */

#        else // __SYCL_TARGET_*

#            define SYCL_SUBGROUP_SIZE (0) /* unknown target */

#        endif // __SYCL_TARGET_*

#    else

#        define SYCL_SUBGROUP_SIZE (0) /* host compilation */

#    endif // __SYCL_DEVICE_ONLY__

#endif // ALPAKA_ACC_SYCL_ENABLED
