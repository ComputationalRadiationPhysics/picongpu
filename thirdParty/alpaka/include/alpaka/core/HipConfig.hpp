/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"

// We can not use ALPAKA_LANG_HIP because this file is required by core/config.hpp where ALPAKA_LANG_HIP is defined.
#if defined(__HIP__)

#    include <hip/hip_version.h>

// version numbers are only defined on the device side
#    if !defined(ALPAKA_AMDGPU_ARCH) && defined(__HIP__) && __HIP_DEVICE_COMPILE__ == 1

/* Map AMDGPU arch macro -> ALPAKA_VRRPP_TO_VERSION(wrapped code)
 *  Rules:
 *   - gfx9xy (numeric): 9xy -> 90x0y  (e.g., 908->90008, 906->90006, 942->90402)
 *   - gfx10xy / gfx11xy: stxy -> st0x0y (e.g., 1036->100306, 1103->110003)
 *   - Suffix: a == 10 (90a->90010), b == 11, c == 11
 *
 * An overview of AMD GPU architectures can be found here:
 * https://llvm.org/docs/AMDGPUUsage.html#processors
 */

#        if defined(__gfx1200__)
/* RDNA 4 dGPU (RX 9060 XT) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(120000)
#        elif defined(__gfx1201__)
/* RDNA 4 dGPU (RX 9070 / 9070 XT) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(120001)
#        elif defined(__gfx1250__)
/* RDNA 4 APU (APU) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(120500)
#        elif defined(__gfx1251__)
/* RDNA 4 APU variant */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(120501)

#        elif defined(__gfx1153__)
/* RDNA 3.5 iGPU (Medusa Point / Strix Halo successor) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(110503)
#        elif defined(__gfx1152__)
/* RDNA 3.5 iGPU (Krackan Point) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(110502)
#        elif defined(__gfx1151__)
/* RDNA 3.5 iGPU (Strix Halo) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(110501)
#        elif defined(__gfx1150__)
/* RDNA 3.5 iGPU (Radeon 890M on Strix Point) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(110500)

#        elif defined(__gfx1103__)
/* RDNA 3 APU (Radeon 780M, 760M, ROG Ally Extreme) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(110003)
#        elif defined(__gfx1102__)
/* RDNA 3 Desktop (RX 7600 / 7600 XT) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(110002)
#        elif defined(__gfx1101__)
/* RDNA 3 Desktop (RX 7700 / 7700 XT, Pro W7700 / V710) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(110001)
#        elif defined(__gfx1100__)
/* RDNA 3 Desktop (RX 7900 XT, XTX, Pro W7900) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(110000)

#        elif defined(__gfx1036__)
/* RDNA 2 APU (Radeon Graphics 128-SP iGPU) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100306)
#        elif defined(__gfx1035__)
/* RDNA 2 APU (Radeon 660M, 680M) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100305)
#        elif defined(__gfx1034__)
/* RDNA 2 Mobile (Pro W6300/W6400, RX 6400-6500) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100304)
#        elif defined(__gfx1033__)
/* RDNA 2 APU (Steam Deck) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100303)
#        elif defined(__gfx1032__)
/* RDNA 2 Desktop (RX 6600 XT, 6650 XT/S, 6700S) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100302)
#        elif defined(__gfx1031__)
/* RDNA 2 Desktop (RX 6700 series, 6750/6850M XT) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100301)
#        elif defined(__gfx1030__)
/* RDNA 2 Desktop (RX 6800 / 6900 XT, Pro W6800) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100300)

#        elif defined(__gfx1013__)
/* RDNA 1 Mobile (RX 5300M / 5500M) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100103)
#        elif defined(__gfx1012__)
/* RDNA 1 Desktop (RX 5500 / 5500 XT) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100102)
#        elif defined(__gfx1011__)
/* RDNA 1 Desktop (Pro V520) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100101)
#        elif defined(__gfx1010__)
/* RDNA 1 Desktop (RX 5700 / 5700 XT, Pro 5600 XT/M) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(100100)

#        elif defined(__gfx942__)
/* CDNA 3 (Instinct MI300 series: MI300/MI300A/MI300X) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(90402)
#        elif defined(__gfx941__)
/* CDNA 2/3 (Instinct MI210) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(90401)
#        elif defined(__gfx940__)
/* CDNA 2 (Instinct MI200) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(90400)

#        elif defined(__gfx90c__)
/* CDNA 1 (Renoir APUs), c -> 12 */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(90012)
#        elif defined(__gfx90b__)
/* (If present) b -> 11 */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(90011)
#        elif defined(__gfx90a__)
/* CDNA 2 (Instinct MI250 / MI250X), a -> 10 */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(90010)
#        elif defined(__gfx908__)
/* CDNA 1 (Instinct MI100) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(90008)
#        elif defined(__gfx906__)
/* Vega 20 (Radeon VII, Instinct MI50/60) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRPP_TO_VERSION(90006)

#        else
#            error                                                                                                    \
                "Unknown AMDGPU architecture, please define __gfxXXX__ macro for your target. Until alpaka is updated you can define the macro ALPAKA_AMDGPU_ARCH to avoid this error."
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#        endif

#    endif
#endif
