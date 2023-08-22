/* Copyright 2023 Benjamin Worpitz, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

ALPAKA_FN_HOST_ACC ALPAKA_FN_EXTERN auto mysqrt(float x) noexcept -> float;
