/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once


#include "picongpu/defines.hpp"
#include "picongpu/plugins/radiation/vector.hpp"

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            using vector_X = cuda_vec<picongpu::float3_X, picongpu::float_X>;
            using vector_32 = /*__align__(16)*/ cuda_vec<picongpu::float3_32, picongpu::float_32>;
            using vector_64 = /*__align__(32)*/ cuda_vec<picongpu::float3_64, picongpu::float_64>;
        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
