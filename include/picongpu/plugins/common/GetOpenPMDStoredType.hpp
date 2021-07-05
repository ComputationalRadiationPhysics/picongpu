/* Copyright 2021 Pawel Ordyna
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <complex>
namespace picongpu
{
    namespace openPMD
    {
        template<typename PIConGPUType>
        struct GetOpenPMDStoredType
        {
            using type = PIConGPUType;
        };

        template<>
        struct GetOpenPMDStoredType<pmacc::math::Complex<picongpu::float_32>>
        {
            using type = std::complex<float_32>;
        };
        template<>
        struct GetOpenPMDStoredType<pmacc::math::Complex<picongpu::float_64>>
        {
            using type = std::complex<float_64>;
        };
    } // namespace openPMD
} // namespace picongpu
