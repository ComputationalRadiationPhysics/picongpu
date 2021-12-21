/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "Memcopy.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace copier
    {
        template<int T_dim>
        struct H2HCopier
        {
            static constexpr int dim = T_dim;

            PMACC_NO_NVCC_HDWARNING /* Should never be called from device functions */
                template<typename Type>
                HDINLINE static void copy(
                    Type* dest,
                    const math::Size_t<dim - 1>& pitchDest,
                    Type* source,
                    const math::Size_t<dim - 1>& pitchSource,
                    const math::Size_t<dim>& size)
            {
                cuplaWrapper::Memcopy<dim>()(
                    dest,
                    pitchDest,
                    source,
                    pitchSource,
                    size,
                    cuplaWrapper::flags::Memcopy::hostToHost);
            }
        };

    } // namespace copier
} // namespace pmacc
