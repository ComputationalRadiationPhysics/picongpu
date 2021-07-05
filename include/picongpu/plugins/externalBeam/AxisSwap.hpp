/* Copyright 2020-2021 Pawel Ordyna
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

namespace picongpu
{
    namespace plugins
    {
        namespace externalBeam
        {
            struct AxisSwap
            {
            protected:
                uint8_t axis[3];
                int8_t a_m[3];
                PMACC_ALIGN(globalDomainSize, DataSpace<DIM3>);

            public:
                HINLINE AxisSwap(
                    uint8_t const& axis0,
                    uint8_t const& axis1,
                    uint8_t const& axis2,
                    bool const& reverse0,
                    bool const& reverse1,
                    bool const& reverse2)
                    : axis{axis0, axis1, axis2}
                    , a_m{reverse0 ? int8_t(-1) : int8_t(1),
                          reverse1 ? int8_t(-1) : int8_t(1),
                          reverse2 ? int8_t(-1) : int8_t(1)}
                {
                    switch(simDim)
                    {
                    case DIM2:
                    {
                        DataSpace<simDim> globalDomainSizeTmp
                            = Environment<simDim>::get().SubGrid().getGlobalDomain().size;
                        globalDomainSize = {globalDomainSizeTmp[0], globalDomainSizeTmp[1], 1};
                        break;
                    }
                    case DIM3:
                        globalDomainSize = Environment<simDim>::get().SubGrid().getGlobalDomain().size;
                        break;
                    }
                }

                //! Performs the axis swap and reverse on a float3_X object.
                HDINLINE float3_X rotate(float3_X const& vec, bool const& swapOnly = false) const
                {
                    constexpr int8_t aNoReverse[3] = {1, 1, 1};
                    int8_t const(&a)[3] = swapOnly ? aNoReverse : a_m;
                    return float3_X(a[0] * vec[axis[0]], a[1] * vec[axis[1]], a[2] * vec[axis[2]]);
                }

                //! Performs the axis swap and reverse on a DataSpace object.
                HDINLINE DataSpace<DIM3> rotate(DataSpace<DIM3> const& vec, bool const& swapOnly = false) const
                {
                    constexpr int8_t aNoReverse[3] = {1, 1, 1};
                    int8_t const(&a)[3] = swapOnly ? aNoReverse : a_m;
                    return DataSpace<DIM3>{a[0] * vec[axis[0]], a[1] * vec[axis[1]], a[2] * vec[axis[2]]};
                }
                //! Performs the reversed operation (back rotation) on a float3_X.
                HDINLINE float3_X reverse(float3_X const& vec, bool const& swapOnly = false) const
                {
                    constexpr int8_t aNoReverse[3] = {1, 1, 1};
                    int8_t const(&a)[3] = swapOnly ? aNoReverse : a_m;
                    float3_X result;
                    result[axis[0]] = vec[0] / a[0];
                    result[axis[1]] = vec[1] / a[1];
                    result[axis[2]] = vec[2] / a[2];
                    return result;
                }

                //! Performs the reversed operation (back rotation) on a DataSpace vector.
                HDINLINE DataSpace<DIM3> reverse(DataSpace<DIM3> const& vec, bool const& swapOnly = false) const
                {
                    constexpr int8_t aNoReverse[3] = {1, 1, 1};
                    int8_t const(&a)[3] = swapOnly ? aNoReverse : a_m;
                    DataSpace<DIM3> result;
                    result[axis[0]] = vec[0] / a[0];
                    result[axis[1]] = vec[1] / a[1];
                    result[axis[2]] = vec[2] / a[2];
                    return result;
                }

                //! transform cell coordinates (global domain)
                HDINLINE DataSpace<DIM3> transformCellIdx(
                    DataSpace<simDim> const& cellIdx,
                    bool const& swapOnly = false) const
                {
                    constexpr int8_t aNoReverse[3] = {1, 1, 1};
                    int8_t const(&a)[3] = swapOnly ? aNoReverse : a_m;
                    DataSpace<DIM3> cellIdx3d;
                    switch(simDim)
                    {
                    case DIM2:
                        cellIdx3d = {cellIdx[0], cellIdx[1], 1};
                        break;
                    case DIM3:
                        cellIdx3d = cellIdx;
                        break;
                    }
                    DataSpace<DIM3> result;
                    result[0] = a[0] >= 0 ? cellIdx3d[axis[0]] : (globalDomainSize[axis[0]] - 1) - cellIdx3d[axis[0]];
                    result[1] = a[1] >= 0 ? cellIdx3d[axis[1]] : (globalDomainSize[axis[1]] - 1) - cellIdx3d[axis[1]];
                    result[2] = a[2] >= 0 ? cellIdx3d[axis[2]] : (globalDomainSize[axis[2]] - 1) - cellIdx3d[axis[2]];
                    return result;
                }
            };
        } // namespace externalBeam
    } // namespace plugins
} // namespace picongpu
