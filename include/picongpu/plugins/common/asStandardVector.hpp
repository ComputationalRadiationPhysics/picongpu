/* Copyright 2014-2020 Pawel Ordyna
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

#include <pmacc/math/Vector.hpp>

#include <type_traits>
#include <vector>

namespace picongpu
{
    namespace openPMD
    {
        template<typename T_Vec, typename T_Ret = std::vector<typename std::remove_reference<T_Vec>::type::type>>
        T_Ret asStandardVector(T_Vec const& v)
        {
            using __T_Vec = typename std::remove_reference<T_Vec>::type;
            constexpr auto dim = __T_Vec::dim;
            T_Ret res(dim);
            for(unsigned i = 0; i < dim; ++i)
            {
                res[dim - i - 1] = v[i];
            }
            return res;
        }
    } // namespace openPMD
} // namespace picongpu
