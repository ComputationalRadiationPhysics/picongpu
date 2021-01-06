/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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

#include "pmacc/dimensions/DataSpace.hpp"

#include "pmacc/traits/GetComponentsType.hpp"
#include "pmacc/traits/GetNComponents.hpp"
#include "pmacc/algorithms/math.hpp"
#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace traits
    {
        template<unsigned DIM>
        struct GetComponentsType<DataSpace<DIM>, false>
        {
            typedef typename DataSpace<DIM>::type type;
        };

        /** Trait for float_X */
        template<unsigned DIM>
        struct GetNComponents<DataSpace<DIM>, false>
        {
            static constexpr uint32_t value = DIM;
        };

    } // namespace traits

    namespace algorithms
    {
        namespace precisionCast
        {
            template<unsigned T_Dim>
            struct TypeCast<int, pmacc::DataSpace<T_Dim>>
            {
                typedef const pmacc::DataSpace<T_Dim>& result;

                HDINLINE result operator()(const pmacc::DataSpace<T_Dim>& vector) const
                {
                    return vector;
                }
            };

            template<typename T_CastToType, unsigned T_Dim>
            struct TypeCast<T_CastToType, pmacc::DataSpace<T_Dim>>
            {
                typedef ::pmacc::math::Vector<T_CastToType, T_Dim> result;

                HDINLINE result operator()(const pmacc::DataSpace<T_Dim>& vector) const
                {
                    return result(vector);
                }
            };

        } // namespace precisionCast
    } // namespace algorithms

} // namespace pmacc
