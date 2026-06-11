/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/algorithms/math.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/traits/GetComponentsType.hpp"
#include "pmacc/traits/GetNComponents.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace traits
    {
        template<unsigned T_dim>
        struct GetComponentsType<DataSpace<T_dim>, false>
        {
            using type = typename DataSpace<T_dim>::type;
        };

        /** Trait for float_X */
        template<unsigned T_dim>
        struct GetNComponents<DataSpace<T_dim>, false>
        {
            static constexpr uint32_t value = T_dim;
        };

    } // namespace traits

    namespace algorithms
    {
        namespace precisionCast
        {
            template<unsigned T_Dim>
            struct TypeCast<int, pmacc::DataSpace<T_Dim>>
            {
                using result = pmacc::DataSpace<T_Dim> const;

                constexpr result operator()(pmacc::DataSpace<T_Dim> const& vector) const
                {
                    return vector;
                }
            };

            template<typename T_CastToType, unsigned T_Dim>
            struct TypeCast<T_CastToType, pmacc::DataSpace<T_Dim>>
            {
                using result = ::pmacc::math::Vector<T_CastToType, T_Dim>;

                constexpr result operator()(pmacc::DataSpace<T_Dim> const& vector) const
                {
                    return result(vector);
                }
            };

        } // namespace precisionCast
    } // namespace algorithms

} // namespace pmacc
