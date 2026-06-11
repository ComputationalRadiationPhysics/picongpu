/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace pmacc
{
    namespace traits
    {
        /** Get component type of an object
         *
         * @tparam T_Type any type
         * @return \p ::type get result type
         *            If T_Type is fundamental c++ type, the identity is returned
         *
         * Attention: do not defines this trait for structs with different attributes inside
         *
         * Intentionally not using std::is_fundamental_v<T_Type> due to a compiler bug in cuda-13/gcc-11 combination,
         * ref. https://github.com/ComputationalRadiationPhysics/picongpu/issues/5554
         */
        template<typename T_Type, bool T_IsFundamental = std::is_fundamental<T_Type>::value>
        struct GetComponentsType;

        template<typename T_Type>
        struct GetComponentsType<T_Type, true>
        {
            using type = T_Type;
        };

    } // namespace traits

} // namespace pmacc
