/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/traits/GetComponentsType.hpp>

#include <type_traits>

namespace pmacc
{
    namespace traits
    {
        /** Get component type trait for bools in openPMD output
         *
         * Specializes the general trait in pmacc/traits/GetComponentsType.hpp.
         * For use with the openPMD API, both files must be included.
         *
         * The reason is that ADIOS2 backend of openPMD API currently does not support bool datasets #3732.
         * So with this specialization, PIConGPU particle attributes of type bool (e.g. radiationMask,
         * transitionRadiationMask) are treated as chars.
         *
         * This requires sizeof(bool) == sizeof(char), ::type is defined only in this case.
         */
        template<>
        struct GetComponentsType<bool>
        {
            using type = std::enable_if_t<sizeof(bool) == sizeof(char), char>;
        };

    } // namespace traits
} // namespace pmacc
