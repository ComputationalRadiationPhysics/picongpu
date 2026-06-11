/*
 * SPDX-FileCopyrightText: Marco Garten, Rene Widera, Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu::traits
{
    /** get atomicNumbers (number of protons and neutrons) flag from species
     *
     * @tparam T_Species particle type or resolved species type
     *
     * @return struct with two static constexpr members numberOfProtons:float_X and numberOfNeutrons:float_X,
     *  stored in member type of this struct
     */
    template<typename T_Species>
    struct GetAtomicNumbers
    {
        using FrameType = typename T_Species::FrameType;

        using hasAtomicNumbers = typename HasFlag<FrameType, atomicNumbers<>>::type;
        /* throw static assert if species lacks flag*/
        PMACC_CASSERT_MSG(This_species_has_no_atomic_numbers, hasAtomicNumbers::value == true);

        using FoundAtomicNumbersAlias = typename pmacc::traits::GetFlagType<FrameType, atomicNumbers<>>::type;
        using type = typename pmacc::traits::Resolve<FoundAtomicNumbersAlias>::type;
    };
} // namespace picongpu::traits
