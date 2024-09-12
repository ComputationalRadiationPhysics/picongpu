/* Copyright 2020-2023 Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/Unique.hpp>

#include <cstdint>
#include <type_traits>


namespace picongpu::fields::incidentField
{
    //! Typelist of all enabled profiles, can contain duplicates
    using EnabledProfiles = pmacc::MakeSeq_t<
        XMin,
        XMax,
        YMin,
        YMax,
        std::conditional_t<simDim == 3, pmacc::MakeSeq_t<ZMin, ZMax>, pmacc::MakeSeq_t<>>>;

    //! Typelist of all unique enabled profiles, can contain duplicates
    using UniqueEnabledProfiles = pmacc::Unique_t<EnabledProfiles>;
} // namespace picongpu::fields::incidentField
