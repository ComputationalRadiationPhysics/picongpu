/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/param.hpp"

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
