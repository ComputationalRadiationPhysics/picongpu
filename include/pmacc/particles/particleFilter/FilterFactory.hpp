/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/meta/conversion/MakeSeq.hpp"
#include "pmacc/particles/boostExtension/InheritGenerators.hpp"
#include "pmacc/particles/memory/frames/NullFrame.hpp"
#include "pmacc/particles/particleFilter/system/TrueFilter.hpp"

namespace pmacc
{
    template<typename UserTypeList = mp_list<NullFrame>>
    class FilterFactory
    {
    public:
        using FilterType = typename LinearInherit<MakeSeq_t<UserTypeList, TrueFilter>>::type;
    };
} // namespace pmacc
