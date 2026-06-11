/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/particles/memory/dataTypes/Pointer.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/placeholders.hpp>

namespace pmacc
{
    template<typename T_Type = boost::mpl::_1>
    struct PreviousFramePtr
    {
        PMACC_ALIGN(previousFrame, Pointer<T_Type>);
    };

    template<typename T_Type = boost::mpl::_1>
    struct NextFramePtr
    {
        PMACC_ALIGN(nextFrame, Pointer<T_Type>);
    };

} // namespace pmacc
