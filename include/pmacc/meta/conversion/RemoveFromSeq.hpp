/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    /* remove types from a sequence
     *
     * @tparam T_MPLSeqSrc source sequence from were we delete types
     * @tparam T_MPLSeqObjectsToRemove sequence with types which shuld be deleted
     */
    template<typename T_MPLSeqSrc, typename T_MPLSeqObjectsToRemove>
    struct RemoveFromSeq
    {
        template<typename T_Value>
        using hasId = boost::mp11::mp_contains<T_MPLSeqObjectsToRemove, T_Value>; // FIXME(bgruber): boost::mp11::
                                                                                  // needed for nvcc 11.0

        using type = mp_remove_if<T_MPLSeqSrc, hasId>;
    };

} // namespace pmacc
