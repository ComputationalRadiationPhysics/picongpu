/*
 * SPDX-FileCopyrightText: Rene Widera, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/meta/conversion/RemoveFromSeq.hpp"
#include "pmacc/meta/conversion/ResolveAliases.hpp"
#include "pmacc/meta/errorHandlerPolicies/ReturnValue.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** Resolve and remove types from a sequence
     *
     * @tparam T_MPLSeqSrc source sequence from were we delete types
     * @tparam T_MPLSeqObjectsToRemove sequence with types which should be deleted (pmacc aliases are allowed)
     */
    template<typename T_MPLSeqSrc, typename T_MPLSeqObjectsToRemove>
    struct ResolveAndRemoveFromSeq
    {
        using MPLSeqSrc = T_MPLSeqSrc;
        using MPLSeqObjectsToRemove = T_MPLSeqObjectsToRemove;
        using ResolvedSeqWithObjectsToRemove =
            typename ResolveAliases<MPLSeqObjectsToRemove, MPLSeqSrc, errorHandlerPolicies::ReturnValue>::type;
        using type = typename RemoveFromSeq<MPLSeqSrc, ResolvedSeqWithObjectsToRemove>::type;
    };

} // namespace pmacc
