/* Copyright 2014-2021 Rene Widera, Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"

#include "pmacc/meta/conversion/RemoveFromSeq.hpp"
#include "pmacc/meta/conversion/ResolveAliases.hpp"
#include "pmacc/meta/errorHandlerPolicies/ReturnValue.hpp"

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
        typedef T_MPLSeqSrc MPLSeqSrc;
        typedef T_MPLSeqObjectsToRemove MPLSeqObjectsToRemove;
        typedef typename ResolveAliases<MPLSeqObjectsToRemove, MPLSeqSrc, errorHandlerPolicies::ReturnValue>::type
            ResolvedSeqWithObjectsToRemove;
        typedef typename RemoveFromSeq<MPLSeqSrc, ResolvedSeqWithObjectsToRemove>::type type;
    };

} // namespace pmacc
