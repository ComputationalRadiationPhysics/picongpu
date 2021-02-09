/* Copyright 2013-2021 Rene Widera
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

#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/remove_if.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/contains.hpp>

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
        struct hasId
        {
            typedef bmpl::contains<T_MPLSeqObjectsToRemove, T_Value> type;
        };

        typedef typename bmpl::remove_if<T_MPLSeqSrc, hasId<bmpl::_>>::type type;
    };

} // namespace pmacc
