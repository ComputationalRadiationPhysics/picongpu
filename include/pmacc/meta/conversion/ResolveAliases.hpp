/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Alexander Grund
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
#include "pmacc/meta/GetKeyFromAlias.hpp"
#include "pmacc/meta/errorHandlerPolicies/ThrowValueNotFound.hpp"

#include <boost/mpl/vector.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/insert.hpp>

namespace pmacc
{
    /** Translate all pmacc alias types to full specialized types
     *
     * Use lookup sequence to translate types
     * The policy is used if the type from T_MPLSeq is not in T_MPLSeqLookup a compile time error is triggered
     *
     * @tparam T_MPLSeq source sequence with types to translate
     * @tparam T_MPLSeqLookup lookup sequence to translate aliases
     */
    template<
        typename T_MPLSeq,
        typename T_MPLSeqLookup,
        typename T_AliasNotFoundPolicy = errorHandlerPolicies::ThrowValueNotFound>
    struct ResolveAliases
    {
        typedef T_MPLSeq MPLSeq;
        typedef T_MPLSeqLookup MPLSeqLookup;
        typedef T_AliasNotFoundPolicy AliasNotFoundPolicy;
        typedef bmpl::back_inserter<bmpl::vector<>> Inserter;

        template<typename T_Identifier>
        struct GetKeyFromAliasAccessor
        {
            typedef typename GetKeyFromAlias<MPLSeqLookup, T_Identifier, AliasNotFoundPolicy>::type type;
        };

        typedef typename bmpl::transform<MPLSeq, GetKeyFromAliasAccessor<bmpl::_1>>::type type;
    };

} // namespace pmacc
