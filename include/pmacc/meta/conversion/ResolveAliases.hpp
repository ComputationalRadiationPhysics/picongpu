/* Copyright 2013-2023 Rene Widera, Felix Schmitt, Alexander Grund
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/meta/GetKeyFromAlias.hpp"
#include "pmacc/meta/Mp11.hpp"
#include "pmacc/meta/errorHandlerPolicies/ThrowValueNotFound.hpp"
#include "pmacc/types.hpp"

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
        using MPLSeq = T_MPLSeq;
        using MPLSeqLookup = T_MPLSeqLookup;
        using AliasNotFoundPolicy = T_AliasNotFoundPolicy;

        template<typename T_Identifier>
        using GetKeyFromAliasAccessor =
            typename GetKeyFromAlias<MPLSeqLookup, T_Identifier, AliasNotFoundPolicy>::type;

        using type = mp_transform<GetKeyFromAliasAccessor, MPLSeq>;
    };

} // namespace pmacc
