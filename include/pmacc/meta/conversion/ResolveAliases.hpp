/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
