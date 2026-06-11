/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/meta/Pair.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** Wrapper to use any type as identifier
     *
     * Wrap a type thus we can call default constructor on every class
     * This is needed to support that any type can used as identifier in for math::MapTuple
     */
    template<typename T_Type>
    struct TypeAsIdentifier
    {
        using type = T_Type;
    };

    /** Unary functor to wrap any type with TypeAsIdentifier
     *
     * @tparam T_Type to to wrap
     */
    template<typename T_Type>
    struct MakeIdentifier
    {
        using type = TypeAsIdentifier<T_Type>;
    };

    /** Pass through of an already existing Identifier
     *
     * Avoids double-wrapping of an Identifier
     */
    template<typename T_Type>
    struct MakeIdentifier<TypeAsIdentifier<T_Type>>
    {
        using type = TypeAsIdentifier<T_Type>;
    };

    /** create pmacc::meta::Pair<TypeAsIdentifier<Type>,PointerOfType>
     *
     * @tparam T_Type any type
     * @return ::type pmacc::meta::Pair<TypeAsIdentifier<Type>,PointerOfType>
     */
    template<typename T_Type>
    struct TypeToPointerPair
    {
        using TypePtr = T_Type*;
        using type = pmacc::meta::Pair<typename MakeIdentifier<T_Type>::type, TypePtr>;
    };

} // namespace pmacc
