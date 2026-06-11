/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc
{
    /** DataBox which apply a unary functor on every operator () and [] access
     *
     * @tparam T_Base base class to inherit from
     * @tparam T_UnaryFunctor unary functor which is applied on every access
     *         - template parameter of functor is the input type for the functor
     *         - functor must have defined the result type as ::result
     */
    template<typename T_Base, template<typename> class T_UnaryFunctor>
    class DataBoxUnaryTransform : public T_Base
    {
    public:
        using Base = T_Base;
        using UnaryFunctor = T_UnaryFunctor<typename Base::ValueType>;
        using ValueType = typename UnaryFunctor::result;

        static constexpr std::uint32_t Dim = Base::Dim;

        HDINLINE DataBoxUnaryTransform() = default;

        HDINLINE DataBoxUnaryTransform(Base base) : Base(std::move(base))
        {
        }

        HDINLINE DataBoxUnaryTransform(DataBoxUnaryTransform const&) = default;

        template<typename T_Index>
        HDINLINE ValueType operator()(T_Index const& idx) const
        {
            return UnaryFunctor()(Base::operator[](idx));
        }

        template<typename T_Index>
        HDINLINE ValueType operator[](T_Index const idx) const
        {
            return UnaryFunctor()(Base::operator[](idx));
        }
    };
} // namespace pmacc
