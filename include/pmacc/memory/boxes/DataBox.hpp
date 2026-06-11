/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/dimensions/DataSpace.hpp"

namespace pmacc
{
    template<typename T_Base>
    struct DataBox : T_Base
    {
        using Base = T_Base;
        using typename Base::RefValueType;
        using typename Base::ValueType;

        DataBox() = default;

        HDINLINE DataBox(Base base) : Base{std::move(base)}
        {
        }

        DataBox(DataBox const&) = default;

        HDINLINE decltype(auto) operator()(DataSpace<Base::Dim> const& idx = {}) const
        {
            return Base::operator[](idx);
        }

        HDINLINE decltype(auto) operator()(DataSpace<Base::Dim> const& idx = {})
        {
            return Base::operator[](idx);
        }

        HDINLINE DataBox shift(DataSpace<Base::Dim> const& offset) const
        {
            DataBox result(*this);
            result.m_ptr = const_cast<typename Base::ValueType*>(&((*this)(offset)));
            return result;
        }
    };
} // namespace pmacc
