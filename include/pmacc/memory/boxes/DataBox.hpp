/* Copyright 2013-2022 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz
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

#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/dimensions/DataSpace.hpp"

namespace pmacc
{
    namespace detail
    {
        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<1> const& idx = {})
        {
            return db[idx.x()];
        }

        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<2> const& idx = {})
        {
            return db[idx.y()][idx.x()];
        }

        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<3> const& idx = {})
        {
            return db[idx.z()][idx.y()][idx.x()];
        }
    } // namespace detail

    template<typename Base>
    struct DataBox : Base
    {
        HDINLINE DataBox() = default;

        HDINLINE DataBox(Base base) : Base{std::move(base)}
        {
        }

        HDINLINE DataBox(DataBox const&) = default;

        HDINLINE decltype(auto) operator()(DataSpace<Base::Dim> const& idx = {}) const
        {
            ///@todo(bgruber): inline and replace this by if constexpr in C++17
            return detail::access(*this, idx);
        }

        HDINLINE DataBox shift(DataSpace<Base::Dim> const& offset) const
        {
            DataBox result(*this);
            result.fixedPointer = &((*this)(offset));
            return result;
        }
    };
} // namespace pmacc
