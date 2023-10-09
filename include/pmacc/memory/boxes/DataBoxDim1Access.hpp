/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"
#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc
{
    template<typename T_Base>
    struct DataBoxDim1Access : protected T_Base
    {
        using Base = T_Base;
        static constexpr std::uint32_t Dim = Base::Dim;
        using ValueType = typename Base::ValueType;

        HDINLINE DataBoxDim1Access(DataSpace<Dim> const& originalSize) : Base(), originalSize(originalSize)
        {
        }

        HDINLINE DataBoxDim1Access(Base base, DataSpace<Dim> const& originalSize)
            : Base(std::move(base))
            , originalSize(originalSize)
        {
        }

        DataBoxDim1Access(DataBoxDim1Access const&) = default;

        HDINLINE decltype(auto) operator()(DataSpace<DIM1> const& idx = {}) const
        {
            return (*this)[idx.x()];
        }

        HDINLINE decltype(auto) operator()(DataSpace<DIM1> const& idx = {})
        {
            return (*this)[idx.x()];
        }

        HDINLINE decltype(auto) operator[](const int idx) const
        {
            return Base::operator[](DataSpaceOperations<Dim>::map(originalSize, idx));
        }

        HDINLINE decltype(auto) operator[](const int idx)
        {
            return Base::operator[](DataSpaceOperations<Dim>::map(originalSize, idx));
        }

    private:
        PMACC_ALIGN(originalSize, const DataSpace<Dim>);
    };
} // namespace pmacc
