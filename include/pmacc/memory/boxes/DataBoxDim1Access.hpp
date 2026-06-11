/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
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

        HDINLINE decltype(auto) operator[](int const idx) const
        {
            return Base::operator[](pmacc::math::mapToND(originalSize, idx));
        }

        HDINLINE decltype(auto) operator[](int const idx)
        {
            return Base::operator[](pmacc::math::mapToND(originalSize, idx));
        }

    private:
        PMACC_ALIGN(originalSize, DataSpace<Dim> const);
    };
} // namespace pmacc
