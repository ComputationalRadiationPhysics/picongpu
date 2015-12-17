/**
 * Copyright 2015 Heiko Burau
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "simulation_types.hpp"
#include "NumericalCellTypes.hpp"

namespace picongpu
{

/** Get the numerical field position of a given numerical
 * cell type by field type index.
 *
 * \tparam NumericalCellType \see: YeeCell.hpp or EMFCenteredCell.hpp
 * \tparam fieldType \see: FieldType in simulation_types.hpp
 */
template<typename NumericalCellType, int fieldType>
struct GetNumericalFieldPos;

template<typename NumericalCellType>
struct GetNumericalFieldPos<NumericalCellType, FIELD_TYPE_E>
{
    typedef typename NumericalCellType::VectorVector VectorVector;

    HDINLINE VectorVector operator()() const
    {
        return NumericalCellType::getEFieldPosition();
    }
};

template<typename NumericalCellType>
struct GetNumericalFieldPos<NumericalCellType, FIELD_TYPE_B>
{
    typedef typename NumericalCellType::VectorVector VectorVector;

    HDINLINE VectorVector operator()() const
    {
        return NumericalCellType::getBFieldPosition();
    }
};

} // namespace picongpu
