/* Copyright 2016-2021 Axel Huebl
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

namespace picongpu
{
    namespace traits
    {
        /** Forward names that are identical in PIConGPU & openPMD
         */
        template<typename T_Identifier>
        struct OpenPMDName
        {
            std::string operator()() const
            {
                return T_Identifier::getName();
            }
        };

        /** Translate the totalCellIdx (unitless index) into the openPMD
         *  positionOffset (3D position vector, length)
         */
        template<>
        struct OpenPMDName<totalCellIdx>
        {
            std::string operator()() const
            {
                return std::string("positionOffset");
            }
        };

        /** Translate the particleId (unitless, global) into the openPMD
         *  id (unitless, global)
         */
        template<>
        struct OpenPMDName<particleId>
        {
            std::string operator()() const
            {
                return std::string("id");
            }
        };

        /** Forward units that are identical in PIConGPU & openPMD
         */
        template<typename T_Identifier>
        struct OpenPMDUnit
        {
            std::vector<double> operator()() const
            {
                return Unit<T_Identifier>::get();
            }
        };

        /** the totalCellIdx can be converted into a positionOffset
         *  until the beginning of the cell by multiplying with the component-wise
         *  cell size in SI
         */
        template<>
        struct OpenPMDUnit<totalCellIdx>
        {
            std::vector<double> operator()() const
            {
                std::vector<double> unit(simDim);
                /* cell positionOffset needs two transformations to get to SI:
                   cell begin -> dimensionless scaling to grid -> SI */
                for(uint32_t i = 0; i < simDim; ++i)
                    unit[i] = cellSize[i] * UNIT_LENGTH;

                return unit;
            }
        };

        /** Forward dimensionalities that are identical in PIConGPU & openPMD
         */
        template<typename T_Identifier>
        struct OpenPMDUnitDimension
        {
            std::vector<float_64> operator()() const
            {
                return UnitDimension<T_Identifier>::get();
            }
        };

        /** the openPMD positionOffset is an actual (vector) with a lengths that
         *  is added to the position (vector) attribute
         */
        template<>
        struct OpenPMDUnitDimension<totalCellIdx>
        {
            std::vector<float_64> operator()() const
            {
                /* L, M, T, I, theta, N, J
                 *
                 * positionOffset is in meter: m
                 *   -> L
                 */
                std::vector<float_64> unitDimension(NUnitDimension, 0.0);
                unitDimension.at(SIBaseUnits::length) = 1.0;

                return unitDimension;
            }
        };

    } // namespace traits
} // namespace picongpu
