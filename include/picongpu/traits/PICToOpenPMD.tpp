/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/param/speciesAttributes.param"
#include "picongpu/traits/SIBaseUnits.hpp"
#include "picongpu/traits/Unit.hpp"
#include "picongpu/traits/UnitDimension.hpp"

#include <string>
#include <vector>

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
                    unit[i] = sim.pic.getCellSize()[i] * sim.unit.length();

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
