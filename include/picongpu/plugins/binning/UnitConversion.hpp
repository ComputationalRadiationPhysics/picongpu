/* Copyright 2023 Tapish Narwal
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <array>
#include <map>

#include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace plugins::binning
    {
        constexpr unsigned numUnits = 7;
        // @todo add where this 7D format is from
        std::array<double, numUnits> UnitDimensions{
            sim.unit.length(), // length
            sim.unit.mass(), // mass
            sim.unit.time(), // time
            sim.unit.charge() / sim.unit.time(), // current
            1., // thermodynamicTemperature
            1., // amountOfSubstance add sim.si.getNAvogadro() HERE? FROM physicalConstants.param
            1., // luminousIntensity
            // 1. // add weighting?
        };

        /**
         * In this format the conversion factor needs to be divided(?)
         * is it faster/better to calculate the inverse and then multiply?
         */
        HINLINE double getConversionFactor(const std::array<double, numUnits>& myDimension)
        {
            double conversion_factor = 1.;
            for(size_t i = 0; i < 7; i++)
            {
                conversion_factor *= std::pow(UnitDimensions[i], myDimension[i]);
            };
            return conversion_factor;
        }

        template<typename T>
        HINLINE T toPICUnits(T varSI, const std::array<double, numUnits>& myDimension)
        {
            if constexpr(std::is_integral_v<T>)
            {
                for(auto&& dim : myDimension)
                {
                    PMACC_VERIFY(dim == 0.0);
                }
            };
            return static_cast<T>(static_cast<double>(varSI) / getConversionFactor(myDimension));
        }

        template<typename T>
        HINLINE double toSIUnits(T varPIC, const std::array<double, numUnits>& myDimension)
        {
            return static_cast<double>(varPIC) * getConversionFactor(myDimension);
        }


        HINLINE std::map<::openPMD::UnitDimension, double> makeOpenPMDUnitMap(
            const std::array<double, numUnits>& myDimension)
        {
            using UD = ::openPMD::UnitDimension;

            std::array<UD, numUnits> keys = {UD::L, UD::M, UD::T, UD::I, UD::theta, UD::N, UD::J};

            std::map<UD, double> map;

            // Combine the two arrays into the map
            for(size_t i = 0; i < keys.size(); ++i)
            {
                map[keys[i]] = myDimension[i];
            }
            return map;
        }
    } // namespace plugins::binning
} // namespace picongpu
