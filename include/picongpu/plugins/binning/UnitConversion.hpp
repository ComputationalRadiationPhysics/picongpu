/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <array>
#include <map>

#if (ENABLE_OPENPMD == 1)
#    include <openPMD/openPMD.hpp>
#endif

namespace picongpu
{
    namespace plugins::binning
    {
        constexpr unsigned numUnits = 7;

        // @todo add where this 7D format is from
        static std::array<double, numUnits> UnitDimensions{
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
        HINLINE double getConversionFactor(std::array<double, numUnits> const& myDimension)
        {
            double conversion_factor = 1.;
            for(size_t i = 0; i < 7; i++)
            {
                conversion_factor *= std::pow(UnitDimensions[i], myDimension[i]);
            };
            return conversion_factor;
        }

        template<typename T>
        HINLINE T toPICUnits(T varSI, std::array<double, numUnits> const& myDimension)
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
        HINLINE double toSIUnits(T varPIC, std::array<double, numUnits> const& myDimension)
        {
            return static_cast<double>(varPIC) * getConversionFactor(myDimension);
        }

#if (ENABLE_OPENPMD == 1)
        HINLINE std::map<::openPMD::UnitDimension, double> makeOpenPMDUnitMap(
            const std::array<double, numUnits>& myDimension)
        {
            using UD = ::openPMD::UnitDimension;

            static constexpr std::array<UD, numUnits> keys = {UD::L, UD::M, UD::T, UD::I, UD::theta, UD::N, UD::J};

            std::map<UD, double> map;

            // Combine the two arrays into the map
            for(size_t i = 0; i < keys.size(); ++i)
            {
                map[keys[i]] = myDimension[i];
            }
            return map;
        }
#endif
    } // namespace plugins::binning
} // namespace picongpu
