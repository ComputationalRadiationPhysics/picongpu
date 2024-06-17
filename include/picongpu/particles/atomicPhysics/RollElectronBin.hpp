/* Copyright 2023 Brian Marre
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

#include "picongpu/simulation_defines.hpp" // need: picongpu/unitless/grid.unitless for CELL_VOLUME

#include <cstdint>

namespace picongpu::particles::atomicPhysics::rollElectronBin
{
    namespace detail
    {
        /** container for rate function call for a collisional transition
         *
         * @attention reference interface implementation only, do not call!
         *
         * @param energy central energy of histogram bin
         * @param binWidth energy width of histogram bin
         * @param density number density of electrons in histogram bin, 1/(UNIT_LENGTH^3 * eV)
         * @param atomicData atomicData dataBoxes
         */
        struct RateFunctional
        {
            template<typename... AtomicData>
            HDINLINE static float_X rate(
                float_X const energy,
                float_X const binWidth,
                float_X const density,
                uint32_t const transitionIndex,
                AtomicData... atomicData)
            {
                return 0._X;
            }
        };
    } // namespace detail

    /** find electron bin corresponding to randomly rolled number according to rate contributed to total rate
     *
     * @tparam T_Histogram type of electron histogram
     * @tparam T_RateFunctional type providing function calculating rate of a single bin for a transformation
     * @tparam T_AtomicData types of atomic dataBoxes passed to T_RateFunctional::rate() call
     *
     * @param rngValue random number [0, 1)
     * @param transitionIndex collection index of a collisional transition
     * @param rate_total summed rate of all bins for the transition
     * @param electronHistogram electron histogram
     * @param atomicData atomicData dataBoxes
     */
    template<typename T_Histogram, typename T_RateFunctional, typename... T_AtomicData>
    HDINLINE static uint32_t findBin(
        float_X const rngValue,
        uint32_t const transitionIndex,
        float_X const rate_total,
        T_Histogram const& electronHistogram,
        T_AtomicData... atomicData)
    {
        // UNIT_LENGTH^3
        constexpr float_X volumeScalingFactor
            = pmacc::math::CT::volume<SuperCellSize>::type::value * picongpu::CELL_VOLUME;

        float_X probabilityOffset = 0._X;

        constexpr uint32_t numberBinsMinus1 = T_Histogram::numberBins - 1u;
        for(uint32_t i = 0u; i < numberBinsMinus1; ++i)
        {
            // eV
            float_X const energy = electronHistogram.getBinEnergy(i);
            // eV
            float_X const binWidth = electronHistogram.getBinWidth(i);
            // 1/(UNIT_LENGTH^3 * eV)
            float_X const density = electronHistogram.getBinWeight0(i) / volumeScalingFactor / binWidth;
            // 1/UNIT_TIME
            float_X const rate_bin = T_RateFunctional::rate(energy, binWidth, density, transitionIndex, atomicData...);

            probabilityOffset += rate_bin / rate_total;
            if(probabilityOffset > rngValue)
                return i;
        }
        return numberBinsMinus1;
    }
} // namespace picongpu::particles::atomicPhysics::rollElectronBin
