/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file describes the interface all histograms must follow
 *
 * @attention some functions in this file are commented out, these functions are nevertheless required!
 *
 * NOTE: templates can not be virtual
 */

#pragma once

#include <cstdint>

namespace picongpu::particles::atomicPhysics::electronDistribution
{
    class HistogramInterface
    {
        //! number histogram bins
        // static constexpr uint32_t numberBins

        //! check whether the given energy, [eV] is in the histogram energy range
        // HDINLINE virtual bool inRange(float_X const energy) const = 0;

        //! get binIndex for a given energy [eV]
        // HDINLINE virtual uint32_t getBinIndex(float_X const energy) const = 0;

        //! get the central Energy for a given binIndex, @return unit: eV
        // HDINLINE virtual float_X getBinEnergy(uint32_t const binIndex) const = 0;

        //! get energy width of bin, @return unit: eV
        // HDINLINE virtual float_X getBinWidth(uint32_t const binIndex) const = 0;

        //! get weight of initially binned particles
        // HDINLINE virtual float_X getBinWeight0(uint32_t const binIndex) const = 0;

        //! get reserved/already used weight of bin
        // HDINLINE virtual float_X getBinDeltaWeight(uint32_t const binIndex) const = 0;

        //! get deltaEnergy of Bin, @return unit: eV
        // HDINLINE virtual float_X getBinDeltaEnergy(uint32_t const binIndex) const = 0;

        /** bin the particle, add weight to w0 of the corresponding bin
         *
         * @tparam T_Acc ... accelerator
         *
         * @param acc ... description of the device to execute this on
         * @param energy ... physical particle energy, [eV]
         * @param weight ... weight of the macroParticle, unitless
         */
        /* template<typename T_Worker>
         * void binParticle(
         *      T_Worker const& worker,
         *      float_X const energy,
         *      float_X const weight)
         */

        /** add to the deltaWeight of a given bin
         *
         * @tparam T_Acc ... accelerator type
         *
         * @param acc ... description of the device to execute this on
         * @param binIndex ... physical particle energy, unitless
         * @param weight ... weight of the macroParticle, unitless
         */
        /* template<typename T_Worker>
         * void addDeltaWeight(
         *      T_Worker const& worker,
         *      uint32_t const binIndex,
         *      float_X const
         *      weight)
         */

        /** add to the deltaEnergy of a given bin
         *
         * @tparam T_Acc ... accelerator type
         *
         * @param acc ... description of the device to execute this on
         * @param binIndex ... physical particle energy, unitless
         * @param weight ... weight of the macroParticle, unitless
         */
        /* template<typename T_Worker>
         * void addDeltaEnergy(
         *      T_Worker const& worker,
         *      uint32_t const binIndex,
         *      float_X const deltaEnergy)
         */

        //! returns number of calls we need to make to reset the histogram
        // static constexpr uint32_t getNumberResetOps() = 0;
    };
} // namespace picongpu::particles::atomicPhysics::electronDistribution
