/* Copyright 2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Filip Optolowicz
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

#include "picongpu/defines.hpp"

#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>

namespace picongpu::simulation::stage
{
    /** Functor for the stage of the PIC loop performing synchrotron radiation
     *
     * Only affects particle species with the Synchrotron attribute.
     */
    class SynchrotronRadiation
    {
    private:
        /** exponential integration
         *
         * @brief
         * 1) approximate function with exponential function by using the xLeft and xMiddle point to fit the exponent
         * 2) integrate the exponential function on log scale,
         *
         * @note overestimates function below xMiddle but underestimates function above xMiddle leading to better fit
         * overall (for bessel function).
         *
         * @param xLeft left point
         * @param xMiddle middle point
         * @param xRight right point
         * @param yLeft function value at xLeft
         * @param yMiddle function value at xMiddle
         */
        template<typename T_Number>
        T_Number integrateAsExponential(
            T_Number const xLeft,
            T_Number const xMiddle,
            T_Number const xRight,
            T_Number const yLeft,
            T_Number const yMiddle);

        /** compute first synchrotorn function. See paper: "Extended particle-in-cell schemes for physics in
         * ultrastrong laser fields: Review and developments" by A. Gonoskov et.Al.
         *
         * @param zq
         *
         * @returns zq * (integral of 2nd kind bessel function from zq to ~infinity)
         */
        float_64 firstSynchrotronFunction(float_64 const zq);

        //! @note see paper: "Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and
        //! developments" by A. Gonoskov et.Al.
        float_64 secondSynchrotronFunction(float_64 const x);

        /** Set the failedRequirementQ to true or false
         * @param value
         */
        void setFailedRequirementQ(bool const value);

        /** Check if the requirements are met
         * @returns true if the requirements are not met
         */
        bool checkFailedRequirementQ();

        /** set tableValuesF1F2 at index iZq and Accessor
         * @param iZq index
         * @param accessor Accessor
         * @param value value to set
         */
        template<typename T_Accessor>
        void setTableValuesF1F2(uint32_t const iZq, T_Accessor const accessor, float_X const value);

    public:
        /** Create a synchrotron radiation instance: precompute first and second synchrotron functions
         *
         * @param cellDescription mapping for kernels
         */
        SynchrotronRadiation(MappingDesc const cellDescription);

        /** Radiation happens here
         *
         * @param step index of time iteration
         */
        void operator()(uint32_t const step);

    private:
        /** creates and initialize all required helper fields */
        void init();

        //! Mapping for kernels
        MappingDesc cellDescription;

        /** precomputed first and second synchrotron functions:
         *  -> 2d grid of floats_64 -> tableValuesF1F2[zq][0/1] ; 0/1 = F1/F2
         */
        std::shared_ptr<GridBuffer<float_X, 2>> tableValuesF1F2;
        //! used for table access
        enum struct Accessor : uint32_t
        {
            f1 = 0u,
            f2 = 1u
        };
        constexpr uint32_t u32(Accessor const t)
        {
            return static_cast<uint32_t>(t);
        }

        /** flag to check if the requirements 1 and 2 are met
         * We check the requirements in class SynchrotronIdea() in:
         * picongpu/include/picongpu/particles/synchrotron/AlgorithmSynchrotron.hpp
         */
        std::shared_ptr<GridBuffer<int32_t, 1>> failedRequirementQ;
        bool failedRequirementPrinted;
    };
} // namespace picongpu::simulation::stage
