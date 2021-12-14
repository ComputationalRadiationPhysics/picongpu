/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
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

#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"

#include <cstdint>


namespace picongpu
{
    namespace currentSolver
    {
        /** Helper base class for Esirkepov and Esirkepov-like current deposition implementation
         *
         * Implements basic calculations for the given particle assignment function.
         * Naming matches the Esirkepov paper, however indexing is from PIConGPU.
         * For a fully paper-conforming notation see EsirkepovNative implementation.
         *
         * @tparam T_ParticleAssignFunctor assignment functor type
         */
        template<typename T_ParticleAssignFunctor>
        struct Base
        {
            using ParticleAssignFunctor = T_ParticleAssignFunctor;

            /** Calculate terms depending on particle position and assignment function
             *
             * @param line element with previous and current position of the particle
             * @param gridPoint used grid point to evaluate assignment shape
             * @param d dimension range {0,1,2} means {x,y,z}
             *          different to Esirkepov paper, here we use C style
             * @{
             */

            //! Calculate term in S0 for the start position
            DINLINE static float_X S0(const Line<floatD_X>& line, const float_X gridPoint, const uint32_t d)
            {
                return ParticleAssignFunctor()(gridPoint - line.m_pos0[d]);
            }

            //! Calculate term in S1 for end position
            DINLINE static float_X S1(const Line<floatD_X>& line, const float_X gridPoint, const uint32_t d)
            {
                return ParticleAssignFunctor()(gridPoint - line.m_pos1[d]);
            }

            //! Calculate difference term in DS
            DINLINE static float_X DS(const Line<floatD_X>& line, const float_X gridPoint, const uint32_t d)
            {
                return S1(line, gridPoint, d) - S0(line, gridPoint, d);
            }

            /*! @} */
        };

    } // namespace currentSolver
} // namespace picongpu
