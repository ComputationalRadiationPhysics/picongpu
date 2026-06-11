/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"

#include <cstdint>

namespace picongpu
{
    namespace currentSolver
    {
        /** Helper for Esirkepov and Esirkepov-like current deposition implementation
         *
         * Implements basic calculations for the given particle assignment function.
         * Naming matches the Esirkepov paper, however indexing is from PIConGPU.
         *
         * @tparam T_StartAssignmentFunction type of shape functor of for the particle at the start position
         * @tparam T_EndAssignmentFunction type of shape functor of for the particle at the end position
         */
        template<typename T_StartAssignmentFunction, typename T_EndAssignmentFunction>
        class TrajectoryAssignmentShapeFunction
        {
            T_StartAssignmentFunction const m_shapeStartParticle;
            T_EndAssignmentFunction const m_shapeEndParticle;

        public:
            /** Constructor
             *
             * The shape functor must have the interface to call `operator()(relative_grid_point)` and return the
             * assignment value for the given grid point.
             *
             * @param shapeStartParticle Shape functor for the start particle.
             * @param shapeEndParticle Shape functor for the start particle.
             */
            HDINLINE TrajectoryAssignmentShapeFunction(
                T_StartAssignmentFunction shapeStartParticle,
                T_EndAssignmentFunction shapeEndParticle)
                : m_shapeStartParticle(shapeStartParticle)
                , m_shapeEndParticle(shapeEndParticle)
            {
            }

            /** Calculate terms depending on particle position and assignment function
             *
             * @param gridPoint used grid point to evaluate assignment shape
             * @{
             */

            //! Calculate term in S0 for the start position
            DINLINE float_X S0(int const gridPoint) const
            {
                return m_shapeStartParticle(gridPoint);
            }

            //! Calculate term in S1 for end position
            DINLINE float_X S1(int const gridPoint) const
            {
                return m_shapeEndParticle(gridPoint);
            }

            //! Calculate difference term in DS
            DINLINE float_X DS(int const gridPoint) const
            {
                return S1(gridPoint) - S0(gridPoint);
            }

            /*! @} */
        };

        /** Factory
         *
         *  The shape functor must have the interface to call `operator()(relative_grid_point)` and return the
         * assignment value for the given grid point.
         * @tparam T_StartAssignmentFunction type of shape functor of for the particle at the start position
         * @tparam T_EndAssignmentFunction type of shape functor of for the particle at the end position
         *
         * @param shapeStartParticle Shape functor for the start particle.
         * @param shapeEndParticle Shape functor for the start particle.
         * @return Instance of the functor required for the Esirkepov like current deposition.
         */
        template<typename T_StartAssignmentFunction, typename T_EndAssignmentFunction>
        HDINLINE auto makeTrajectoryAssignmentShapeFunction(
            T_StartAssignmentFunction shapeStartParticle,
            T_EndAssignmentFunction shapeEndParticle)
        {
            return TrajectoryAssignmentShapeFunction<T_StartAssignmentFunction, T_EndAssignmentFunction>(
                shapeStartParticle,
                shapeEndParticle);
        }

    } // namespace currentSolver
} // namespace picongpu
