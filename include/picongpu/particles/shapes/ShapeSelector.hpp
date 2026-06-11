/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace shapes
    {
        /** Just in time assignment shape evaluation functor.
         *
         * @tparam T_ParticleAssignFunctor functor for assignment
         */
        template<typename T_ParticleAssignFunctor>
        struct Jit
        {
            /** Create the Jit shape assignment functor.
             *
             * @param position position of the particle
             * @param isInBaseAssignmentCell If the position given is within the assignment cell.
             *                               True for event particle shape if position is [-0.5;0.5), else false.
             *                               True for odd particle shape if position is [0.0;1.0), else false,
             */
            float_X const particlePosition;

            HDINLINE Jit(float_X const position, [[maybe_unused]] bool const isInBaseAssignmentCell)
                : particlePosition(position)
            {
            }

            /** Create the Jit shape assignment functor.
             *
             * @param position position of the particle
             */
            HDINLINE Jit(float_X const position) : particlePosition(position)
            {
            }

            /** Assignment value on the position of the grid point.
             *
             * @param gridPointOffset grid offset relative to the cell where the particle is located.
             * @return Assignment value based on the particle shape.
             */
            HDINLINE float_X operator()(int const gridPointOffset) const
            {
                return T_ParticleAssignFunctor()(float_X(gridPointOffset) - particlePosition);
            }
        };

        /** Cached assignment shape functor.
         *
         * The result of the assignment shape evaluation will be cached to avoid duplicated calculations.
         *
         * @tparam T_ParticleAssignFunctor functor for assignment
         */
        template<typename T_ParticleAssignFunctor>
        struct Cached
        {
            /** Create the Cached shape assignment functor.
             *
             * @param position Position of the particle.
             *                 Range of position for even particles shapes [-0.5;1.5).
             *                 Range of position for odd particles shapes [0.0;2.0).
             * @param isInBaseAssignmentCell If the position given is within the assignment cell.
             *                               True for event particle shape if position is [-0.5;0.5), else false.
             *                               True for odd particle shape if position is [0.0;1.0), else false,
             */
            HDINLINE Cached(float_X const position, bool const isInBaseAssignmentCell)
                : shapeArray(std::move(T_ParticleAssignFunctor().shapeArray(position, !isInBaseAssignmentCell)))
            {
            }

            /** Assignment value on the position of the grid point.
             *
             * @param gridPointOffset grid offset relative to the cell where the particle is located.
             * @return Assignment value based on the particle shape.
             */
            HDINLINE float_X operator()(int const gridPointOffset) const
            {
                return shapeArray[gridPointOffset - T_ParticleAssignFunctor::begin];
            }

            using ShapeArrayType = decltype(T_ParticleAssignFunctor().shapeArray(
                alpaka::core::declval<float_X>(),
                alpaka::core::declval<bool>()));
            ShapeArrayType shapeArray;
        };

    } // namespace shapes
} // namespace picongpu
