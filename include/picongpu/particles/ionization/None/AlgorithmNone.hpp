/* Copyright 2014-2021 Marco Garten
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

#include <pmacc/types.hpp>

/** \file AlgorithmNone.hpp
 *
 * IONIZATION ALGORITHM for the model None
 *
 * - implements the calculation of ionization probability and changes charge states
 * - is called with the IONIZATION MODEL, specifically by setting the flag in @see speciesDefinition.param
 */

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** \struct AlgorithmNone
             *
             * \brief ionization algorithm that does nothing
             */
            struct AlgorithmNone
            {
                /** Functor implementation
                 *
                 * \tparam EType type of electric field
                 * \tparam BType type of magnetic field
                 * \tparam ParticleType type of particle to be ionized
                 *
                 * \param bField magnetic field value at t=0
                 * \param eField electric field value at t=0
                 * \param parentIon particle instance to be ionized with position at t=0 and momentum at t=-1/2
                 */
                template<typename EType, typename BType, typename ParticleType>
                HDINLINE void operator()(const BType bField, const EType eField, ParticleType& parentIon)
                {
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
