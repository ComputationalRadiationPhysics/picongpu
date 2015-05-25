/**
 * Copyright 2015 Marco Garten
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

#include "particles/ionization/byField/BSI/BSI.def"
#include "particles/ionization/byField/BSI/AlgorithmBSI.hpp"

#include "compileTime/conversion/TypeToPointerPair.hpp"

namespace picongpu
{
namespace particles
{
namespace ionization
{

    /** \struct BSI
     * \brief Barrier Suppression Ionization
     */
    template<typename T_DestSpecies>
    struct BSI
    {

        /* define ionization ALGORITHM for ionization MODEL */
        typedef particles::ionization::AlgorithmBSI IonizationAlgorithm;

        /** Functor implementation
         * \tparam T_SrcSpecies particle species to be ionized
         * \tparam T_ParticleStorage \see picongpu/src/libPMacc/include/math/MapTuple.hpp
         *
         * \param currentStep current time step */
        template<typename T_SrcSpecies,typename T_ParticleStorage>
        void operator()(T_SrcSpecies& src, T_ParticleStorage& pst, const uint32_t currentStep)
        {

            /* alias for electron species to be created */
            PMACC_AUTO(electrons,pst[typename MakeIdentifier<T_DestSpecies>::type()]);
            /* call the ionization function */
            src.ionize(electrons, currentStep);
        }

    };

} // namespace ionization
} // namespace particles
} // namespace picongpu
