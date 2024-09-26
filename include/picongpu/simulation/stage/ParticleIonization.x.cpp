/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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


#include "picongpu/simulation/stage/ParticleIonization.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/particles/creation/creation.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/ionization/byCollision/ionizers.hpp"
#include "picongpu/particles/ionization/byField/ionizers.hpp"
#include "picongpu/particles/traits/GetIonizerList.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        /** Call an ionization method upon an ion species
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is going to be ionized
         * with ionization scheme T_SelectIonizer
         */
        template<typename T_SpeciesType, typename T_SelectIonizer>
        struct CallIonizationScheme
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using SelectIonizer = T_SelectIonizer;
            using FrameType = typename SpeciesType::FrameType;

            /* define the type of the species to be created
             * from inside the ionization model specialization
             */
            using DestSpecies = typename SelectIonizer::DestSpecies;
            using DestFrameType = typename DestSpecies::FrameType;

            /** Functor implementation
             *
             * @tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * @param cellDesc logical block information like dimension and cell sizes
             * @param currentStep The current time step
             */
            template<typename T_CellDescription>
            HINLINE void operator()(T_CellDescription cellDesc, const uint32_t currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                // alias for pointer on source species
                auto srcSpeciesPtr = dc.get<SpeciesType>(FrameType::getName());
                // alias for pointer on destination species
                auto electronsPtr = dc.get<DestSpecies>(DestFrameType::getName());

                SelectIonizer selectIonizer(currentStep);

                creation::createParticlesFromSpecies(*srcSpeciesPtr, *electronsPtr, selectIonizer, cellDesc);

                /* fill the gaps in the created species' particle frames to ensure that only
                 * the last frame is not completely filled but every other before is full
                 */
                electronsPtr->fillAllGaps();
            }
        };

        /** Call all ionization schemes of an ion species
         *
         * Tests if species can be ionized and calls the kernels to do that
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked for ionization
         */
        template<typename T_SpeciesType>
        struct CallIonization
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            // SelectIonizer will be either the specified one or fallback: None
            using SelectIonizerList = typename traits::GetIonizerList<SpeciesType>::type;

            /** Functor implementation
             *
             * @tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * @param cellDesc logical block information like dimension and cell sizes
             * @param currentStep The current time step
             */
            template<typename T_CellDescription>
            HINLINE void operator()(T_CellDescription cellDesc, const uint32_t currentStep) const
            {
                // only if an ionizer has been specified, this is executed
                using hasIonizers = typename HasFlag<FrameType, ionizers<>>::type;
                if(hasIonizers::value)
                {
                    meta::ForEach<SelectIonizerList, CallIonizationScheme<SpeciesType, boost::mpl::_1>>
                        particleIonization;
                    particleIonization(cellDesc, currentStep);
                }
            }
        };
    } // namespace particles
    namespace simulation
    {
        namespace stage
        {
            /** Ionize particles
             *
             * @param step index of time iteration
             */
            void ParticleIonization::operator()(uint32_t const step) const
            {
                using pmacc::particles::traits::FilterByFlag;
                using SpeciesWithIonizers = typename FilterByFlag<VectorAllSpecies, ionizers<>>::type;
                pmacc::meta::ForEach<SpeciesWithIonizers, particles::CallIonization<boost::mpl::_1>>
                    particleIonization;
                particleIonization(cellDescription, step);
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
