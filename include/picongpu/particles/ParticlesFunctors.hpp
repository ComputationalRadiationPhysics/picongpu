/* Copyright 2014-2024 Rene Widera, Marco Garten, Alexander Grund,
 *                     Heiko Burau, Axel Huebl, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <memory>

namespace picongpu
{
    namespace particles
    {
        /** assign nullptr to all attributes of a species
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the species
         */
        template<typename T_SpeciesType>
        struct AssignNull
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            void operator()()
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                species = nullptr;
            }
        };

        /** create memory for the given species type
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the species
         */
        template<typename T_SpeciesType>
        struct CreateSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            template<typename T_DeviceHeap, typename T_CellDescription>
            HINLINE void operator()(std::shared_ptr<T_DeviceHeap> const& deviceHeap, T_CellDescription* cellDesc) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                dc.consume(std::make_unique<SpeciesType>(deviceHeap, *cellDesc, FrameType::getName()));
            }
        };

        /** write memory statistics to the terminal
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the species
         */
        template<typename T_SpeciesType>
        struct LogMemoryStatisticsForSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            template<typename T_DeviceHeap>
            HINLINE void operator()(const std::shared_ptr<T_DeviceHeap>& deviceHeap) const
            {
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
                auto alpakaStream = pmacc::eventSystem::getComputeDeviceQueue(ITask::TASK_DEVICE)->getAlpakaQueue();
                log<picLog::MEMORY>("mallocMC: free slots for species %3%: %1% a %2%")
                    % deviceHeap->getAvailableSlots(
                        manager::Device<ComputeDevice>::get().current(),
                        alpakaStream,
                        sizeof(FrameType))
                    % sizeof(FrameType) % FrameType::getName();
#endif
            }
        };

        /** call method reset for the given species
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the species to reset
         */
        template<typename T_SpeciesType>
        struct CallReset
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                species->reset(currentStep);
            }
        };
    } // namespace particles
} // namespace picongpu
