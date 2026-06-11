/*
 * SPDX-FileCopyrightText: Rene Widera, Marco Garten, Alexander Grund, Heiko Burau, Axel Huebl, Sergei Bastrakov Filip Optolowicz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/param.hpp"

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
            HINLINE void operator()(std::shared_ptr<T_DeviceHeap> const& deviceHeap) const
            {
#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
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

            HINLINE void operator()(uint32_t const currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                species->reset(currentStep);
            }
        };
    } // namespace particles
} // namespace picongpu
