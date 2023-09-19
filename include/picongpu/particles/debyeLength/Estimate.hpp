/* Copyright 2020-2023 Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/debyeLength/Estimate.kernel"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/AllReduce.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace debyeLength
        {
            /** Estimate Debye length for the given electron species in the local domain
             *
             * @tparam T_ElectronSpecies electron species type
             *
             * @param cellDescription mapping for kernels
             * @param minMacroparticlesPerSupercell only use supercells with at least this many macroparticles
             */
            template<typename T_ElectronSpecies>
            HINLINE Estimate
            estimateLocalDebyeLength(MappingDesc const cellDescription, uint32_t const minMacroparticlesPerSupercell)
            {
                using Frame = typename T_ElectronSpecies::FrameType;
                DataConnector& dc = Environment<>::get().DataConnector();
                auto& electrons = *(dc.get<T_ElectronSpecies>(Frame::getName()));

                auto const mapper = pmacc::makeAreaMapper<CORE + BORDER>(cellDescription);

                auto hostDeviceBuffer = pmacc::HostDeviceBuffer<Estimate, 1>{1u};
                auto hostBox = hostDeviceBuffer.getHostBuffer().getDataBox();
                hostDeviceBuffer.hostToDevice();
                auto kernel = DebyeLengthEstimateKernel{};
                auto workerCfg = lockstep::makeWorkerCfg<T_ElectronSpecies::FrameType::frameSize>();
                PMACC_LOCKSTEP_KERNEL(kernel, workerCfg)
                (mapper.getGridDim())(
                    electrons.getDeviceParticlesBox(),
                    mapper,
                    minMacroparticlesPerSupercell,
                    hostDeviceBuffer.getDeviceBuffer().getDataBox());
                hostDeviceBuffer.deviceToHost();

                // Copy is asynchronous, need to wait for it to finish
                eventSystem::getTransactionEvent().waitForFinished();
                return hostBox(0);
            }

            /** Estimate Debye length for the given electron species in the global domain
             *
             * This function must be called from all MPI ranks.
             * The resulting estimate is a reduction of local estimates from all local domains.
             *
             * @tparam T_ElectronSpecies electron species type
             *
             * @param cellDescription mapping for kernels
             * @param minMacroparticlesPerSupercell only use supercells with at least this many macroparticles
             */
            template<typename T_ElectronSpecies>
            HINLINE Estimate
            estimateGlobalDebyeLength(MappingDesc const cellDescription, uint32_t const minMacroparticlesPerSupercell)
            {
                auto localEstimate
                    = estimateLocalDebyeLength<T_ElectronSpecies>(cellDescription, minMacroparticlesPerSupercell);
                auto globalEstimate = Estimate{};
                pmacc::mpi::MPIReduce reduce;
                reduce(
                    pmacc::math::operation::Add(),
                    &globalEstimate.numUsedSupercells,
                    &localEstimate.numUsedSupercells,
                    1,
                    pmacc::mpi::reduceMethods::AllReduce());
                reduce(
                    pmacc::math::operation::Add(),
                    &globalEstimate.numFailingSupercells,
                    &localEstimate.numFailingSupercells,
                    1,
                    pmacc::mpi::reduceMethods::AllReduce());
                reduce(
                    pmacc::math::operation::Add(),
                    &globalEstimate.sumWeighting,
                    &localEstimate.sumWeighting,
                    1,
                    pmacc::mpi::reduceMethods::AllReduce());
                reduce(
                    pmacc::math::operation::Add(),
                    &globalEstimate.sumTemperatureKeV,
                    &localEstimate.sumTemperatureKeV,
                    1,
                    pmacc::mpi::reduceMethods::AllReduce());
                reduce(
                    pmacc::math::operation::Add(),
                    &globalEstimate.sumDebyeLength,
                    &localEstimate.sumDebyeLength,
                    1,
                    pmacc::mpi::reduceMethods::AllReduce());
                return globalEstimate;
            }

        } // namespace debyeLength
    } // namespace particles
} // namespace picongpu
