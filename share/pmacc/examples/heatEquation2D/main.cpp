/* Copyright 2023 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/PngCreator.hpp"
#include "include/SetBoundaryConditions.hpp"
#include "include/StencilFourPoint.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/algorithms/GlobalReduce.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/mpi/GatherSlice.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>

#include <fstream>
#include <iostream>
#include <memory>

#define NUM_STEPS 1000
#define NUM_DEVICES_PER_DIM 2
#define THERMAL_DIFFUSIVITY 4 // POSITIVE FACTOR
#define DX 4 // GRID SPACING
#define DT 1 // TIME STEP - STABLE IF DT < (DX * DX) / (4 * THERMAL_DIFFUSIVITY)


template<typename T_Gather, typename T_GridBuffer>
inline auto createPng(uint32_t currentStep, T_Gather& gather, std::unique_ptr<T_GridBuffer> const& gridBuffer)
{
    /* gather::operator() gathers all the buffers and assembles those to
     * a complete picture discarding the guards.
     */
    if(gather->isParticipating())
    {
        const pmacc::SubGrid<DIM2>& subGrid = pmacc::Environment<DIM2>::get().SubGrid();
        auto bufferLayout = gridBuffer->getGridLayout();
        auto localDataExtents = bufferLayout.getDataSpaceWithoutGuarding();
        auto view = std::make_unique<pmacc::DeviceBuffer<float, DIM2>>(
            gridBuffer->getDeviceBuffer(),
            localDataExtents,
            bufferLayout.getGuard());
        // create a contiguous buffer required for gathering the data
        auto dataWithoutGuard = std::make_unique<pmacc::HostBuffer<float, DIM2>>(localDataExtents);
        dataWithoutGuard->copyFrom(*view.get());
        auto picture = gather->gatherSlice(
            *dataWithoutGuard.get(),
            subGrid.getGlobalDomain().size,
            subGrid.getLocalDomain().offset);
        PngCreator png;
        if(gather->isMaster())
            png(currentStep, picture->getDataBox(), picture->getDataSpace());
    }
}

auto main(int argc, char** argv) -> int
{
    const auto devices = pmacc::DataSpace<DIM2>::create(NUM_DEVICES_PER_DIM);
    const auto periodic = pmacc::DataSpace<DIM2>::create(0);
    pmacc::Environment<DIM2>::get().initDevices(devices, periodic);

    /** define a gloabl grid */
    const pmacc::DataSpace<DIM2> gridSize{256u, 256u};

    auto& gc = pmacc::Environment<DIM2>::get().GridController();

    /** device local grid size */
    const pmacc::DataSpace<DIM2> localGridSize{gridSize / devices};

    pmacc::Environment<DIM2>::get().initGrids(gridSize, localGridSize, gc.getPosition() * localGridSize);

    /** Get reference to subGrid object, which holds local position, global
     *  position, and size information, as offset of local position wrt global
     *  position
     */
    const auto& subGrid = pmacc::Environment<DIM2>::get().SubGrid();

    /** define mapping description, this defines the supercell size */
    using MappingDesc = pmacc::MappingDescription<DIM2, pmacc::math::CT::Int<16, 16>>;

    /** adds guards to the dataspace
     *  here guard size is calulated by using the supercell size
     */
    pmacc::GridLayout<DIM2> layout{subGrid.getLocalDomain().size, MappingDesc::SuperCellSize::toRT()};

    /** mapping description
     *  takes in the grid layout - CELLS dataspace + guard, and num of SUPERCELLS in guard
     */
    pmacc::DataSpace<DIM2> guardingSuperCells{1, 1};
    auto mapping = std::make_unique<MappingDesc>(layout.getDataSpace(), guardingSuperCells);

    /** define grid buffers, two because we dont do in place writes */

    auto buff1 = std::make_unique<pmacc::GridBuffer<float, DIM2>>(layout);
    auto buff2 = std::make_unique<pmacc::GridBuffer<float, DIM2>>(layout);

    pmacc::DataSpace<DIM2> guardingCells{1, 1};

    // add stencil directions only add up, down, left, and right exchanges. dont need
    // diagonals both buffers need exchanges because later we will swap pointers
    // and do exhanges in an alternating fashion
    StencilFourPoint stencilKernel{};
    for(auto i : stencilKernel.stencilDirections)
    {
        buff1->addExchange(pmacc::GUARD, pmacc::Mask(i), guardingCells, 0u);
        buff2->addExchange(pmacc::GUARD, pmacc::Mask(i), guardingCells, 1u);
    }

    /** Make locktep workercfg, takes in supercell size */
    auto workerCfg = pmacc::lockstep::makeWorkerCfg(typename MappingDesc::SuperCellSize{});

    // define mappers which run over the certain area, with a supercell size
    pmacc::AreaMapping<pmacc::type::CORE, MappingDesc> coreMapper(*mapping);
    pmacc::AreaMapping<pmacc::type::BORDER, MappingDesc> borderMapper(*mapping);

    auto setBoundaryConditions = SetBoundaryConditions{};

    /** the databox should be accessed from the buffer->getDataBox,
     *  as the state of the databox can change when someone else writes to the buffer
     */
    PMACC_LOCKSTEP_KERNEL(setBoundaryConditions, workerCfg)
    (borderMapper.getGridDim())(
        buff1->getDeviceBuffer().getDataBox(),
        NUM_DEVICES_PER_DIM,
        gc.getPosition(),
        subGrid.getLocalDomain().offset,
        gridSize,
        borderMapper);
    PMACC_LOCKSTEP_KERNEL(setBoundaryConditions, workerCfg)
    (borderMapper.getGridDim())(
        buff2->getDeviceBuffer().getDataBox(),
        NUM_DEVICES_PER_DIM,
        gc.getPosition(),
        subGrid.getLocalDomain().offset,
        gridSize,
        borderMapper);

    // create png on host of the initial conditions
    auto gather = std::make_unique<pmacc::mpi::GatherSlice>();
    createPng(0u, gather, buff1);

    // buffer to store residual
    auto residualBuffer = std::make_unique<pmacc::HostDeviceBuffer<float, DIM1>>(pmacc::DataSpace<DIM1>::create(1));

    auto hReducedResidual = pmacc::HostBuffer<float, DIM1>(pmacc::DataSpace<DIM1>::create(1));

    // scope for reduce
    {
        pmacc::mpi::MPIReduce reduce;

        // run the simulation for steps
        for(uint32_t i = 0; i < NUM_STEPS; i++)
        {
            auto splitEvent = pmacc::eventSystem::getTransactionEvent();
            auto send = buff1->asyncCommunication(splitEvent);

            /* Update Core Cells */
            PMACC_LOCKSTEP_KERNEL(StencilFourPoint{}, workerCfg)
            (coreMapper.getGridDim())(
                buff1->getDeviceBuffer().getDataBox(),
                buff2->getDeviceBuffer().getDataBox(),
                residualBuffer->getDeviceBuffer().getDataBox(),
                THERMAL_DIFFUSIVITY,
                DX,
                DT,
                coreMapper);

            /** Reset boundary borders to boundary conditions
             * not required for iter 0 as borders havent been updated yet, but is done anyway
             */
            PMACC_LOCKSTEP_KERNEL(SetBoundaryConditions{}, workerCfg)
            (borderMapper.getGridDim())(
                buff1->getDeviceBuffer().getDataBox(),
                NUM_DEVICES_PER_DIM,
                gc.getPosition(),
                subGrid.getLocalDomain().offset,
                gridSize,
                borderMapper);

            pmacc::eventSystem::setTransactionEvent(send);

            /* Update Border Cells */
            PMACC_LOCKSTEP_KERNEL(StencilFourPoint{}, workerCfg)
            (borderMapper.getGridDim())(
                buff1->getDeviceBuffer().getDataBox(),
                buff2->getDeviceBuffer().getDataBox(),
                residualBuffer->getDeviceBuffer().getDataBox(),
                THERMAL_DIFFUSIVITY,
                DX,
                DT,
                borderMapper);

            // Swap the read and write buffers
            std::swap(buff1, buff2);
            residualBuffer->deviceToHost();

            // MPI Reduce the residual
            reduce(
                pmacc::math::operation::Add(),
                hReducedResidual.getBasePointer(),
                residualBuffer->getHostBuffer().getBasePointer(),
                1, // this is a 1D dataspace, just access it?
                pmacc::mpi::reduceMethods::Reduce());
            // Reset residuals to zero for next iteration
            residualBuffer->reset(false);
            createPng(i + 1u, gather, buff1);

            if(reduce.hasResult(pmacc::mpi::reduceMethods::Reduce()))
                std::cout << "Residual at time " << DT * i << " = " << hReducedResidual.getDataBox()[0] << std::endl;
        }
    }
    /* Finalize */
    gather.reset();
    pmacc::eventSystem::getTransactionEvent().waitForFinished();
    pmacc::Environment<DIM2>::get().finalize();

    return 0;
}
