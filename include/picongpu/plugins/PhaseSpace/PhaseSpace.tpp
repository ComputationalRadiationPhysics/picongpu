/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Marco Garten
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

#include "DumpHBufferOpenPMD.hpp"
#include "PhaseSpace.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/vector/Int.hpp>
#include <pmacc/math/vector/Size_t.hpp>

#include <algorithm>
#include <vector>


namespace picongpu
{
    template<class AssignmentFunction, class Species>
    PhaseSpace<AssignmentFunction, Species>::PhaseSpace(
        std::shared_ptr<plugins::multi::IHelp>& help,
        size_t const id,
        MappingDesc* cellDescription)
        : m_cellDescription(cellDescription)
        , m_help(std::static_pointer_cast<Help>(help))
        , m_id(id)
    {
        // unit is m_species c (for a single "real" particle)
        float_X pRangeSingle_unit(frame::getMass<typename Species::FrameType>() * SPEED_OF_LIGHT);

        axis_p_range.first = m_help->momentum_range_min.get(id) * pRangeSingle_unit;
        axis_p_range.second = m_help->momentum_range_max.get(id) * pRangeSingle_unit;
        /* String to Enum conversion */
        uint32_t el_space;
        if(m_help->element_space.get(id) == "x")
            el_space = AxisDescription::x;
        else if(m_help->element_space.get(id) == "y")
            el_space = AxisDescription::y;
        else if(m_help->element_space.get(id) == "z")
            el_space = AxisDescription::z;
        else
            throw PluginException("[Plugin] [" + m_help->getOptionPrefix() + "] space must be x, y or z");

        uint32_t el_momentum = AxisDescription::px;
        if(m_help->element_momentum.get(id) == "px")
            el_momentum = AxisDescription::px;
        else if(m_help->element_momentum.get(id) == "py")
            el_momentum = AxisDescription::py;
        else if(m_help->element_momentum.get(id) == "pz")
            el_momentum = AxisDescription::pz;
        else
            throw PluginException("[Plugin] [" + m_help->getOptionPrefix() + "] momentum must be px, py or pz");

        axis_element.momentum = el_momentum;
        axis_element.space = el_space;

        bool activatePlugin = true;

        if constexpr(simDim == DIM2)
            if(el_space == AxisDescription::z)
            {
                std::cerr << "[Plugin] [" + m_help->getOptionPrefix() + "] Skip requested output for "
                          << m_help->element_space.get(id) << m_help->element_momentum.get(id) << std::endl;
                activatePlugin = false;
            }

        if(activatePlugin)
        {
            /** create dir */
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions("phaseSpace");

            const uint32_t r_element = axis_element.space;

            /* CORE + BORDER elements for spatial bins */
            this->r_bins = this->m_cellDescription->getGridLayout().getDataSpaceWithoutGuarding()[r_element];

            auto const num_pbinsToAvoidOdrUse = this->num_pbins;
            this->dBuffer
                = std::make_unique<HostDeviceBuffer<float_PS, 2>>(DataSpace<2>(num_pbinsToAvoidOdrUse, r_bins));

            /* reduce-add phase space from other GPUs in range [p0;p1]x[r;r+dr]
             * to "lowest" node in range
             * e.g.: phase space x-py: reduce-add all nodes with same x range in
             *                         spatial y and z direction to node with
             *                         lowest y and z position and same x range
             */
            pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();
            pmacc::math::Size_t<simDim> gpuDim = gc.getGpuNodes();
            pmacc::math::Int<simDim> gpuPos = gc.getPosition();

            /* my plane means: the r_element I am calculating should be 1GPU in width */
            pmacc::math::Size_t<simDim> sizeTransversalPlane(gpuDim);
            sizeTransversalPlane[this->axis_element.space] = 1;

            for(int planePos = 0; planePos <= (int) gpuDim[this->axis_element.space]; ++planePos)
            {
                auto mpiReduce = std::make_unique<mpi::MPIReduce>();
                bool isInGroup = (gpuPos[this->axis_element.space] == planePos);

                mpiReduce->participate(isInGroup);
                if(isInGroup)
                {
                    this->isPlaneReduceRoot = mpiReduce->hasResult(::pmacc::mpi::reduceMethods::Reduce{});
                    this->planeReduce = std::move(mpiReduce);
                }
            }

            /* Create MPI communicator for openPMD IO with ranks of each plane reduce root */
            {
                /* Array with root ranks of the planeReduce operations */
                std::vector<int> planeReduceRootRanks(gc.getGlobalSize(), -1);
                /* Am I one of the planeReduce root ranks? my global rank : -1 */
                int myRootRank = gc.getGlobalRank() * this->isPlaneReduceRoot - (!this->isPlaneReduceRoot);

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_Group world_group, new_group;
                MPI_CHECK(
                    MPI_Allgather(&myRootRank, 1, MPI_INT, planeReduceRootRanks.data(), 1, MPI_INT, MPI_COMM_WORLD));

                /* remove all non-roots (-1 values) */
                std::sort(planeReduceRootRanks.begin(), planeReduceRootRanks.end());
                std::vector<int> ranks(
                    std::lower_bound(planeReduceRootRanks.begin(), planeReduceRootRanks.end(), 0),
                    planeReduceRootRanks.end());

                MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
                MPI_CHECK(MPI_Group_incl(world_group, ranks.size(), ranks.data(), &new_group));
                MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, new_group, &commFileWriter));
                MPI_CHECK(MPI_Group_free(&new_group));
                MPI_CHECK(MPI_Group_free(&world_group));
            }

            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));
        }
    }


    template<class AssignmentFunction, class Species>
    PhaseSpace<AssignmentFunction, Species>::~PhaseSpace()
    {
        if(commFileWriter != MPI_COMM_NULL)
        {
            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            eventSystem::getTransactionEvent().waitForFinished();
            MPI_CHECK_NO_EXCEPT(MPI_Comm_free(&commFileWriter));
        }
    }

    template<class AssignmentFunction, class Species>
    template<uint32_t r_dir>
    void PhaseSpace<AssignmentFunction, Species>::calcPhaseSpace(const uint32_t currentStep)
    {
        /* register particle species observer */
        DataConnector& dc = Environment<>::get().DataConnector();
        auto particles = dc.get<Species>(Species::FrameType::getName());

        StartBlockFunctor<r_dir> startBlockFunctor(
            particles->getDeviceParticlesBox(),
            dBuffer->getDeviceBuffer().getDataBox(),
            this->axis_element.momentum,
            this->axis_p_range,
            *this->m_cellDescription);

        auto bindFunctor = std::bind(
            startBlockFunctor,
            // particle filter
            std::placeholders::_1);

        meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<boost::mpl::_1>>{}(
            m_help->filter.get(m_id),
            currentStep,
            bindFunctor);
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpace<AssignmentFunction, Species>::notify(uint32_t currentStep)
    {
        /* reset device buffer */
        dBuffer->getDeviceBuffer().setValue(float_PS(0.0));

        /* calculate local phase space */
        if(this->axis_element.space == AxisDescription::x)
            calcPhaseSpace<AxisDescription::x>(currentStep);
        else if(this->axis_element.space == AxisDescription::y)
            calcPhaseSpace<AxisDescription::y>(currentStep);
#if(SIMDIM == DIM3)
        else
            calcPhaseSpace<AxisDescription::z>(currentStep);
#endif

        /* transfer to host */
        this->dBuffer->deviceToHost();
        auto bufferExtent = this->dBuffer->getDeviceBuffer().getDataSpace();

        auto hReducedBuffer = HostBuffer<float_PS, 2>(this->dBuffer->getDeviceBuffer().getDataSpace());

        eventSystem::getTransactionEvent().waitForFinished();

        /* reduce-add phase space from other GPUs in range [p0;p1]x[r;r+dr]
         * to "lowest" node in range
         * e.g.: phase space x-py: reduce-add all nodes with same x range in
         *                         spatial y and z direction to node with
         *                         lowest y and z position and same x range
         */
        (*planeReduce)(
            pmacc::math::operation::Add(),
            hReducedBuffer.data(),
            this->dBuffer->getHostBuffer().data(),
            bufferExtent.productOfComponents(),
            mpi::reduceMethods::Reduce());

        eventSystem::getTransactionEvent().waitForFinished();

        /** all non-reduce-root processes are done now */
        if(!this->isPlaneReduceRoot)
            return;

        /* write to file */
        const float_64 UNIT_VOLUME = UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH;
        const float_64 unit = UNIT_CHARGE / UNIT_VOLUME;

        /* (momentum) p range: unit is m_species * c
         *   During the kernels we calculate with a typical single/real
         *   momentum range. Now for the dump the meta information of units
         *   on the p-axis should be scaled to represent single/real particles.
         *   @see PhaseSpaceMulti::pluginLoad( )
         */
        float_64 const pRange_unit = UNIT_MASS * UNIT_SPEED;

        DumpHBuffer dumpHBuffer;

        if(this->commFileWriter != MPI_COMM_NULL)
            dumpHBuffer(
                hReducedBuffer,
                this->axis_element,
                this->axis_p_range,
                pRange_unit,
                unit,
                Species::FrameType::getName() + "_" + m_help->filter.get(m_id),
                m_help->file_name_extension.get(m_id),
                m_help->json_config.get(m_id),
                currentStep,
                this->commFileWriter);
    }

} /* namespace picongpu */
