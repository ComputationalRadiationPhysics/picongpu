/* Copyright 2013-2021 Heiko Burau, Axel Huebl
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
#include <pmacc/Environment.hpp>

#include "picongpu/simulation/control/Simulation.hpp"

#include <pmacc/simulationControl/SimulationHelper.hpp>

#include "picongpu/fields/FieldE.hpp"

#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/eventSystem/EventSystem.hpp>

#include <pmacc/nvidia/memory/MemoryInfo.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include "picongpu/ArgsParser.hpp"
#include "picongpu/plugins/PluginController.hpp"

#include <pmacc/cuSTL/container/DeviceBuffer.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/cuSTL/algorithm/kernel/Foreach.hpp>
#include <pmacc/cuSTL/algorithm/kernel/Reduce.hpp>
#include <pmacc/cuSTL/algorithm/mpi/Gather.hpp>
#include <pmacc/cuSTL/algorithm/mpi/Reduce.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/cuSTL/cursor/tools/slice.hpp>
#include <pmacc/cuSTL/cursor/FunctorCursor.hpp>

#include <pmacc/cuSTL/container/allocator/DeviceMemEvenPitchAllocator.hpp>
#include <pmacc/cuSTL/algorithm/host/Foreach.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/cuSTL/algorithm/functor/GetComponent.hpp>
#include <pmacc/cuSTL/algorithm/functor/Add.hpp>

#include <cassert>
#include <memory>

namespace picongpu
{
    using namespace pmacc;

    class ThermalTestSimulation : public Simulation
    {
    public:
        ThermalTestSimulation() : Simulation()
        {
        }

        void init()
        {
            Simulation::init();

            using namespace ::pmacc::math;

            DataConnector& dc = Environment<>::get().DataConnector();
            auto fieldE = dc.get<FieldE>(FieldE::getName(), true);

            auto fieldE_coreBorder = fieldE->getGridBuffer().getDeviceBuffer().cartBuffer().view(
                precisionCast<int>(GuardDim().toRT()),
                -precisionCast<int>(GuardDim().toRT()));

            this->eField_zt[0] = std::make_unique<container::HostBuffer<float, 2>>(
                Size_t<2>(fieldE_coreBorder.size().z(), this->collectTimesteps));
            this->eField_zt[1] = std::make_unique<container::HostBuffer<float, 2>>(this->eField_zt[0]->size());

            dc.releaseData(FieldE::getName());
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            Simulation::pluginRegisterHelp(desc);
        }

        void pluginLoad()
        {
            Simulation::pluginLoad();
        }

        virtual ~ThermalTestSimulation() = default;

        void writeOutput()
        {
            using namespace ::pmacc::math;

            auto& con = Environment<simDim>::get().GridController();
            Size_t<SIMDIM> gpuDim = (Size_t<SIMDIM>) con.getGpuNodes();
            Int<3> gpuPos = (Int<3>) con.getPosition();
            zone::SphericZone<SIMDIM> gpuGatheringZone(Size_t<SIMDIM>(1, 1, gpuDim.z()));
            algorithm::mpi::Gather<SIMDIM> gather(gpuGatheringZone);

            container::HostBuffer<float, 2> eField_zt_reduced(eField_zt[0]->size());

            for(int i = 0; i < 2; i++)
            {
                bool reduceRoot = (gpuPos.x() == 0) && (gpuPos.y() == 0);
                for(int gpuPos_z = 0; gpuPos_z < (int) gpuDim.z(); gpuPos_z++)
                {
                    zone::SphericZone<3> gpuReducingZone(Size_t<3>(gpuDim.x(), gpuDim.y(), 1), Int<3>(0, 0, gpuPos_z));

                    algorithm::mpi::Reduce<3> reduce(gpuReducingZone, reduceRoot);

                    reduce(eField_zt_reduced, *(eField_zt[i]), pmacc::algorithm::functor::Add());
                }
                if(!reduceRoot)
                    continue;

                container::HostBuffer<float, 2> global_eField_zt(
                    gpuDim.z() * eField_zt_reduced.size().x(),
                    eField_zt_reduced.size().y());

                gather(global_eField_zt, eField_zt_reduced, 1);
                if(gather.root())
                {
                    std::string filename;
                    if(i == 0)
                        filename = "eField_zt_trans.dat";
                    else
                        filename = "eField_zt_long.dat";
                    std::ofstream eField_zt_dat(filename.data());
                    eField_zt_dat << global_eField_zt;
                    eField_zt_dat.close();
                }
            }
        }

        /**
         * Run one simulation step.
         *
         * @param currentStep iteration number of the current step
         */
        void runOneStep(uint32_t currentStep)
        {
            Simulation::runOneStep(currentStep);

            if(currentStep > this->collectTimesteps + firstTimestep)
                return;
            if(currentStep < firstTimestep)
                return;

            using namespace math;

            DataConnector& dc = Environment<>::get().DataConnector();
            auto fieldE = dc.get<FieldE>(FieldE::getName(), true);

            auto fieldE_coreBorder = fieldE->getGridBuffer().getDeviceBuffer().cartBuffer().view(
                precisionCast<int>(GuardDim().toRT()),
                -precisionCast<int>(GuardDim().toRT()));

            for(size_t z = 0; z < eField_zt[0]->size().x(); z++)
            {
                zone::SphericZone<2> reduceZone(fieldE_coreBorder.size().shrink<2>());
                for(int i = 0; i < 2; i++)
                {
                    *(eField_zt[i]->origin()(z, currentStep - firstTimestep)) = algorithm::kernel::Reduce()(
                        cursor::make_FunctorCursor(
                            cursor::tools::slice(fieldE_coreBorder.origin()(0, 0, z)),
                            pmacc::algorithm::functor::GetComponent<typename FieldE::ValueType::type>(i == 0 ? 0 : 2)),
                        reduceZone,
                        nvidia::functors::Add());
                }
            }

            dc.releaseData(FieldE::getName());

            if(currentStep == this->collectTimesteps + firstTimestep)
                writeOutput();
        }

    private:
        // number of timesteps which collect the data
        static constexpr uint32_t collectTimesteps = 512;
        // first timestep which collects data
        //   you may like to let the plasma develope/thermalize a little bit
        static constexpr uint32_t firstTimestep = 1024;

        std::array<std::unique_ptr<container::HostBuffer<float, 2>>, 2> eField_zt;

        using BlockDim = pmacc::math::CT::Size_t<16, 16, 1>;
        using GuardDim = SuperCellSize;
    };

} // namespace picongpu
