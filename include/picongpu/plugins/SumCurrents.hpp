/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Felix Schmitt, Benjamin Worpitz, Richard Pausch
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

#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/plugins/ILightweightPlugin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

#include <iostream>
#include <memory>


namespace picongpu
{
    using namespace pmacc;

    namespace po = boost::program_options;

    using J_DataBox = FieldJ::DataBoxType;

    struct KernelSumCurrents
    {
        template<typename Mapping, typename T_Worker>
        DINLINE void operator()(T_Worker const& worker, J_DataBox fieldJ, float3_X* gCurrent, Mapping mapper) const
        {
            using SuperCellSize = typename Mapping::SuperCellSize;

            PMACC_SMEM(worker, sh_sumJ, float3_X);

            const DataSpace<simDim> threadIndex(cupla::threadIdx(worker.getAcc()));

            auto onlyMaster = lockstep::makeMaster(worker);

            onlyMaster([&]() { sh_sumJ = float3_X::create(0.0); });

            worker.sync();

            const DataSpace<simDim> superCellIdx(
                mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc()))));

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;
            lockstep::makeForEach<cellsPerSuperCell>(worker)(
                [&](int32_t const linearIdx)
                {
                    const auto cellIdxInSupercell = pmacc::math::mapToND(SuperCellSize::toRT(), linearIdx);
                    const DataSpace<simDim> cell(superCellIdx * SuperCellSize::toRT() + cellIdxInSupercell);
                    const float3_X myJ = fieldJ(cell);
                    cupla::atomicAdd(worker.getAcc(), &(sh_sumJ.x()), myJ.x(), ::alpaka::hierarchy::Threads{});
                    cupla::atomicAdd(worker.getAcc(), &(sh_sumJ.y()), myJ.y(), ::alpaka::hierarchy::Threads{});
                    cupla::atomicAdd(worker.getAcc(), &(sh_sumJ.z()), myJ.z(), ::alpaka::hierarchy::Threads{});
                });

            worker.sync();

            onlyMaster(
                [&]()
                {
                    cupla::atomicAdd(worker.getAcc(), &(gCurrent->x()), sh_sumJ.x(), ::alpaka::hierarchy::Blocks{});
                    cupla::atomicAdd(worker.getAcc(), &(gCurrent->y()), sh_sumJ.y(), ::alpaka::hierarchy::Blocks{});
                    cupla::atomicAdd(worker.getAcc(), &(gCurrent->z()), sh_sumJ.z(), ::alpaka::hierarchy::Blocks{});
                });
        }
    };

    class SumCurrents : public ILightweightPlugin
    {
    private:
        MappingDesc* cellDescription{nullptr};
        std::string notifyPeriod;

        std::unique_ptr<GridBuffer<float3_X, DIM1>> sumcurrents;

    public:
        SumCurrents()
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        ~SumCurrents() override = default;

        void notify(uint32_t currentStep) override
        {
            const int rank = Environment<simDim>::get().GridController().getGlobalRank();
            const float3_X gCurrent = getSumCurrents();

            // gCurrent is just j
            // j = I/A

            auto const realCurrent = float3_X(
                gCurrent.x() * CELL_HEIGHT * CELL_DEPTH,
                gCurrent.y() * CELL_WIDTH * CELL_DEPTH,
                gCurrent.z() * CELL_WIDTH * CELL_HEIGHT);

            float3_64 realCurrent_SI(
                float_64(realCurrent.x()) * (UNIT_CHARGE / UNIT_TIME),
                float_64(realCurrent.y()) * (UNIT_CHARGE / UNIT_TIME),
                float_64(realCurrent.z()) * (UNIT_CHARGE / UNIT_TIME));

            /*FORMAT OUTPUT*/
            using dbl = std::numeric_limits<float_64>;

            std::cout.precision(dbl::digits10);
            if(math::abs(gCurrent.x()) + math::abs(gCurrent.y()) + math::abs(gCurrent.z()) != float_X(0.0))
                std::cout << "[ANALYSIS] [" << rank << "] [COUNTER] [SumCurrents] [" << currentStep << std::scientific
                          << "] " << realCurrent_SI << " l2norm:" << pmacc::math::l2norm(realCurrent_SI) << std::endl;
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()(
                "sumcurr.period",
                po::value<std::string>(&notifyPeriod),
                "enable plugin [for each n-th step]");
        }

        std::string pluginGetName() const override
        {
            return "SumCurrents";
        }

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            this->cellDescription = cellDescription;
        }

    private:
        void pluginLoad() override
        {
            if(!notifyPeriod.empty())
            {
                sumcurrents = std::make_unique<GridBuffer<float3_X, DIM1>>(
                    DataSpace<DIM1>(1)); // create one int on gpu und host

                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
            }
        }

        float3_X getSumCurrents()
        {
            DataConnector& dc = Environment<>::get().DataConnector();
            auto fieldJ = dc.get<FieldJ>(FieldJ::getName());

            sumcurrents->getDeviceBuffer().setValue(float3_X::create(0.0));

            auto const mapper = makeAreaMapper<CORE + BORDER>(*cellDescription);

            auto workerCfg = lockstep::makeWorkerCfg(SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(KernelSumCurrents{}, workerCfg)
            (mapper.getGridDim())(fieldJ->getDeviceDataBox(), sumcurrents->getDeviceBuffer().getBasePointer(), mapper);

            sumcurrents->deviceToHost();
            return sumcurrents->getHostBuffer().getDataBox()[0];
        }
    };

} // namespace picongpu
