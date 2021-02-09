/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include <pmacc/static_assert.hpp>

#include "picongpu/fields/FieldJ.hpp"

#include <pmacc/math/vector/Int.hpp>
#include <pmacc/math/vector/Float.hpp>
#include <pmacc/math/vector/Size_t.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/cuSTL/container/DeviceBuffer.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/cuSTL/container/PseudoBuffer.hpp>
#include <pmacc/cuSTL/cursor/NestedCursor.hpp>
#include <pmacc/cuSTL/algorithm/kernel/Foreach.hpp>
#include <pmacc/cuSTL/algorithm/host/Foreach.hpp>
#include <pmacc/cuSTL/algorithm/mpi/Gather.hpp>
#include <pmacc/cuSTL/algorithm/kernel/Reduce.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/meta/ForEach.hpp>

#include "common/txtFileHandling.hpp"

#include <sstream>


namespace picongpu
{
    ChargeConservation::ChargeConservation()
        : name("ChargeConservation: Print the maximum charge deviation between particles and div E to textfile "
               "'chargeConservation.dat'")
        , prefix("chargeConservation")
        , filename("chargeConservation.dat")
        , cellDescription(nullptr)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    void ChargeConservation::pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()(
            (this->prefix + ".period").c_str(),
            po::value<std::string>(&this->notifyPeriod),
            "enable plugin [for each n-th step]");
    }

    std::string ChargeConservation::pluginGetName() const
    {
        return this->name;
    }

    void ChargeConservation::pluginLoad()
    {
        if(this->notifyPeriod.empty())
            return;

        Environment<>::get().PluginConnector().setNotificationPeriod(this, this->notifyPeriod);

        pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
        using namespace pmacc::math;
        Size_t<simDim> gpuDim = (Size_t<simDim>) con.getGpuNodes();
        zone::SphericZone<simDim> zone_allGPUs(gpuDim);
        this->allGPU_reduce = AllGPU_reduce(new pmacc::algorithm::mpi::Reduce<simDim>(zone_allGPUs));

        if(this->allGPU_reduce->root())
        {
            this->output_file.open(this->filename.c_str(), std::ios_base::app);
            this->output_file << "#timestep max-charge-deviation unit[As]" << std::endl;
        }
    }

    void ChargeConservation::restart(uint32_t restartStep, const std::string restartDirectory)
    {
        if(this->notifyPeriod.empty())
            return;

        if(!this->allGPU_reduce->root())
            return;

        restoreTxtFile(this->output_file, this->filename, restartStep, restartDirectory);
    }

    void ChargeConservation::checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
        if(this->notifyPeriod.empty())
            return;

        if(!this->allGPU_reduce->root())
            return;

        checkpointTxtFile(this->output_file, this->filename, currentStep, checkpointDirectory);
    }

    void ChargeConservation::setMappingDescription(MappingDesc* cellDescription)
    {
        this->cellDescription = cellDescription;
    }

    namespace detail
    {
        /**
         * @class Div
         * @brief divergence functor for 2D and 3D
         *
         * NOTE: This functor uses a Yee-cell stencil.
         */
        template<int dim, typename ValueType>
        struct Div;

        template<typename ValueType>
        struct Div<DIM3, ValueType>
        {
            using result_type = ValueType;

            template<typename Field>
            HDINLINE ValueType operator()(Field field) const
            {
                const ValueType reciWidth = float_X(1.0) / cellSize.x();
                const ValueType reciHeight = float_X(1.0) / cellSize.y();
                const ValueType reciDepth = float_X(1.0) / cellSize.z();
                return ((*field).x() - (*field(-1, 0, 0)).x()) * reciWidth
                    + ((*field).y() - (*field(0, -1, 0)).y()) * reciHeight
                    + ((*field).z() - (*field(0, 0, -1)).z()) * reciDepth;
            }
        };

        template<typename ValueType>
        struct Div<DIM2, ValueType>
        {
            using result_type = ValueType;

            template<typename Field>
            HDINLINE ValueType operator()(Field field) const
            {
                const ValueType reciWidth = float_X(1.0) / cellSize.x();
                const ValueType reciHeight = float_X(1.0) / cellSize.y();
                return ((*field).x() - (*field(-1, 0)).x()) * reciWidth
                    + ((*field).y() - (*field(0, -1)).y()) * reciHeight;
            }
        };

        // functor for all species to calculate density
        template<typename T_SpeciesType, typename T_Area>
        struct ComputeChargeDensity
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            static const uint32_t area = T_Area::value;

            HINLINE void operator()(FieldTmp* fieldTmp, const uint32_t currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                /* load species without copying the particle data to the host */
                auto speciesTmp = dc.get<SpeciesType>(SpeciesType::FrameType::getName(), true);

                /* run algorithm */
                using ChargeDensitySolver = typename particles::particleToGrid::CreateFieldTmpOperation_t<
                    SpeciesType,
                    particles::particleToGrid::derivedAttributes::ChargeDensity>::Solver;

                fieldTmp->computeValue<area, ChargeDensitySolver>(*speciesTmp, currentStep);
                dc.releaseData(SpeciesType::FrameType::getName());
            }
        };

        struct CalculateAndAssignChargeDeviation
        {
            template<typename T_Rho, typename T_FieldE, typename T_Acc>
            HDINLINE void operator()(const T_Acc& acc, T_Rho& rho, const T_FieldE& fieldECursor) const
            {
                typedef Div<simDim, typename FieldTmp::ValueType> MyDiv;

                /* rho := | div E * eps_0 - rho | */
                rho.x() = math::abs((MyDiv{}(fieldECursor) *EPS0 - rho).x());
            }
        };

    } // namespace detail

    void ChargeConservation::notify(uint32_t currentStep)
    {
        typedef SuperCellSize BlockDim;

        DataConnector& dc = Environment<>::get().DataConnector();

        /* load FieldTmp without copy data to host */
        PMACC_CASSERT_MSG(_please_allocate_at_least_one_FieldTmp_in_memory_param, fieldTmpNumSlots > 0);
        auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0), true);
        /* reset density values to zero */
        fieldTmp->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));

        using EligibleSpecies = typename bmpl::
            copy_if<VectorAllSpecies, particles::traits::SpeciesEligibleForSolver<bmpl::_1, ChargeConservation>>::type;

        // todo: log species that are used / ignored in this plugin with INFO

        /* calculate and add the charge density values from all species in FieldTmp */
        meta::ForEach<
            EligibleSpecies,
            picongpu::detail::ComputeChargeDensity<bmpl::_1, bmpl::int_<CORE + BORDER>>,
            bmpl::_1>
            computeChargeDensity;
        computeChargeDensity(fieldTmp.get(), currentStep);

        /* add results of all species that are still in GUARD to next GPUs BORDER */
        EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(fieldTmpEvent);

        /* cast PMacc Buffer to cuSTL Buffer */
        auto fieldTmp_coreBorder = fieldTmp->getGridBuffer().getDeviceBuffer().cartBuffer().view(
            this->cellDescription->getGuardingSuperCells() * BlockDim::toRT(),
            this->cellDescription->getGuardingSuperCells() * -BlockDim::toRT());

        /* cast PMacc Buffer to cuSTL Buffer */
        auto fieldE_coreBorder = dc.get<FieldE>(FieldE::getName(), true)
                                     ->getGridBuffer()
                                     .getDeviceBuffer()
                                     .cartBuffer()
                                     .view(
                                         this->cellDescription->getGuardingSuperCells() * BlockDim::toRT(),
                                         this->cellDescription->getGuardingSuperCells() * -BlockDim::toRT());

        /* run calculation: fieldTmp = | div E * eps_0 - rho | */
        typedef picongpu::detail::Div<simDim, typename FieldTmp::ValueType> myDiv;
        algorithm::kernel::Foreach<BlockDim>()(
            fieldTmp_coreBorder.zone(),
            fieldTmp_coreBorder.origin(),
            cursor::make_NestedCursor(fieldE_coreBorder.origin()),
            ::picongpu::detail::CalculateAndAssignChargeDeviation());

        /* reduce charge derivation (fieldTmp) to get the maximum value */
        typename FieldTmp::ValueType maxChargeDiff = algorithm::kernel::Reduce()(
            fieldTmp_coreBorder.origin(),
            fieldTmp_coreBorder.zone(),
            pmacc::nvidia::functors::Max());

        /* reduce again across mpi cluster */
        container::HostBuffer<typename FieldTmp::ValueType, 1> maxChargeDiff_host(1);
        *maxChargeDiff_host.origin() = maxChargeDiff;
        container::HostBuffer<typename FieldTmp::ValueType, 1> maxChargeDiff_cluster(1);
        (*this->allGPU_reduce)(
            maxChargeDiff_cluster,
            maxChargeDiff_host,
            ::pmacc::math::Max<typename FieldTmp::ValueType, typename FieldTmp::ValueType>());

        if(!this->allGPU_reduce->root())
            return;

        this->output_file << currentStep << " " << (*maxChargeDiff_cluster.origin() * CELL_VOLUME).x() << " "
                          << UNIT_CHARGE << std::endl;
    }

} // namespace picongpu
