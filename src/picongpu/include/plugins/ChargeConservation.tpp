/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "math/vector/Int.hpp"
#include "math/vector/Float.hpp"
#include "math/vector/Size_t.hpp"
#include "math/vector/math_functor/abs.hpp"
#include "math/vector/math_functor/max.hpp"
#include "cuSTL/container/PseudoBuffer.hpp"
#include "dataManagement/DataConnector.hpp"
#include "fields/FieldJ.hpp"
#include "math/Vector.hpp"
#include "cuSTL/algorithm/mpi/Gather.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "cuSTL/cursor/NestedCursor.hpp"
#include "lambda/Expression.hpp"
#include <sstream>
#include "algorithms/ForEach.hpp"
#include "cuSTL/algorithm/kernel/Reduce.hpp"
#include "nvidia/functors/Add.hpp"
#include "common/txtFileHandling.hpp"

namespace picongpu
{

ChargeConservation::ChargeConservation()
    : name("ChargeConservation: Print the maximum charge deviation between particles and div E to textfile 'chargeConservation.dat'"),
      prefix("chargeConservation"), filename("chargeConservation.dat"),
      cellDescription(NULL)
{
    Environment<>::get().PluginConnector().registerPlugin(this);
}

void ChargeConservation::pluginRegisterHelp(po::options_description& desc)
{
    desc.add_options()
        ((this->prefix + ".period").c_str(),
        po::value<uint32_t > (&this->notifyPeriod)->default_value(0), "enable plugin [for each n-th step]");
}

std::string ChargeConservation::pluginGetName() const {return this->name;}

void ChargeConservation::pluginLoad()
{
    if(this->notifyPeriod == 0u)
        return;

    Environment<>::get().PluginConnector().setNotificationPeriod(this, this->notifyPeriod);

    PMacc::GridController<simDim>& con = PMacc::Environment<simDim>::get().GridController();
    using namespace PMacc::math;
    Size_t<simDim> gpuDim = (Size_t<simDim>)con.getGpuNodes();
    zone::SphericZone<simDim> zone_allGPUs(gpuDim);
    this->allGPU_reduce = AllGPU_reduce(new PMacc::algorithm::mpi::Reduce<simDim>(zone_allGPUs));

    if(this->allGPU_reduce->root())
    {
        this->output_file.open(this->filename.c_str(), std::ios_base::app);
        this->output_file << "#timestep max-charge-deviation unit[As]" << std::endl;
    }
}

void ChargeConservation::restart(uint32_t restartStep, const std::string restartDirectory)
{
    if(this->notifyPeriod == 0u)
        return;

    if(!this->allGPU_reduce->root())
        return;

    restoreTxtFile( this->output_file,
                    this->filename,
                    restartStep,
                    restartDirectory );
}

void ChargeConservation::checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
{
    if(this->notifyPeriod == 0u)
        return;

    if(!this->allGPU_reduce->root())
        return;

    checkpointTxtFile( this->output_file,
                       this->filename,
                       currentStep,
                       checkpointDirectory );
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
    typedef ValueType result_type;

    template<typename Field>
    HDINLINE ValueType operator()(Field field) const
    {
        const ValueType reciWidth = float_X(1.0) / cellSize.x();
        const ValueType reciHeight = float_X(1.0) / cellSize.y();
        const ValueType reciDepth = float_X(1.0) / cellSize.z();
        return ((*field).x() - (*field(-1,0,0)).x()) * reciWidth +
               ((*field).y() - (*field(0,-1,0)).y()) * reciHeight +
               ((*field).z() - (*field(0,0,-1)).z()) * reciDepth;
    }
};

template<typename ValueType>
struct Div<DIM2, ValueType>
{
    typedef ValueType result_type;

    template<typename Field>
    HDINLINE ValueType operator()(Field field) const
    {
        const ValueType reciWidth = float_X(1.0) / cellSize.x();
        const ValueType reciHeight = float_X(1.0) / cellSize.y();
        return ((*field).x() - (*field(-1,0)).x()) * reciWidth +
               ((*field).y() - (*field(0,-1)).y()) * reciHeight;
    }
};

// functor for all species to calculate density
template<typename T_SpeciesName, typename T_Area>
struct ComputeChargeDensity
{
    typedef typename T_SpeciesName::type SpeciesName;
    static const uint32_t area = T_Area::value;

    HINLINE void operator()( FieldTmp* fieldTmp,
                             const uint32_t currentStep) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        /* load species without copying the particle data to the host */
        SpeciesName* speciesTmp = &(dc.getData<SpeciesName >(SpeciesName::FrameType::getName(), true));

        /* run algorithm */
        typedef typename CreateDensityOperation<SpeciesName>::type::Solver ChargeDensitySolver;
        fieldTmp->computeValue < area, ChargeDensitySolver > (*speciesTmp, currentStep);
        dc.releaseData(SpeciesName::FrameType::getName());
    }
};

} // namespace detail

void ChargeConservation::notify(uint32_t currentStep)
{
    typedef SuperCellSize BlockDim;

    DataConnector &dc = Environment<>::get().DataConnector();

    /* load FieldTmp without copy data to host */
    FieldTmp* fieldTmp = &(dc.getData<FieldTmp > (FieldTmp::getName(), true));
    /* reset density values to zero */
    fieldTmp->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));

    /* calculate and add the charge density values from all species in FieldTmp */
    ForEach<VectorAllSpecies, picongpu::detail::ComputeChargeDensity<bmpl::_1,bmpl::int_<CORE + BORDER> >, MakeIdentifier<bmpl::_1> > computeChargeDensity;
    computeChargeDensity(forward(fieldTmp), currentStep);

    /* add results of all species that are still in GUARD to next GPUs BORDER */
    EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
    __setTransactionEvent(fieldTmpEvent);

    /* cast libPMacc Buffer to cuSTL Buffer */
    BOOST_AUTO(fieldTmp_coreBorder,
                 fieldTmp->getGridBuffer().
                 getDeviceBuffer().cartBuffer().
                 view(this->cellDescription->getGuardingSuperCells()*BlockDim::toRT(),
                      this->cellDescription->getGuardingSuperCells()*-BlockDim::toRT()));

    /* cast libPMacc Buffer to cuSTL Buffer */
    BOOST_AUTO(fieldE_coreBorder,
                 dc.getData<FieldE > (FieldE::getName(), true).getGridBuffer().
                 getDeviceBuffer().cartBuffer().
                 view(this->cellDescription->getGuardingSuperCells()*BlockDim::toRT(),
                      this->cellDescription->getGuardingSuperCells()*-BlockDim::toRT()));

    /* run calculation: fieldTmp = | div E * eps_0 - rho | */
    using namespace lambda;
    using namespace PMacc::math::math_functor;
    typedef picongpu::detail::Div<simDim, typename FieldTmp::ValueType> myDiv;
    algorithm::kernel::Foreach<BlockDim>()
        (fieldTmp_coreBorder.zone(), fieldTmp_coreBorder.origin(),
        cursor::make_NestedCursor(fieldE_coreBorder.origin()),
            _1 = _abs(expr(myDiv())(_2) * EPS0 - _1));

    /* reduce charge derivation (fieldTmp) to get the maximum value */
    typename FieldTmp::ValueType maxChargeDiff =
        algorithm::kernel::Reduce()
            (fieldTmp_coreBorder.origin(), fieldTmp_coreBorder.zone(), PMacc::nvidia::functors::Max());

    /* reduce again across mpi cluster */
    container::HostBuffer<typename FieldTmp::ValueType, 1> maxChargeDiff_host(1);
    *maxChargeDiff_host.origin() = maxChargeDiff;
    container::HostBuffer<typename FieldTmp::ValueType, 1> maxChargeDiff_cluster(1);
    (*this->allGPU_reduce)(maxChargeDiff_cluster, maxChargeDiff_host, _max(_1, _2));

    if(!this->allGPU_reduce->root()) return;

    this->output_file << currentStep << " " << (*maxChargeDiff_cluster.origin() * CELL_VOLUME).x()
        << " " << UNIT_CHARGE << std::endl;
}

} // namespace picongpu
