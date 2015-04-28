/**
 * Copyright 2013-2015 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
#include "cuSTL/algorithm/mpi/Reduce.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "lambda/Expression.hpp"
#include <sstream>

#include "cuSTL/algorithm/kernel/Reduce.hpp"

namespace picongpu
{

template<typename Species>
ChargeConservation<Species>::ChargeConservation()
    : name("print the maximum charge derivation between particles and div E"), 
      prefix("electronChargeConservation")
{
    Environment<>::get().PluginConnector().registerPlugin(this);
}

template<typename Species>
void ChargeConservation<Species>::pluginRegisterHelp(po::options_description& desc)
{
    desc.add_options()
        ((this->prefix + "_frequency").c_str(),
        po::value<uint32_t > (&this->notifyFrequency)->default_value(0), "notifyFrequency");
}

template<typename Species>
std::string ChargeConservation<Species>::pluginGetName() const {return this->name;}

template<typename Species>
void ChargeConservation<Species>::pluginLoad()
{
    Environment<>::get().PluginConnector().setNotificationPeriod(this, this->notifyFrequency);
}

template<int dim, typename ValueType>
struct Div;

template<typename ValueType>
struct Div<DIM3, ValueType>
{
    typedef ValueType result_type;

    template<typename Field>
    HDINLINE ValueType operator()(Field field) const
    {
        const ValueType reciWidth = float_X(1.0) / CELL_WIDTH;
        const ValueType reciHeight = float_X(1.0) / CELL_HEIGHT;
        const ValueType reciDepth = float_X(1.0) / CELL_DEPTH;
        return ((*field(1,0,0)).x() - (*field).x()) * reciWidth +
               ((*field(0,1,0)).y() - (*field).y()) * reciHeight +
               ((*field(0,0,1)).z() - (*field).z()) * reciDepth;
    }
};

template<typename ValueType>
struct Div<DIM2, ValueType>
{
    typedef ValueType result_type;

    template<typename Field>
    HDINLINE ValueType operator()(Field field) const
    {
        const ValueType reciWidth = float_X(1.0) / CELL_WIDTH;
        const ValueType reciHeight = float_X(1.0) / CELL_HEIGHT;
        return ((*field(1,0)).x() - (*field).x()) * reciWidth +
               ((*field(0,1)).y() - (*field).y()) * reciHeight;
    }
};

template<typename Species>
void ChargeConservation<Species>::notify(uint32_t currentStep)
{
    typedef SuperCellSize BlockDim;

    DataConnector &dc = Environment<>::get().DataConnector();

    /*## update field tmp with charge information ##*/
    /*load FieldTmp without copy data to host*/
    FieldTmp* fieldTmp = &(dc.getData<FieldTmp > (FieldTmp::getName(), true));
    /*load particle without copy particle data to host*/
    Species* speciesTmp = &(dc.getData<Species >(Species::FrameType::getName(), true));

    fieldTmp->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));
    /*run algorithm*/
    typedef typename CreateDensityOperation<Species>::type::Solver DensitySolver;

    fieldTmp->computeValue < CORE + BORDER, DensitySolver > (*speciesTmp, currentStep);

    EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
    __setTransactionEvent(fieldTmpEvent);
    dc.releaseData(Species::FrameType::getName());
    
    /* cast libPMacc Buffer to cuSTL Buffer */
    BOOST_AUTO(fieldTmp_coreBorder,
                 fieldTmp->getGridBuffer().
                 getDeviceBuffer().cartBuffer().
                 view(BlockDim::toRT(), -BlockDim::toRT()));
    
    /* cast libPMacc Buffer to cuSTL Buffer */
    BOOST_AUTO(fieldE_coreBorder,
                 dc.getData<FieldE > (FieldE::getName(), true).getGridBuffer().
                 getDeviceBuffer().cartBuffer().
                 view(BlockDim::toRT(), -BlockDim::toRT()));

    /* run calculation: fieldTmp = | div E - rho / eps_0 | */
    using namespace lambda;
    using namespace PMacc::math::math_functor;
    typedef Div<simDim, typename FieldTmp::ValueType> myDiv;
    algorithm::kernel::Foreach<BlockDim>()
        (fieldTmp_coreBorder.zone(), fieldTmp_coreBorder.origin(), 
        cursor::make_NestedCursor(fieldE_coreBorder.origin()),
            _1 = _abs(expr(myDiv())(_2) - _1 / EPS0));
            
    /* reduce charge derivation (fieldTmp) to get the maximum value */
    container::DeviceBuffer<typename FieldTmp::ValueType, 1> maxChargeDiff(1);
    algorithm::kernel::Reduce<BlockDim>()
        (maxChargeDiff.origin(), fieldTmp_coreBorder.zone(), fieldTmp_coreBorder.origin(), _max(_1, _2));
    container::HostBuffer<typename FieldTmp::ValueType, 1> maxChargeDiff_host(1);
    maxChargeDiff_host = maxChargeDiff;
    
    /* reduce again across mpi cluster */
    PMacc::GridController<simDim>& con = PMacc::Environment<simDim>::get().GridController();
    using namespace PMacc::math;
    Size_t<simDim> gpuDim = (Size_t<simDim>)con.getGpuNodes();
    zone::SphericZone<simDim> zone_allGPUs(gpuDim);
    PMacc::algorithm::mpi::Reduce<simDim> allGPU_reduce(zone_allGPUs);
    container::HostBuffer<typename FieldTmp::ValueType, 1> maxChargeDiff_cluster(1);
    allGPU_reduce(maxChargeDiff_cluster, maxChargeDiff_host, _max(_1, _2));

    if(!allGPU_reduce.root()) return;

    static std::ofstream file("chargeConservation.dat");

    file << "step: " << currentStep << ", max charge derivation: "
        << *maxChargeDiff_cluster.origin() * CELL_VOLUME << " * " << UNIT_CHARGE
        << " Coulomb" << std::endl;
}

} // namespace picongpu
