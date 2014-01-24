/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
#include "cuSTL/container/PseudoBuffer.hpp"
#include "dataManagement/DataConnector.hpp"
#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "math/vector/compile-time/Int.hpp"
#include "math/vector/compile-time/Size_t.hpp"
#include "cuSTL/algorithm/mpi/Gather.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/cursor/tools/slice.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "lambda/Expression.hpp"
#include <sstream>

#include "cuSTL/algorithm/kernel/Reduce.hpp"

namespace picongpu
{

FieldEnergy::FieldEnergy(std::string name, std::string prefix)
    : name(name), prefix(prefix)
{
    ModuleConnector::getInstance().registerModule(this);
}

void FieldEnergy::moduleRegisterHelp(po::options_description& desc)
{
    desc.add_options()
        ((this->prefix + "_frequency").c_str(),
        po::value<uint32_t > (&this->notifyFrequency)->default_value(0), "notifyFrequency");
}

std::string FieldEnergy::moduleGetName() const {return this->name;}

void FieldEnergy::moduleLoad()
{
    DataConnector::getInstance().registerObserver(this, this->notifyFrequency);
}
void FieldEnergy::moduleUnload(){}

void FieldEnergy::notify(uint32_t currentStep)
{

    namespace math = PMacc::math;
    using namespace math;
    typedef math::CT::Size_t<TILE_WIDTH,TILE_HEIGHT,TILE_DEPTH> BlockDim;
    
    DataConnector &dc = DataConnector::getInstance();
    FieldE& fieldE = dc.getData<FieldE > (FIELD_E, true);
    FieldB& fieldB = dc.getData<FieldB > (FIELD_B, true);

    BOOST_AUTO(fieldE_coreBorder,
            fieldE.getGridBuffer().getDeviceBuffer().cartBuffer().view(typeCast<int>(BlockDim().vec()), -typeCast<int>(BlockDim().vec())));
    BOOST_AUTO(fieldB_coreBorder,
            fieldB.getGridBuffer().getDeviceBuffer().cartBuffer().view(typeCast<int>(BlockDim().vec()), -typeCast<int>(BlockDim().vec())));
            
    PMacc::GridController<3>& con = PMacc::GridController<3>::getInstance();
    PMacc::math::Size_t<3> gpuDim = (math::Size_t<3>)con.getGpuNodes();
    PMacc::math::Size_t<3> globalGridSize = gpuDim * fieldE_coreBorder.size();
    int globalCellZPos = globalGridSize.z() / 2;
    int localCellZPos = globalCellZPos % fieldE_coreBorder.size().z();
    int gpuZPos = globalCellZPos / fieldE_coreBorder.size().z();
    
    zone::SphericZone<3> gpuGatheringZone(math::Size_t<3>(gpuDim.x(), gpuDim.y(), 1), 
                                          PMacc::math::Int<3>(0,0,gpuZPos));
    algorithm::mpi::Gather<3> gather(gpuGatheringZone);
    if(!gather.participate()) return;
    container::DeviceBuffer<float, 2> energyDBuffer(fieldE_coreBorder.size().shrink<2>());
        
    using namespace lambda;
    BOOST_AUTO(_abs2, expr(math::Abs2()));
    
    algorithm::kernel::Foreach<math::CT::Int<TILE_WIDTH,TILE_HEIGHT,1> >()(
        energyDBuffer.zone(), energyDBuffer.origin(),
                              cursor::tools::slice(fieldE_coreBorder.origin()(0,0,localCellZPos)),
                              cursor::tools::slice(fieldB_coreBorder.origin()(0,0,localCellZPos)),
        _1 = (_abs2(_2) + _abs2(_3) * MUE0_EPS0) * 
            (float_X(0.5) * EPS0 * UNIT_ENERGY * UNITCONV_Joule_to_keV / (UNIT_LENGTH*UNIT_LENGTH*UNIT_LENGTH)));
            
            
    container::HostBuffer<float, 2> energyHBuffer(energyDBuffer.size());
    energyHBuffer = energyDBuffer;
    
    /*\todo: domain size is now different for any gpu [fixme] */
    container::HostBuffer<float, 2> globalEnergyBuffer(energyHBuffer.size() * gpuDim.shrink<2>());
    gather(globalEnergyBuffer, energyHBuffer, 2);
    if(!gather.root()) return;
    std::ostringstream filename;
    filename << "FieldEnergy_" << currentStep << ".dat";
    std::ofstream file(filename.str().c_str());
    file << globalEnergyBuffer;
}

}
