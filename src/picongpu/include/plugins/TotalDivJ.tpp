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
#include "fields/FieldJ.hpp"
#include "math/vector/compile-time/Int.hpp"
#include "math/vector/compile-time/Size_t.hpp"
#include "cuSTL/algorithm/mpi/Gather.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "lambda/Expression.hpp"
#include <sstream>

#include "cuSTL/algorithm/kernel/Reduce.hpp"

namespace picongpu
{

TotalDivJ::TotalDivJ(std::string name, std::string prefix)
    : name(name), prefix(prefix)
{
    ModuleConnector::getInstance().registerModule(this);
}

void TotalDivJ::moduleRegisterHelp(po::options_description& desc)
{
    desc.add_options()
        ((this->prefix + "_frequency").c_str(),
        po::value<uint32_t > (&this->notifyFrequency)->default_value(0), "notifyFrequency");
}

std::string TotalDivJ::moduleGetName() const {return this->name;}

void TotalDivJ::moduleLoad()
{
    DataConnector::getInstance().registerObserver(this, this->notifyFrequency);
}
void TotalDivJ::moduleUnload(){}

struct Div
{
    typedef float_X result_type;
    
    template<typename Field>
    HDINLINE float_X operator()(Field field) const
    {
        return ((*field(1,0,0)).x() - (*field).x()) * (float_X(1.0) / CELL_WIDTH) +
               ((*field(0,1,0)).y() - (*field).y()) * (float_X(1.0) / CELL_HEIGHT) +
               ((*field(0,0,1)).z() - (*field).z()) * (float_X(1.0) / CELL_DEPTH);
    }
};

void TotalDivJ::notify(uint32_t currentStep)
{
    namespace vec = PMacc::math;
    using namespace vec;
    typedef vec::CT::Size_t<TILE_WIDTH,TILE_HEIGHT,TILE_DEPTH> BlockDim;
    
    DataConnector &dc = DataConnector::getInstance();
    
    container::PseudoBuffer<float3_X, 3> fieldJ
        (dc.getData<FieldJ > (FieldJ::getName(), true).getGridBuffer().getDeviceBuffer());
        
    container::DeviceBuffer<float, 3> fieldDivJ(fieldJ.size());
    zone::SphericZone<3> coreBorderZone(fieldJ.zone().size - (size_t)2*BlockDim().vec(),
                                        fieldJ.zone().offset + precisionCast<int>(BlockDim().vec()));
    //std::cout << coreBorderZone.size << ", " << coreBorderZone.offset << std::endl;
    using namespace lambda;
    algorithm::kernel::Foreach<BlockDim>()
        (coreBorderZone, fieldDivJ.origin(), cursor::make_NestedCursor(fieldJ.origin()), 
            _1 = expr(Div())(_2));
        
    container::DeviceBuffer<float, 1> totalDivJ(1);
    algorithm::kernel::Reduce<BlockDim>()
        (totalDivJ.origin(), coreBorderZone, fieldDivJ.origin(), _1 + _2);
    container::HostBuffer<float, 1> totalDivJ_host(1);
    totalDivJ_host = totalDivJ;
    
    static std::ofstream file("totalDivJ.dat");
    
    file << "step: " << currentStep << ", totalDivJ: " 
        << *totalDivJ_host.origin() * CELL_VOLUME * UNIT_CHARGE
        << " Coulomb" << std::endl;
}

}
