/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera, Felix Schmitt
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
#include "SliceFieldPrinter.hpp"
#include <math/vector/tools/twistVectorAxes.hpp>
#include <sstream>

namespace picongpu
{

template<typename Field>
void SliceFieldPrinter<Field>::pluginLoad()
{
    Environment<>::get().DataConnector().registerObserver(this, this->notifyFrequency);
    namespace vec = ::PMacc::math;
    typedef vec::CT::Size_t<TILE_WIDTH,TILE_HEIGHT,TILE_DEPTH> BlockDim;
    
    vec::Size_t<3> size = vec::Size_t<3>(this->cellDescription->getGridSuperCells()) * BlockDim().vec()
        - (size_t)2 * BlockDim().vec();
    this->dBuffer = new container::DeviceBuffer<float3_X, 2>(
        size.shrink<2>((this->plane+1)%3));
}

template<typename Field>
void SliceFieldPrinter<Field>::pluginUnload()
{
    delete this->dBuffer;
}

template<typename Field>
void SliceFieldPrinter<Field>::notify(uint32_t currentStep)
{
    namespace vec = ::PMacc::math;
    typedef vec::CT::Size_t<TILE_WIDTH,TILE_HEIGHT,TILE_DEPTH> BlockDim;
    DataConnector &dc = Environment<>::get().DataConnector();

    BOOST_AUTO(field_coreBorder,
        dc.getData<Field > (Field::getName(), true).getGridBuffer().
            getDeviceBuffer().cartBuffer().
            view(precisionCast<int>(BlockDim().vec()), -precisionCast<int>(BlockDim().vec())));

    std::ostringstream filename;
    filename << this->fieldName << "_" << currentStep << ".dat";
    printSlice(field_coreBorder, this->plane, this->slicePoint, filename.str());
}

template<typename Field>
template<typename TField>
void SliceFieldPrinter<Field>::printSlice(const TField& field, int nAxis, float slicePoint, std::string filename)
{
    namespace vec = PMacc::math;
    using namespace vec::tools;
    typedef vec::CT::Size_t<TILE_WIDTH,TILE_HEIGHT,TILE_DEPTH> BlockDim;
        
    PMacc::GridController<3>& con = PMacc::Environment<3>::get().GridController();
    vec::Size_t<3> gpuDim = (vec::Size_t<3>)con.getGpuNodes();
    vec::Size_t<3> globalGridSize = gpuDim * field.size();
    int globalPlane = globalGridSize[nAxis] * slicePoint;
    int localPlane = globalPlane % field.size()[nAxis];
    int gpuPlane = globalPlane / field.size()[nAxis];
        
    vec::Int<3> nVector(0);
    nVector[nAxis] = 1;
    
    zone::SphericZone<3> gpuGatheringZone(vec::Size_t<3>(gpuDim.x(), gpuDim.y(), gpuDim.z()),
                                              nVector * gpuPlane);
    gpuGatheringZone.size[nAxis] = 1;
        
    algorithm::mpi::Gather<3> gather(gpuGatheringZone);
    if(!gather.participate()) return;
    
    using namespace lambda;
    vec::UInt<3> twistedVector((nAxis+1)%3, (nAxis+2)%3, nAxis);
    
    float_X SI = UNIT_EFIELD;
    if(Field::getName() == FieldB::getName())
        SI = UNIT_BFIELD;

    algorithm::kernel::Foreach<vec::CT::UInt<4,4,1> >()(
        dBuffer->zone(), dBuffer->origin(), 
        cursor::tools::slice(field.originCustomAxes(twistedVector)(0,0,localPlane)),
        _1 = _2 * SI);
    
    container::HostBuffer<float3_X, 2> hBuffer(dBuffer->size());
    hBuffer = *dBuffer;
        
    container::HostBuffer<float3_X, 2> globalBuffer(hBuffer.size() * gpuDim.shrink<2>((nAxis+1)%3));
    gather(globalBuffer, hBuffer, nAxis);
    if(!gather.root()) return;
    std::ofstream file(filename.c_str());
    file << globalBuffer;
}

}
