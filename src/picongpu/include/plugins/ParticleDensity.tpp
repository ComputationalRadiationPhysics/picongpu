/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
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
#include "cuSTL/algorithm/kernel/Reduce.hpp"
#include "cuSTL/algorithm/kernel/ForeachBlock.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/cursor/tools/slice.hpp"
#include "lambda/Expression.hpp"
#include "cuSTL/container/PNGBuffer.hpp"
#include <sstream>
#include "particles/access/Cell2Particle.hpp"
#include "cuSTL/cursor/MultiIndexCursor.hpp"
#include <fstream>

namespace picongpu
{
namespace heiko
{
    
template<typename BlockDim>
struct ParticleDensityKernel
{
    typedef void result_type;

    int planeDir;
    int localPlane;
    ParticleDensityKernel() {}
    ParticleDensityKernel(int planeDir, int localPlane)
    : planeDir(planeDir), localPlane(localPlane) {}

    template<typename FramePtr, typename Field>
    DINLINE void operator()(FramePtr frame, uint16_t particleID, Field field, const ::PMacc::math::Int<3>& blockCellIdx)
    {
        lcellId_t linearCellIdx = (*frame)[particleID][localCellIdx_];
        ::PMacc::math::Int<3> cellIdx(linearCellIdx % BlockDim::x::value,
                               (linearCellIdx / BlockDim::x::value) % BlockDim::x::value,
                               linearCellIdx / (BlockDim::x::value * BlockDim::y::value));
        if(cellIdx[planeDir] != localPlane) return;

        ::PMacc::math::Int<3> globalCellIdx = blockCellIdx - (PMacc::math::Int<3>)BlockDim().vec() + cellIdx;
        /// \warn reduce a normalized float_X with particleAccess::Weight() / NUM_EL_PER_PARTICLE
        ///       to avoid overflows for heavy weightings
        ///
        atomicAdd(&(*field(globalCellIdx.shrink<2>((planeDir+1)%3))), (int)(*frame)[particleID][weighting_]);
    }
};

template<typename ParticlesType>
ParticleDensity<ParticlesType>::ParticleDensity(std::string name, std::string prefix)
    : name(name), prefix(prefix)
{
    ModuleConnector::getInstance().registerModule(this);
}

template<typename ParticlesType>
void ParticleDensity<ParticlesType>::moduleRegisterHelp(po::options_description& desc)
{
    desc.add_options()
        ((this->prefix + "_frequency").c_str(),
        po::value<uint32_t > (&this->notifyFrequency)->default_value(0), "notify frequency");
    desc.add_options()
        ((this->prefix + "_factor").c_str(),
        po::value<float_X> (&this->factor)->default_value(float_X(1.0)), "factor");
    desc.add_options()
        ((this->prefix + "_plane").c_str(),
        po::value<int> (&this->plane)->default_value(2), "specifies the axis which stands on the cutting plane (0,1,2)");
    desc.add_options()
        ((this->prefix + "_slicePoint").c_str(),
        po::value<float_X> (&this->slicePoint)->default_value(float_X(0.5)), "slice point 0.0 <= x <= 1.0");
}

template<typename ParticlesType>
std::string ParticleDensity<ParticlesType>::moduleGetName() const {return this->name;}

template<typename ParticlesType>
void ParticleDensity<ParticlesType>::moduleLoad()
{
    DataConnector::getInstance().registerObserver(this, this->notifyFrequency);
}

template<typename ParticlesType>
void ParticleDensity<ParticlesType>::moduleUnload(){}

template<typename ParticlesType>
void ParticleDensity<ParticlesType>::notify(uint32_t currentStep)
{
    DataConnector &dc = DataConnector::getInstance();
    this->particles = &(dc.getData<ParticlesType > ((uint32_t) ParticlesType::FrameType::CommunicationTag, true));
    

    namespace vec = ::PMacc::math;
    typedef vec::CT::Size_t<TILE_WIDTH, TILE_HEIGHT, TILE_DEPTH> BlockDim;
    container::PseudoBuffer<float3_X, 3> fieldE
        (dc.getData<FieldE > (FIELD_E, true).getGridBuffer().getDeviceBuffer());
    zone::SphericZone<3> coreBorderZone(fieldE.zone().size - (size_t)2*BlockDim().vec(),
                                        fieldE.zone().offset + (vec::Int<3>)BlockDim().vec());

    container::DeviceBuffer<int, 2> density(coreBorderZone.size.shrink<2>((plane+1)%3));
    density.assign(0);
    
    PMacc::GridController<3>& con = PMacc::GridController<3>::getInstance();
    vec::Size_t<3> gpuDim = (vec::Size_t<3>)con.getGpuNodes();
    vec::Size_t<3> globalGridSize = gpuDim * coreBorderZone.size;
    
    int globalPlanePos = globalGridSize[plane] * this->slicePoint;
    int localPlanePos = globalPlanePos % coreBorderZone.size[plane];
    int gpuPos = globalPlanePos / coreBorderZone.size[plane];
    int superCell = localPlanePos / BlockDim().vec()[plane];
    int cellWithinSuperCell = localPlanePos % BlockDim().vec()[plane];
    vec::Size_t<3> planeVec(0); planeVec[plane] = 1;
    vec::Size_t<3> orthoPlaneVec(1); orthoPlaneVec[plane] = 0;
    
    zone::SphericZone<3> gpuGatheringZone(orthoPlaneVec * gpuDim + planeVec * vec::Size_t<3>(1,1,1), 
                                          (vec::Int<3>)planeVec * gpuPos);
    algorithm::mpi::Gather<3> gather(gpuGatheringZone);
    if(!gather.participate()) return;
    
    zone::SphericZone<3> superCellSliceZone(orthoPlaneVec * coreBorderZone.size + planeVec * (vec::Size_t<3>)BlockDim().vec(),
                                            coreBorderZone.offset + (vec::Int<3>)planeVec * superCell * (int)BlockDim().vec()[plane]);
    
    using namespace lambda;
    algorithm::kernel::ForeachBlock<BlockDim>()
        (superCellSliceZone, cursor::make_MultiIndexCursor<3>(),
        expr(particleAccess::Cell2Particle<BlockDim>())
            (this->particles->getDeviceParticlesBox(),
            _1, ParticleDensityKernel<BlockDim>(plane, cellWithinSuperCell), density.origin(), _1));
            
    container::HostBuffer<int, 2> density_host(density.size());
    density_host = density;
    container::HostBuffer<int, 2> globalDensity(density.size() * gpuDim.shrink<2>((plane+1)%3));
    gather(globalDensity, density_host, plane);
    if(!gather.root()) return;
        
    std::stringstream filename;
    filename << "density_" << currentStep << ".dat";
    std::ofstream file(filename.str().c_str());
    /// \warn reduce a normalized float_X with particleAccess::Weight() / NUM_EL_PER_PARTICLE
    ///       to avoid overflows for heavy weightings
    ///
    /// \warn multiply normalized float_X "count" with NUM_EL_PER_PARTICLE here (as double!)
    ///
    file << globalDensity;
}

}
}
