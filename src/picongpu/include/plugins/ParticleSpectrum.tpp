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
#include "cuSTL/algorithm/mpi/Reduce.hpp"
#include "cuSTL/algorithm/kernel/Reduce.hpp"
#include "cuSTL/algorithm/kernel/ForeachBlock.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/cursor/FunctorCursor.hpp"
#include "lambda/Expression.hpp"
#include <sstream>
#include "particles/access/Cell2Particle.hpp"
#include "cuSTL/cursor/MultiIndexCursor.hpp"
#include <fstream>
#include "simulation_defines.hpp"

namespace picongpu
{
    
struct Particle2Histrogram
{
    typedef void result_type;
    float_X minEnergy, maxEnergy;
    DINLINE Particle2Histrogram(float_X minEnergy, float_X maxEnergy)
        : minEnergy(minEnergy), maxEnergy(maxEnergy) {}
    
    template<typename FramePtr, typename Histogram>
    DINLINE void operator()(FramePtr particle, uint16_t particleID, Histogram histogram) const
    {
        float3_X mom = particle->getMomentum()[particleID];
        float_X weighting = particle->getWeighting()[particleID];
        const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        const float_X mass = M_EL;
        const float_X mass_reci = float_X(1.0) / mass;
        const float_X mass2 = mass * mass;
        
        mom /= weighting;
        float_X mom2 = abs2(mom);
        float_X energy;
        if(mom2 < 1.0e-3f * (4.0f * mass2 * c2)) // relative error of the taylor approx. smaller than 10^-3
            energy = mom2 * float_X(0.5) * mass_reci;
        else
            energy = sqrtf(mom2 * c2 + mass2 * c2*c2) - mass * c2;
        int bin = math::float2int_rd(Histogram::type::numBins * (energy - minEnergy) / (maxEnergy - minEnergy)) + 1;
        bin = max(0, bin); bin = min(Histogram::type::numBinsEx-1, bin);
        atomicAddWrapper(&(histogram.get().bin[bin]), weighting);
    }
};

template<typename BlockDim, int numBins>
struct ParticleSpectrumKernel
{
    typedef void result_type;
    float_X minEnergy, maxEnergy;
    ParticleSpectrumKernel() {}
    ParticleSpectrumKernel(float_X minEnergy, float_X maxEnergy)
        : minEnergy(minEnergy), maxEnergy(maxEnergy) {}
    
    template<typename ParticlesBox, typename Result>
    DINLINE void operator()(ParticlesBox pb, 
                             const ::PMacc::math::Int<3>& blockCellIdx,
                             Result result) const
    {
        uint16_t linearThreadIdx = threadIdx.z * BlockDim::x::value * BlockDim::y::value +
                               threadIdx.y * BlockDim::x::value + threadIdx.x;
        
        __shared__ detail::Histrogram<numBins> shHistogram;
        __syncthreads(); /*wait that all shared memory is initialised*/
        if(linearThreadIdx < numBins+2) shHistogram.bin[linearThreadIdx] = 0; //\todo: durch assign bzw. foreach ersetzen
        __syncthreads();
        
        particleAccess::Cell2Particle<BlockDim>()
            (pb, blockCellIdx, Particle2Histrogram(minEnergy, maxEnergy), ref(shHistogram));
            
        ::PMacc::math::Int<3> _blockIdx = blockCellIdx / (PMacc::math::Int<3>)(BlockDim().vec());
        __syncthreads();
        result[_blockIdx] = shHistogram;
    }
};

template<typename ParticlesType>
ParticleSpectrum<ParticlesType>::ParticleSpectrum(std::string name, std::string prefix)
    : name(name), prefix(prefix)
{
    Environment<>::get().PluginConnector().registerPlugin(this);
}

template<typename ParticlesType>
void ParticleSpectrum<ParticlesType>::pluginRegisterHelp(po::options_description& desc)
{
    desc.add_options()
        ((this->prefix + "_frequency").c_str(),
        po::value<uint32_t > (&this->notifyFrequency)->default_value(0), "notify frequency");
    desc.add_options()
        ((this->prefix + "_minEnergy").c_str(),
        po::value<float_X> (&this->minEnergy)->default_value(0.0), "min energy [keV]");
    desc.add_options()
        ((this->prefix + "_maxEnergy").c_str(),
        po::value<float_X> (&this->maxEnergy)->default_value(1000.0), "max energy [keV]");
    /*desc.add_options()
        ((this->prefix + "_numBins").c_str(),
        po::value<int> (&this->numBins)->default_value(10), "number of bins");*/
}

template<typename ParticlesType>
std::string ParticleSpectrum<ParticlesType>::pluginGetName() const {return this->name;}

template<typename ParticlesType>
void ParticleSpectrum<ParticlesType>::pluginLoad()
{
    Environment<>::get().DataConnector().registerObserver(this, this->notifyFrequency);
    
    this->minEnergy = this->minEnergy * UNITCONV_keV_to_Joule / UNIT_ENERGY;
    this->maxEnergy = this->maxEnergy * UNITCONV_keV_to_Joule / UNIT_ENERGY;
}

template<typename ParticlesType>
void ParticleSpectrum<ParticlesType>::pluginUnload(){}

struct GetBin
{
    typedef float& result_type;
    int idx;
    HDINLINE GetBin() {} //\todo: due to lambda lib
    HDINLINE GetBin(int idx) : idx(idx) {}
    
    template<typename Histogram>
    DINLINE float& operator()(Histogram& histogram)
    {
        return histogram.bin[this->idx];
    }
};

template<typename ParticlesType>
void ParticleSpectrum<ParticlesType>::notify(uint32_t)
{/*
    DataConnector &dc = Environment<>::get().DataConnector();
    this->particles = &(dc.getData<ParticlesType > (ParticlesType::FrameType::getName(), true));
    
    namespace vec = ::vector;
    using namespace vec;
    typedef vec::CT::Size_t<8,8,4> BlockDim;
    container::PseudoBuffer<float3_X, 3> fieldE
        (dc.getData<FieldE > (FieldE::getName(), true).getGridBuffer().getDeviceBuffer());
    zone::SphericZone<3> coreBorderZone(fieldE.zone().size - (size_t)2*BlockDim().vec(),
                                        fieldE.zone().offset + (vec::Int<3>)BlockDim().vec());
    
    container::DeviceBuffer<detail::Histrogram<numBins>, 3> spectrumBlocks(coreBorderZone.size / BlockDim().vec());
    
    using namespace lambda;
    algorithm::kernel::ForeachBlock<BlockDim>()
        (coreBorderZone, cursor::make_MultiIndexCursor<3>(),
        expr(ParticleSpectrumKernel<BlockDim, numBins>(minEnergy, maxEnergy))
            (this->particles->getDeviceParticlesBox(),
            _1, spectrumBlocks.origin()(-coreBorderZone.offset / (vec::Int<3>)BlockDim().vec())));
        
    container::DeviceBuffer<detail::Histrogram<numBins>, 1> spectrum(1);
    for(int i = 0; i < numBinsEx; i++)
    {
        algorithm::kernel::Reduce<vec::CT::Int<1,1,1> >()
            (cursor::make_FunctorCursor(spectrum.origin(), GetBin(i)),
             spectrumBlocks.zone(),
             cursor::make_FunctorCursor(spectrumBlocks.origin(), GetBin(i)),
             _1 + _2);
    }
    
    container::HostBuffer<detail::Histrogram<numBins>, 1> spectrumHost(1);
    spectrumHost = spectrum;
    container::HostBuffer<float, 1> spectrumMPI(numBinsEx), globalSpectrum(numBinsEx);
    for(size_t i = 0; i < numBinsEx; i++) spectrumMPI.origin()[(int)i] = (*spectrumHost.origin()).bin[i];
    
    PMacc::GridController<3>& con = PMacc::Environment<3>::get().GridController();
    vec::Size_t<3> gpuDim = (vec::Size_t<3>)con.getGpuNodes();
    zone::SphericZone<3> gpuReducingZone(gpuDim);
    algorithm::mpi::Reduce<3> reduce(gpuReducingZone);
    reduce(globalSpectrum, spectrumMPI, MPI_FLOAT, MPI_SUM);
    if(!reduce.root()) return;
    
    std::stringstream filename;
    filename << "spectrum_" << currentStep << ".dat";
    std::ofstream file(filename.str().c_str());
    file << globalSpectrum;*/
}

}
