///**
//* Copyright 2013 Axel Huebl, Heiko Burau
//*
//* This file is part of PIConGPU.
//*
//* PIConGPU is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//*
//* PIConGPU is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//* GNU General Public License for more details.
//*
//* You should have received a copy of the GNU General Public License
//* along with PIConGPU.
//* If not, see <http://www.gnu.org/licenses/>.
//*/
//
//#pragma once
//
//#include <utility>
//
//#include "cuSTL/cursor/MultiIndexCursor.hpp"
//#include "cuSTL/algorithm/kernel/Foreach.hpp"
//#include "cuSTL/algorithm/kernel/ForeachBlock.hpp"
//#include "math/vector/Int.hpp"
//#include "particles/access/Cell2Particle.hpp"
//
//#include "PhaseSpace.hpp"
//
//namespace picongpu
//{
//    using namespace PMacc;
//
//    /** Atomic Add Functor
//*
//* \tparam Type of the values to perform a atomicAdd on
//*/
//    template<typename Type>
//    struct FunctorAtomicAdd
//    {
//        typedef void result_type;
//
//        DINLINE void operator()( Type& dest, const Type src )
//        {
//            atomicAddWrapper( &dest, src );
//        }
//    };
//
//    /** Functor called for each particle
//*
//* Every particle in a frame of particles will end up here.
//* We calculate where in space the owning (super) cell lives and
//* add the particle to the shared memory buffer for that phase
//* space snipped the super cell contributes to.
//*
//* \tparam r_dir spatial direction of the phase space (0,1,2)
//* \tparam p_bins number of bins in momentum space \see PhaseSpace.hpp
//* \tparam SuperCellSize how many cells form a super cell \see memory.param
//*/
//    template<typename SuperCellSize>
//    struct FunctorParticle
//    {
//        typedef void result_type;
//
//        template<typename FramePtr >
//        DINLINE void operator()( FramePtr frame,
//                                 uint16_t particleID )
//        {
//            PMACC_AUTO( particle, (*frame)[particleID] ); 
//            const float_X mom_i = particle[momentum_][el_p];
//
//            /* cell id in this block */
//            const int linearCellIdx = particle[localCellIdx_];
//            const PMacc::math::Int<3> cellIdx(
//                linearCellIdx % SuperCellSize::x::value,
//                (linearCellIdx % (SuperCellSize::x::value * SuperCellSize::y::value)) / SuperCellSize::x::value,
//                linearCellIdx / (SuperCellSize::x::value * SuperCellSize::y::value) );
//
//            const float_X weighting = particle[weighting_];
//            const float_X charge = particle.getCharge( weighting );
//            const float_PS particleChargeDensity =
//              typeCast<float_PS>( charge / ( CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH ) );
//            if(linearCellIdx == 0)
//            {
//                    printf(charge);
//            }
//        }
//    };
//
//    /** Functor to Run For Each SuperCell
//*
//* This functor is called for each super cell, preparing a shared memory
//* buffer with a supercell-local (spatial) snippet of the phase space.
//* Afterwards all blocks reduce their data to a combined gpu-local (spatial)
//* snippet of the phase space in global memory.
//*
//* \tparam Species the particle species to create the phase space for
//* \tparam SuperCellSize how many cells form a super cell \see memory.param
//* \tparam float_PS type for each bin in the phase space
//* \tparam p_bins number of bins in momentum space \see PhaseSpace.hpp
//* \tparam r_dir spatial direction of the phase space (0,1,2)
//*/
//    template<typename Species, typename SuperCellSize>
//    struct FunctorBlock
//    {
//        typedef void result_type;
//
//        typedef typename Species::ParticlesBoxType TParticlesBox;
//
//        TParticlesBox particlesBox;
//
//
//        FunctorBlock( const TParticlesBox& pb ) :
//        particlesBox(pb)
//        {}
//
//        /** Called for the first cell of each block #-of-cells-in-block times
//*/
//        DINLINE void operator()( const PMacc::math::Int<3>& indexBlockOffset )
//        {
//            const PMacc::math::Int<3> indexInBlock( threadIdx.x, threadIdx.y, threadIdx.z );
//            const PMacc::math::Int<3> indexGlobal = indexBlockOffset + indexInBlock;
//
//
//            FunctorParticle<SuperCellSize> functorParticle;
//            particleAccess::Cell2Particle<SuperCellSize> forEachParticleInCell;
//            forEachParticleInCell( /* mandatory params */
//                                   particlesBox, indexGlobal, functorParticle,
//                                   /* optional params */
//                                 );
//           
//        }
//    };
//
//} // namespace picongpu