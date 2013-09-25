/**
 * Copyright 2013 Axel Huebl
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

#include "simulation_defines.hpp"
#include "cuSTL/container/HostBuffer.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <utility>

#if (ENABLE_HDF5==1)
#include <splash.h>
#endif

namespace picongpu
{
    class DumpHBuffer
    {
    public:
        /** Dump the PhaseSpace host Buffer
         * 
         * \tparam Type the HBuffers element type
         * \tparam int the HBuffers dimension
         * \param hBuffer const reference to the hBuffer
         * \param axis_element plot to create: e.g. x, py from element_coordinate/momentum
         * \param unit sim unit of the buffer
         * \param
         */
        template<typename Type, int bufDim>
        void operator()( const PMacc::container::HostBuffer<Type, bufDim>& hBuffer,
                          const std::pair<uint32_t, uint32_t> axis_element,
                          const double unit,
                          const uint32_t currentStep,
                          MPI_Comm& mpiComm ) const;
    };

} // namespace picongpu


#if (ENABLE_HDF5==1)

#if (SPLASH_SUPPORTED_PARALLEL==1)
#include "DumpHBufferSplashP.tpp"

#elif (SPLASH_SUPPORTED_SERIAL==1)
#include "DumpHBufferSplashS.tpp"
#endif // (SPLASH_SUPPORTED_SERIAL==1)

#else // (ENABLE_HDF5==1)
#include "DumpHBufferTxt.tpp"

#endif // (ENABLE_HDF5==1)
