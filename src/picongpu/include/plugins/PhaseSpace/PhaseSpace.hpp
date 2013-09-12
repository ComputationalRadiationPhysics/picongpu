/**
 * Copyright 2013 Axel Huebl, Heiko Burau
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

#include "dataManagement/ISimulationIO.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "math/vector/compile-time/Size_t.hpp"

#include <boost/program_options/options_description.hpp>

#include <string>
#include <utility>

namespace picongpu
{
    using namespace PMacc;
    namespace po = boost::program_options;

    template<class AssignmentFunction, class Species>
    class PhaseSpace : public ISimulationIO
    {
    private:
        std::string name;
        std::string prefix;
        uint32_t notifyPeriod;
        Species *particles;
        MappingDesc *cellDescription;
        
        // plot to create: e.g. x, py from element_coordinate/momentum
        std::pair<uint32_t, uint32_t> axis_element;
        std::pair<float_X, float_X> axis_p_range;
        uint32_t r_bins;
        
        static const uint32_t p_bins = 1024;
        container::DeviceBuffer<float_X, 2>* dBuffer;
        
        typedef PhaseSpace<AssignmentFunction, Species> This;
        typedef PMacc::math::CT::Size_t<TILE_WIDTH, TILE_HEIGHT, TILE_DEPTH> SuperCellSize;
        
    public:
        enum element_coordinate
        { x = 0u, y = 1u, z = 2u };
        enum element_momentum
        { px = 0u, py = 1u, pz = 2u };
        
        PhaseSpace( const std::string _name,
                     const std::string _prefix,
                     const uint32_t _notifyPeriod,
                     const std::pair<float_X, float_X>& _p_range,
                     const std::pair<uint32_t, uint32_t>& _element );
        virtual ~PhaseSpace(){}

        void notify( uint32_t currentStep );
        template<uint32_t Direction>
        void calcPhaseSpace( );
        void setMappingDescription( MappingDesc* cellDescription);

        void moduleLoad();
        void moduleUnload();
    };

}

#include "PhaseSpace.tpp"
