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

TouchParticles::TouchParticles(std::string name, std::string prefix)
    : name(name), prefix(prefix)
{
    ModuleConnector::getInstance().registerModule(this);
}

void TouchParticles::moduleRegisterHelp(po::options_description& desc)
{
    desc.add_options()
        ((this->prefix + "_frequency").c_str(),
        po::value<uint32_t > (&this->notifyFrequency)->default_value(0), "notifyFrequency");
}

std::string TouchParticles::moduleGetName() const {return this->name;}

void TouchParticles::moduleLoad()
{
    DataConnector::getInstance().registerObserver(this, this->notifyFrequency);
}
void TouchParticles::moduleUnload(){}

void TouchParticles::notify(uint32_t currentStep)
{

    namespace math = PMacc::math;
    using namespace math;
    typedef math::CT::Size_t<TILE_WIDTH,TILE_HEIGHT,TILE_DEPTH> SuperCellSize;
    
    const PMacc::math::Int<3> guardCells = SuperCellSize().vec() * size_t(GUARD_SIZE);
    const PMacc::math::Size_t<3> coreBorderSuperCells( this->cellDescription->getGridSuperCells() - 2*int(GUARD_SIZE) );
    const PMacc::math::Size_t<3> coreBorderCells( coreBorderSuperCells * SuperCellSize().vec());
    
    zone::SphericZone<SIMDIM> coreBorderZone( coreBorderCells, guardCells );

       algorithm::kernel::ForeachBlock<SuperCellSize> forEachSuperCell;

       FunctorBlock<Species, SuperCellSize> functorBlock(
           this->particles->getDeviceParticlesBox());

       forEachSuperCell( /* area to work on */
                         coreBorderZone,
                         /* data below - passed to functor operator() */
                         cursor::make_MultiIndexCursor<3>(),
                         functorBlock
                       );
}

}
