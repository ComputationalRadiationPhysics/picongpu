/**
 * Copyright 2013 Heiko Burau, René Widera
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
#include "SliceFieldPrinterMulti.hpp"
#include <math/vector/tools/twistVectorAxes.hpp>
#include <sstream>

namespace picongpu
{

template<typename Field, int FieldId>
SliceFieldPrinterMulti<Field, FieldId>::SliceFieldPrinterMulti(std::string name, std::string prefix)
    : name(name), prefix(prefix)
{
    ModuleConnector::getInstance().registerModule(this);
}

template<typename Field, int FieldId>
void SliceFieldPrinterMulti<Field, FieldId>::moduleRegisterHelp(po::options_description& desc)
{
    desc.add_options()
        ((this->prefix + "_frequency").c_str(),
        po::value<std::vector<uint32_t> > (&this->notifyFrequency)->multitoken(), "notify frequency");
    desc.add_options()
        ((this->prefix + "_fieldName").c_str(),
        po::value<std::vector<std::string> > (&this->fieldName)->multitoken(), "field Name");
    desc.add_options()
        ((this->prefix + "_plane").c_str(),
        po::value<std::vector<int> > (&this->plane)->multitoken(), "specifies the axis which stands on the cutting plane (0,1,2)");
    desc.add_options()
        ((this->prefix + "_slicePoint").c_str(),
        po::value<std::vector<float_X> > (&this->slicePoint)->multitoken(), "slice point 0.0 <= x <= 1.0");
}

template<typename Field, int FieldId>
std::string SliceFieldPrinterMulti<Field, FieldId>::moduleGetName() const {return this->name;}

template<typename Field, int FieldId>
void SliceFieldPrinterMulti<Field, FieldId>::moduleLoad()
{
    this->childs.resize(this->notifyFrequency.size());
    for(int i = 0; i < this->childs.size(); i++)
    {
        this->childs[i].setMappingDescription(this->cellDescription);
        this->childs[i].notifyFrequency = this->notifyFrequency[i];
        this->childs[i].fieldName = this->fieldName[i];
        this->childs[i].plane = this->plane[i];
        this->childs[i].slicePoint = this->slicePoint[i];
        this->childs[i].moduleLoad();
    }
}

template<typename Field, int FieldId>
void SliceFieldPrinterMulti<Field, FieldId>::moduleUnload()
{
    for(int i = 0; i < this->childs.size(); i++)
        this->childs[i].moduleUnload();
}

template<typename Field, int FieldId>
void SliceFieldPrinterMulti<Field, FieldId>::setMappingDescription(MappingDesc* desc)
{
    this->cellDescription = desc;
}

}
