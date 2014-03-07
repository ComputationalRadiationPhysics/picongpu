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
 
#pragma once

#include "cuSTL/container/DeviceBuffer.hpp"
#include "math/vector/Float.hpp"

namespace picongpu
{
    
using namespace PMacc;

namespace po = boost::program_options;

#include <string>

template<typename Field>
class SliceFieldPrinterMulti;

template<typename Field>
class SliceFieldPrinter : public ISimulationIO
{
private:
    uint32_t notifyFrequency;
    std::string fieldName;
    int plane;
    float_X slicePoint;
    MappingDesc *cellDescription;
    container::DeviceBuffer<float3_X, 2>* dBuffer;
        
    void moduleLoad();
    void moduleUnload();
    
    template<typename TField>
    void printSlice(const TField& field, int nAxis, float slicePoint, std::string filename);
    
    friend class SliceFieldPrinterMulti<Field>;
public:
    void notify(uint32_t currentStep);
    void setMappingDescription(MappingDesc* desc) {this->cellDescription = desc;}
};

}

#include "SliceFieldPrinter.tpp"
