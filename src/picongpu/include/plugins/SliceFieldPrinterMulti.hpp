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

#include "plugins/SliceFieldPrinter.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "math/vector/Float.hpp"

namespace picongpu
{

using namespace PMacc;

namespace po = boost::program_options;

#include <string>

template<typename Field>
class SliceFieldPrinterMulti : public ILightweightPlugin
{
private:
    std::string name;
    std::string prefix;
    std::vector<uint32_t> notifyFrequency;
    std::vector<std::string> fieldName;
    std::vector<int> plane;
    std::vector<float_X> slicePoint;
    MappingDesc *cellDescription;
    std::vector<SliceFieldPrinter<Field> > childs;

    void pluginLoad();
    void pluginUnload();

public:
    SliceFieldPrinterMulti(std::string name, std::string prefix);
    virtual ~SliceFieldPrinterMulti() {}

    void notify(uint32_t) {}
    void setMappingDescription(MappingDesc* desc);
    void pluginRegisterHelp(po::options_description& desc);
    std::string pluginGetName() const;
};

}

#include "SliceFieldPrinterMulti.tpp"

