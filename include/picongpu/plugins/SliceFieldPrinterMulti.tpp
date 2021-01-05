/* Copyright 2013-2021 Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch
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

#include <pmacc/math/vector/Int.hpp>
#include <pmacc/math/vector/Float.hpp>
#include <pmacc/math/vector/Size_t.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include <pmacc/math/Vector.hpp>
#include <pmacc/cuSTL/algorithm/mpi/Gather.hpp>
#include <pmacc/cuSTL/container/DeviceBuffer.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/cuSTL/cursor/tools/slice.hpp>
#include <pmacc/cuSTL/algorithm/kernel/Foreach.hpp>
#include <pmacc/cuSTL/algorithm/host/Foreach.hpp>
#include "SliceFieldPrinterMulti.hpp"
#include <sstream>

namespace picongpu
{
    template<typename Field>
    SliceFieldPrinterMulti<Field>::SliceFieldPrinterMulti()
        : name("SliceFieldPrinter: prints a slice of a field")
        , prefix(Field::getName() + std::string("_slice"))
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    template<typename Field>
    void SliceFieldPrinterMulti<Field>::pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()(
            (this->prefix + ".period").c_str(),
            po::value<std::vector<std::string>>(&this->notifyPeriod)->multitoken(),
            "notify period");
        desc.add_options()(
            (this->prefix + ".fileName").c_str(),
            po::value<std::vector<std::string>>(&this->fileName)->multitoken(),
            "file name to store slices in");
        desc.add_options()(
            (this->prefix + ".plane").c_str(),
            po::value<std::vector<int>>(&this->plane)->multitoken(),
            "specifies the axis which stands on the cutting plane (0,1,2)");
        desc.add_options()(
            (this->prefix + ".slicePoint").c_str(),
            po::value<std::vector<float_X>>(&this->slicePoint)->multitoken(),
            "slice point 0.0 <= x <= 1.0");
    }

    template<typename Field>
    std::string SliceFieldPrinterMulti<Field>::pluginGetName() const
    {
        return this->name;
    }

    template<typename Field>
    void SliceFieldPrinterMulti<Field>::pluginLoad()
    {
        this->childs.resize(this->notifyPeriod.size());
        for(uint32_t i = 0; i < this->childs.size(); i++)
        {
            this->childs[i].setMappingDescription(this->cellDescription);
            this->childs[i].notifyPeriod = this->notifyPeriod[i];
            this->childs[i].fileName = this->fileName[i];
            this->childs[i].plane = this->plane[i];
            this->childs[i].slicePoint = this->slicePoint[i];
            this->childs[i].pluginLoad();
        }
    }

    template<typename Field>
    void SliceFieldPrinterMulti<Field>::pluginUnload()
    {
        for(uint32_t i = 0; i < this->childs.size(); i++)
            this->childs[i].pluginUnload();
    }

    template<typename Field>
    void SliceFieldPrinterMulti<Field>::setMappingDescription(MappingDesc* desc)
    {
        this->cellDescription = desc;
    }

} // namespace picongpu
