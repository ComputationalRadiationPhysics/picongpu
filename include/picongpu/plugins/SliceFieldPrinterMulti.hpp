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

#include "picongpu/plugins/SliceFieldPrinter.hpp"
#include <pmacc/cuSTL/container/DeviceBuffer.hpp>
#include <pmacc/math/vector/Float.hpp>

#include <string>


namespace picongpu
{
    using namespace pmacc;
    namespace po = boost::program_options;

    template<typename Field>
    class SliceFieldPrinterMulti : public ILightweightPlugin
    {
    private:
        std::string name;
        std::string prefix;
        std::vector<std::string> notifyPeriod;
        std::vector<std::string> fileName;
        std::vector<int> plane;
        std::vector<float_X> slicePoint;
        MappingDesc* cellDescription;
        std::vector<SliceFieldPrinter<Field>> childs;

        void pluginLoad();
        void pluginUnload();

    public:
        SliceFieldPrinterMulti();
        virtual ~SliceFieldPrinterMulti()
        {
        }

        void notify(uint32_t)
        {
        }
        void setMappingDescription(MappingDesc* desc);
        void pluginRegisterHelp(po::options_description& desc);
        std::string pluginGetName() const;
    };

} // namespace picongpu

#include "SliceFieldPrinterMulti.tpp"
