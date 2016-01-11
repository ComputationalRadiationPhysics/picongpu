/**
 * Copyright 2013-2016 Axel Huebl
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
#include "plugins/ILightweightPlugin.hpp"

#include "plugins/PhaseSpace/AxisDescription.hpp"
#include "plugins/PhaseSpace/PhaseSpace.hpp"

#include <boost/program_options/options_description.hpp>

#include <string>
#include <vector>
#include <utility>

namespace picongpu
{
    using namespace PMacc;
    namespace po = boost::program_options;

    template<class T_AssignmentFunction, class T_Species>
    class PhaseSpaceMulti : public ILightweightPlugin
    {
    public:
        typedef T_AssignmentFunction AssignmentFunction;
        typedef T_Species Species;
        typedef PhaseSpace<AssignmentFunction, Species> Child;

    private:
        std::string name;
        std::string prefix;

        std::vector<uint32_t> notifyPeriod;

        MappingDesc* cellDescription;

        std::vector<std::string> element_space;
        std::vector<std::string> element_momentum;
        std::vector<float_X> momentum_range_min;
        std::vector<float_X> momentum_range_max;

        /** plot to create: e.g. (py | x) from (momentum | spatial-component) */
        std::vector<AxisDescription > axis_element;
        /** range [pMin : pMax] in m_e c */
        std::vector<std::pair<float_X, float_X> > axis_p_range;

        std::vector<Child* > children;
        size_t numChildren;

        void pluginLoad();
        void pluginUnload();

    public:
        PhaseSpaceMulti( );
        virtual ~PhaseSpaceMulti() {}

        void notify( uint32_t ) {}
        void setMappingDescription( MappingDesc* cellDescription);
        void pluginRegisterHelp( po::options_description& desc );

        std::string pluginGetName() const { return this->name; }
    };

}

#include "PhaseSpaceMulti.tpp"
