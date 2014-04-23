/** Copyright 2013-2014 Axel Huebl
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

#include "PhaseSpaceMulti.hpp"

#include <sys/stat.h>

#pragma once

namespace picongpu
{
    using namespace PMacc;
    namespace po = boost::program_options;

    template<class AssignmentFunction, class Species>
    PhaseSpaceMulti<AssignmentFunction, Species>::PhaseSpaceMulti( const std::string _name,
                                                                   const std::string _prefix ) :
        name(_name), prefix(_prefix), numChildren(0u), cellDescription(NULL)
    {
        /* register our plugin during creation */
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpaceMulti<AssignmentFunction, Species>::pluginRegisterHelp( po::options_description& desc )
    {
        desc.add_options()
            ((this->prefix + ".period").c_str(),
              po::value<std::vector<uint32_t> > (&this->notifyPeriod)->multitoken(), "notify period")
            ((this->prefix + ".space").c_str(),
              po::value<std::vector<std::string> > (&this->element_space)->multitoken(), "spatial component (x, y, z)")
            ((this->prefix + ".momentum").c_str(),
              po::value<std::vector<std::string> > (&this->element_momentum)->multitoken(), "momentum component (px, py, pz)")
            ((this->prefix + ".min").c_str(),
              po::value<std::vector<float_X> > (&this->momentum_range_min)->multitoken(), "min range momentum [m_e c]")
            ((this->prefix + ".max").c_str(),
              po::value<std::vector<float_X> > (&this->momentum_range_max)->multitoken(), "max range momentum [m_e c]");
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpaceMulti<AssignmentFunction, Species>::pluginLoad( )
    {
        this->numChildren = this->notifyPeriod.size();

        this->children.reserve( this->numChildren );
        for(uint32_t i = 0; i < this->numChildren; i++)
        {
            /* unit is m_e c - we use the typical weighting already since it
             * scales linear in momentum and we avoid over- & under-flows
             * during momentum-binning while staying at a
             * "typical macro particle momentum scale"
             *
             * we correct that during the output of the pAxis range
             * (linearly shrinking the single particle scale again)
             */
            float_X unit_pRange = float_X( double(NUM_EL_PER_PARTICLE) *
                                           double(M_EL) * double(SPEED_OF_LIGHT) );
            std::pair<float_X, float_X> new_p_range( this->momentum_range_min.at(i) * unit_pRange,
                                                     this->momentum_range_max.at(i) * unit_pRange );
            /* String to Enum conversion */
            uint32_t el_space;
            if( this->element_space.at(i) == "x" )
                el_space = Child::x;
            else if( this->element_space.at(i) == "y" )
                el_space = Child::y;
            else if( this->element_space.at(i) == "z" )
                el_space = Child::z;
            else
                throw PluginException("[Plugin] [" + this->name + "] space must be x, y or z" );

            uint32_t el_momentum = Child::px;
            if( this->element_momentum.at(i) == "px" )
                el_momentum = Child::px;
            else if( this->element_momentum.at(i) == "py" )
                el_momentum = Child::py;
            else if( this->element_momentum.at(i) == "pz" )
                el_momentum = Child::pz;
            else
                throw PluginException("[Plugin] [" + this->name + "] momentum must be px, py or pz" );

            std::pair<uint32_t, uint32_t> new_elements( el_space, el_momentum );

            PhaseSpace<AssignmentFunction, Species>* newPS =
              new PhaseSpace<AssignmentFunction, Species>( this->name,
                                                           this->prefix,
                                                           this->notifyPeriod.at(i),
                                                           new_p_range,
                                                           new_elements );

            this->children.push_back( newPS );
            this->children.at(i)->setMappingDescription( this->cellDescription );
            this->children.at(i)->pluginLoad();
        }

        /** create dir */
        PMacc::GridController<simDim>& gc = PMacc::Environment<simDim>::get().GridController();
        if( gc.getGlobalRank() == 0 )
        {
            /** \todo make this a boost filesystem call */
            mkdir("phaseSpace", 0755);
        }
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpaceMulti<AssignmentFunction, Species>::pluginUnload( )
    {
        for(uint32_t i = 0; i < this->numChildren; i++)
        {
            this->children.at(i)->pluginUnload();
            __delete( this->children.at(i) );
        }
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpaceMulti<AssignmentFunction, Species>::setMappingDescription( MappingDesc* desc )
    {
        this->cellDescription = desc;
    }

} /* namespace picongpu */
