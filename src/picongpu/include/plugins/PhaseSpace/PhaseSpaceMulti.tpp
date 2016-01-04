/** Copyright 2013-2016 Axel Huebl
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
    PhaseSpaceMulti<AssignmentFunction, Species>::PhaseSpaceMulti( ) :
        name("PhaseSpaceMulti: create phase space of a species"),
        prefix(Species::FrameType::getName() + std::string("_phaseSpace")),
        numChildren(0u),
        cellDescription(NULL)
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
              po::value<std::vector<float_X> > (&this->momentum_range_min)->multitoken(), "min range momentum [m_species c]")
            ((this->prefix + ".max").c_str(),
              po::value<std::vector<float_X> > (&this->momentum_range_max)->multitoken(), "max range momentum [m_species c]");
    }

    template<class AssignmentFunction, class Species>
    void PhaseSpaceMulti<AssignmentFunction, Species>::pluginLoad( )
    {
        this->numChildren = this->notifyPeriod.size();

        if( this->numChildren == 0 )
            return;

        this->children.reserve( this->numChildren );
        for(uint32_t i = 0; i < this->numChildren; i++)
        {
            /* unit is m_species c - we use the typical weighting already since
             * it scales linear in momentum and we avoid over- & under-flows
             * during momentum-binning while staying at a
             * "typical macro particle momentum scale"
             *
             * we correct that during the output of the pAxis range
             * (linearly shrinking the single particle scale again)
             */
            float_X pRangeMakro_unit = float_X( frame::getMass<typename Species::FrameType>()*
                                                particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE *
                                                SPEED_OF_LIGHT );
            std::pair<float_X, float_X> new_p_range(
                    this->momentum_range_min.at(i) * pRangeMakro_unit,
                    this->momentum_range_max.at(i) * pRangeMakro_unit );
            /* String to Enum conversion */
            uint32_t el_space;
            if( this->element_space.at(i) == "x" )
                el_space = AxisDescription::x;
            else if( this->element_space.at(i) == "y" )
                el_space = AxisDescription::y;
            else if( this->element_space.at(i) == "z" )
                el_space = AxisDescription::z;
            else
                throw PluginException("[Plugin] [" + this->prefix + "] space must be x, y or z" );

            uint32_t el_momentum = AxisDescription::px;
            if( this->element_momentum.at(i) == "px" )
                el_momentum = AxisDescription::px;
            else if( this->element_momentum.at(i) == "py" )
                el_momentum = AxisDescription::py;
            else if( this->element_momentum.at(i) == "pz" )
                el_momentum = AxisDescription::pz;
            else
                throw PluginException("[Plugin] [" + this->prefix + "] momentum must be px, py or pz" );

            AxisDescription new_elements;
            new_elements.momentum = el_momentum;
            new_elements.space = el_space;

            if( simDim == DIM2 && el_space == AxisDescription::z )
                std::cerr << "[Plugin] [" + this->prefix + "] Skip requested output for "
                          << this->element_space.at(i)
                          << this->element_momentum.at(i)
                          << std::endl;
            else
            {
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
        }

        /** create dir */
        Environment<simDim>::get().Filesystem().createDirectoryWithPermissions("phaseSpace");
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
