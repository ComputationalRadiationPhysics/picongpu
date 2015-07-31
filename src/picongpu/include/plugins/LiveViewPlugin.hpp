/**
 * Copyright 2013 Axel Huebl, Rene Widera
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

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"
#include "dimensions/DataSpace.hpp"

#include "simulation_classTypes.hpp"
#include "plugins/ILightweightPlugin.hpp"
#include <vector>
#include <list>
#include "plugins/output/images/Visualisation.hpp"
#include "plugins/output/images/LiveViewClient.hpp"

#include <cassert>

#include <stdexcept>


namespace picongpu
{
    using namespace PMacc;

    namespace po = boost::program_options;

    template<class ParticlesType>
    class LiveViewPlugin : public ILightweightPlugin
    {
    public:

        typedef Visualisation<ParticlesType, LiveViewClient> VisType;
        typedef std::list<VisType*> VisPointerList;

        LiveViewPlugin() :
        analyzerName("LiveViewPlugin: 2D (plane) insitu live visualisation of a species"),
        analyzerPrefix(ParticlesType::FrameType::getName() + std::string("_liveView")),
        cellDescription(NULL)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~LiveViewPlugin()
        {

        }

        std::string pluginGetName() const
        {
            return analyzerName;
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            desc.add_options()
                    ((analyzerPrefix + ".period").c_str(), po::value<std::vector<uint32_t> > (&notifyFrequencys)->multitoken(), "enable images/visualisation [for each n-th step]")
                    ((analyzerPrefix + ".ip").c_str(), po::value<std::vector<std::string > > (&ips)->multitoken(), "ip of server")
                    ((analyzerPrefix + ".port").c_str(), po::value<std::vector<std::string > > (&ports)->multitoken(), "port of server")
                    ((analyzerPrefix + ".axis").c_str(), po::value<std::vector<std::string > > (&axis)->multitoken(), "axis which are shown [valid values x,y,z] example: yz")
                    ((analyzerPrefix + ".slicePoint").c_str(), po::value<std::vector<float_32> > (&slicePoints)->multitoken(), "value range: 0 <= x <= 1 , point of the slice");
        }

        void setMappingDescription(MappingDesc *cellDescription)
        {
            this->cellDescription = cellDescription;
        }


    private:

        void pluginLoad()
        {

            if (0 != notifyFrequencys.size())
            {
                if (0 != slicePoints.size() &&
                    0 != ports.size() &&
                    0 != ips.size() &&
                    0 != axis.size())
                {
                    for (int i = 0; i < (int) ports.size(); ++i)
                    {
                        uint32_t frequ = getValue(notifyFrequencys, i);
                        if (frequ != 0)
                        {

                            if (getValue(axis, i).length() == 2u)
                            {
                                LiveViewClient liveViewClient(getValue(ips, i), getValue(ports, i));
                                DataSpace<DIM2 > transpose(
                                                           charToAxisNumber(getValue(axis, i)[0]),
                                                           charToAxisNumber(getValue(axis, i)[1])
                                                           );
                                VisType* tmp = new VisType(analyzerName, liveViewClient, frequ, transpose, getValue(slicePoints, i));
                                visIO.push_back(tmp);
                                tmp->setMappingDescription(cellDescription);
                                tmp->init();
                            }
                            else
                                throw std::runtime_error((std::string("[Live View] wrong charecter count in axis: ") + getValue(axis, i)).c_str());
                        }
                    }
                }
                else
                {
                    throw std::runtime_error("[Live View] One parameter is missing");
                }
            }
        }

        void pluginUnload()
        {
            for (typename VisPointerList::iterator iter = visIO.begin();
                 iter != visIO.end();
                 ++iter)
            {
                __delete(*iter);
            }
            visIO.clear();
        }

        void notify(uint32_t currentStep)
        {
            // nothing to do here
        }

        /*! Get value of the postition in a vector
         * @return value at id postition, if id >= size of vector last valid value is given back
         */
        template<class Vec>
        typename Vec::value_type getValue(Vec vec, size_t id)
        {
            if (vec.size() == 0)
                throw std::runtime_error("[Live View] getValue is used with a parameter set with no parameters (count is 0)");
            if (id >= vec.size())
            {
                return vec[vec.size() - 1];
            }
            return vec[id];
        }

        int charToAxisNumber(char c)
        {
            if (c == 'x')
                return 0;
            if (c == 'y')
                return 1;
            return 2;
        }


        std::string analyzerName;
        std::string analyzerPrefix;

        std::vector<uint32_t> notifyFrequencys;
        std::vector<float_32> slicePoints;
        std::vector<std::string> ips;
        std::vector<std::string> ports;
        std::vector<std::string> axis;
        VisPointerList visIO;

        MappingDesc* cellDescription;

    };

}//namespace

