/**
 * Copyright 2013-2015 Axel Huebl, Rene Widera, Benjamin Worpitz
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
#include "simulationControl/MovingWindow.hpp"
#include <vector>
#include <list>


#include <cassert>

#include <stdexcept>


namespace picongpu
{
    using namespace PMacc;


    namespace po = boost::program_options;

    template<class VisClass>
    class PngPlugin : public ILightweightPlugin
    {
    public:

        typedef VisClass VisType;
        typedef std::list<VisType*> VisPointerList;

        PngPlugin() :
        analyzerName("PngPlugin: create png's of a species and fields"),
        analyzerPrefix(VisType::FrameType::getName() + "_" + VisClass::CreatorType::getName()),
        cellDescription(NULL)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~PngPlugin()
        {

        }

        std::string pluginGetName() const
        {
            return analyzerName;
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            desc.add_options()
                    ((analyzerPrefix + ".period").c_str(), po::value<std::vector<uint32_t> > (&notifyFrequencys)->multitoken(), "enable data output [for each n-th step]")
                    ((analyzerPrefix + ".axis").c_str(), po::value<std::vector<std::string > > (&axis)->multitoken(), "axis which are shown [valid values x,y,z] example: yz")
                    ((analyzerPrefix + ".slicePoint").c_str(), po::value<std::vector<float_32> > (&slicePoints)->multitoken(), "value range: 0 <= x <= 1 , point of the slice")
                    ((analyzerPrefix + ".folder").c_str(), po::value<std::vector<std::string> > (&folders)->multitoken(), "folder for output files");
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
                    0 != axis.size())
                {
                    for (int i = 0; i < (int) slicePoints.size(); ++i) /*!\todo: use vactor with max elements*/
                    {
                        uint32_t frequ = getValue(notifyFrequencys, i);
                        if (frequ != 0)
                        {

                            if (getValue(axis, i).length() == 2u)
                            {
                                std::stringstream o_slicePoint;
                                o_slicePoint << getValue(slicePoints, i);
                                /*add default value for folder*/
                                if (folders.empty())
                                {
                                    folders.push_back(std::string("."));
                                }
                                std::string filename(analyzerPrefix + "_" + getValue(axis, i) + "_" + o_slicePoint.str());
                                typename VisType::CreatorType pngCreator(filename, getValue(folders, i));
                                /** \todo rename me: transpose is the wrong name `swivel` is better
                                 *
                                 * `transpose` is used to map components from one vector to an other, in any order
                                 *
                                 * example: transpose[2,1] means: use x and z from an other vector
                                 */
                                DataSpace<DIM2 > transpose(
                                                           charToAxisNumber(getValue(axis, i)[0]),
                                                           charToAxisNumber(getValue(axis, i)[1])
                                                           );
                                /* if simulation run in 2D ignore all xz, yz slices (we had no z direction)*/
                                const bool isAllowed2DSlice= (simDim==DIM3) || (transpose.x()!=2 && transpose.y()!=2);
                                const bool isSlidingWindowActive=MovingWindow::getInstance().isSlidingWindowActive();
                                /* if sliding window is active we are not allowed to create pngs from xz slice
                                 * This means one dimension in transpose must contain 1 (y direction)
                                 */
                                const bool isAllowedMovingWindowSlice=!isSlidingWindowActive ||
                                                                      (transpose.x()==1 || transpose.y()==1);
                                if( isAllowed2DSlice && isAllowedMovingWindowSlice )
                                {
                                    VisType* tmp = new VisType(analyzerName, pngCreator, frequ, transpose, getValue(slicePoints, i));
                                    visIO.push_back(tmp);
                                    tmp->setMappingDescription(cellDescription);
                                    tmp->init();
                                }
                                else
                                {
                                    if(!isAllowedMovingWindowSlice)
                                        std::cerr << "[WARNING] You are running a simulation with moving window: png output along the axis "<<
                                                 getValue(axis, i) << " will be ignored" << std::endl;
                                    if(!isAllowed2DSlice)
                                        std::cerr << "[WARNING] You are running a 2D simulation: png output along the axis "<<
                                                 getValue(axis, i) << " will be ignored" << std::endl;
                                }
                            }
                            else
                                throw std::runtime_error((std::string("[Png Plugin] wrong charecter count in axis: ") + getValue(axis, i)).c_str());
                        }
                    }
                }
                else
                {
                    throw std::runtime_error("[Png Plugin] One parameter is missing");
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
                throw std::runtime_error("[Png Plugin] getValue is used with a parameter set with no parameters (count is 0)");
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
        std::vector<std::string> folders;
        std::vector<std::string> axis;
        VisPointerList visIO;

        MappingDesc* cellDescription;

    };

}//namespace

