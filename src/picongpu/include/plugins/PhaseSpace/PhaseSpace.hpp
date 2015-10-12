/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau
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

#include <mpi.h>

#include "simulation_defines.hpp"
#include "communication/manager_common.h"
#include "pluginSystem/INotify.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/algorithm/mpi/Reduce.hpp"
#include "math/Vector.hpp"

#include "plugins/PhaseSpace/AxisDescription.hpp"

#include <string>
#include <utility>

#include <boost/mpl/min_max.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/accumulate.hpp>

namespace picongpu
{
    using namespace PMacc;
    namespace po = boost::program_options;

    template<class T_AssignmentFunction, class T_Species>
    class PhaseSpace : public INotify
    {
    public:
        typedef T_AssignmentFunction AssignmentFunction;
        typedef T_Species Species;

    private:
        std::string name;
        std::string prefix;
        uint32_t notifyPeriod;
        Species *particles;
        MappingDesc *cellDescription;

        /** plot to create: e.g. py, x from element_coordinate/momentum */
        AxisDescription axis_element;
        /** range [pMin : pMax] in m_e c */
        std::pair<float_X, float_X> axis_p_range;
        uint32_t r_bins;

        typedef float_32 float_PS;
        /** depending on the super cells edge size and the PS float type
         *  we use not more than 32KB shared memory
         *  Note: checking the longest edge for all phase space configurations
         *        is a conservative work around until #469 is implemented */
        typedef typename bmpl::accumulate<
            typename SuperCellSize::mplVector,
            bmpl::int_<0>,
            bmpl::max<bmpl::_1, bmpl::_2>
            >::type SuperCellsLongestEdge;
        BOOST_STATIC_CONSTEXPR uint32_t maxShared = 32*1024; /* 32 KB */
        BOOST_STATIC_CONSTEXPR uint32_t num_pbins = maxShared/(sizeof(float_PS)*SuperCellsLongestEdge::value);

        container::DeviceBuffer<float_PS, 2>* dBuffer;

        /** reduce functor to a single host per plane */
        PMacc::algorithm::mpi::Reduce<simDim>* planeReduce;
        bool isPlaneReduceRoot;
        /** MPI communicator that contains the root ranks of the \p planeReduce
         */
        MPI_Comm commFileWriter;

    public:
        PhaseSpace( const std::string _name,
                     const std::string _prefix,
                     const uint32_t _notifyPeriod,
                     const std::pair<float_X, float_X>& _p_range,
                     const AxisDescription& _element );
        virtual ~PhaseSpace(){}

        void notify( uint32_t currentStep );
        template<uint32_t Direction>
        void calcPhaseSpace( );
        void setMappingDescription( MappingDesc* cellDescription);

        void pluginLoad();
        void pluginUnload();
    };

}

#include "PhaseSpace.tpp"
