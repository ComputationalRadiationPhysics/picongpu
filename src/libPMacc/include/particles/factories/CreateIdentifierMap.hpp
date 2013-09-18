/**
 * Copyright 2013 René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */

#pragma once

#include "types.h"

#include <boost/mpl/map.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/insert.hpp>

#include <boost/type_traits.hpp>

#include "math/MapTuple.hpp"
#include "math/Vector.hpp"

#include "particles/memory/boxes/TileDataBox.hpp"
#include "algorithms/accessors/Identity.hpp"

namespace PMacc
{
namespace bmpl = boost::mpl;
namespace pmath = PMacc::math;

template<typename InType>
struct CastToVector
{
    typedef
    bmpl::pair<typename InType::first,
        PMacc::math::Vector< typename InType::second, DIM3> >
        type;
};

template<typename InType>
struct AddReference
{
    typedef
    bmpl::pair<
        typename InType::first,
        typename boost::add_reference<typename InType::second>::type>
        type;
};

template<typename InType>
struct CastToVectorBox
{
    typedef
    bmpl::pair<typename InType::first,
        PMacc::VectorDataBox< typename InType::second > >
        type;
};

template<typename Map_, 
        template<typename> class UnaryOperator,
        template<typename> class Accessor = algorithms::accessors::Identity
        >
struct CreateIdentifierMap
{
    typedef Map_ Map;
    typedef bmpl::inserter< bmpl::map0<>, bmpl::insert<bmpl::_1, bmpl::_2> > Map_inserter;
    typedef typename bmpl::transform<
        Map,
        UnaryOperator<
                typename Accessor<
                        bmpl::_1
                >::type
        >, Map_inserter >::type type;

};

}//namespace PMacc
