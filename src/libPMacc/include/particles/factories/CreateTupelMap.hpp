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

template<typename T>
class Array256
{
private:
    T data[256];
public:

    template<class> struct result;

    template<class F, typename TKey>
    struct result<F(TKey)>
    {
        typedef T& type;
    };

    HDINLINE
    T& operator[](const int idx)
    {
        return data[idx];
    }

    HDINLINE
    T& operator[](const int idx) const
    {
        return data[idx];
    }
};

template<typename Key>
struct CastToArray
{
    typedef
    bmpl::pair<Key,
            Array256<typename Key::type> >
            type;
};

template<typename InType>
struct CastToVector
{
    typedef
    bmpl::pair<InType,
            PMacc::math::Vector< typename InType::type, DIM3> >
            type;
};

template<typename InType>
struct AddReference
{
    typedef
    bmpl::pair<
            InType,
            typename boost::add_reference<typename InType::type>::type>
            type;
};

template<typename InType>
struct CastToVectorBox
{
    typedef
    bmpl::pair< InType,
            PMacc::VectorDataBox< typename InType::type > >
            type;
};

template<typename T_MPLSeq,
template<typename> class T_UnaryOperator,
template<typename> class T_Accessor = algorithms::accessors::Identity
>
struct CreateTupelMap
{
    typedef T_MPLSeq MPLSeq;
    typedef bmpl::inserter< bmpl::map0<>, bmpl::insert<bmpl::_1, bmpl::_2> > Map_inserter;
    typedef typename bmpl::transform<
            MPLSeq,
            T_UnaryOperator<
            typename T_Accessor<
            bmpl::_1
            >::type
            >, Map_inserter >::type type;

};

}//namespace PMacc
