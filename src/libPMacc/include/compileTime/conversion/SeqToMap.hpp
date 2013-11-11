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


#include "compileTime/accessors/Identity.hpp"

namespace PMacc
{
namespace bmpl = boost::mpl;

/** convert boost mpl sequence to a mpl map
 * 
 * @tparam T_MPLSeq any boost mpl sequence
 * @tparam T_UnaryOperator unaray operator to translate type from the sequence 
 * to a mpl pair
 * @tparam T_Accessor operator which is used before the type from the sequence is
 * passed to T_UnaryOperator
 * @return ::type mpl map
 */
template<typename T_MPLSeq,
template<typename> class T_UnaryOperator,
template<typename> class T_Accessor = compileTime::accessors::Identity
>
struct SeqToMap
{
    typedef T_MPLSeq MPLSeq;
    typedef bmpl::inserter< bmpl::map<>, bmpl::insert<bmpl::_1, bmpl::_2> > Map_inserter;
    typedef typename bmpl::transform<
            MPLSeq,
            T_UnaryOperator<typename T_Accessor<bmpl::_1>::type>,
                Map_inserter 
            >::type type;

};

}//namespace PMacc
