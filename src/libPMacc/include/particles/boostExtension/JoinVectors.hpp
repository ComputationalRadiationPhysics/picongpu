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

#include <boost/mpl/list.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/front_inserter.hpp>

namespace PMacc
{

namespace bmpl = boost::mpl;


/**Join vector to one vector
 * 
 * Create a vector with the order V1,V2,...,Vn
 * 
 * \todo: use vector instead of V1, V2 ...
 */

template<class V1, class V2, class V3 = bmpl::vector<> >
struct JoinVectors
{
private:
    typedef typename bmpl::copy<
        V2,
        bmpl::back_inserter< V1>
        >::type type_1;

    public:
        typedef typename bmpl::copy<
        V3,
        bmpl::back_inserter< type_1>
        >::type type;
};

}

