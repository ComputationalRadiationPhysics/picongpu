/**
 * Copyright 2013-2016 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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


#include <boost/mpl/vector.hpp>
#include "compileTime/conversion/MakeSeqFromNestedSeq.hpp"

namespace PMacc
{

/** combine all input types to one sequence
 *
 * Note: if the input type is a sequence itself, its elements will be unfolded
 *       and added separately
 *
 * @tparam T_N a boost mpl sequence or single type
 */
template<typename T_1 = bmpl::vector0<>, typename T_2 = bmpl::vector0<>,
         typename T_3 = bmpl::vector0<>, typename T_4 = bmpl::vector0<>,
         typename T_5 = bmpl::vector0<> >
struct MakeSeq
{
    typedef typename MakeSeqFromNestedSeq<
    bmpl::vector5<
    T_1,
    T_2,
    T_3,
    T_4,
    T_5
    >
    >::type type;
};

} //namespace PMacc
