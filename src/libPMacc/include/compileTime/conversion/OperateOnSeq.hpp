/**
 * Copyright 2015 Rene Widera
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

#include "compileTime/accessors/Identity.hpp"
#include "types.h"

#include <boost/mpl/vector.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/type_traits.hpp>

namespace PMacc
{

/** run an unary operator on each element of a sequence
 *
 * @tparam T_MPLSeq any boost mpl sequence
 * @tparam T_UnaryOperator unary operator to translate type from the sequence
 * to a mpl pair
 * @tparam T_Accessor an unary lambda operator that is used before the type
 * from the sequence is passed to T_UnaryOperator
 * @return ::type bmpl::vector
 */
template<typename T_MPLSeq,
typename T_UnaryOperator,
typename T_Accessor = compileTime::accessors::Identity<>
>
struct OperateOnSeq
{

    template<typename X>
    struct Op :bmpl::apply1<T_UnaryOperator, typename bmpl::apply1<T_Accessor,X>::type >
    {
    };

    typedef T_MPLSeq MPLSeq;
    typedef bmpl::back_inserter< bmpl::vector<> > Inserter;
    typedef typename bmpl::transform<
            MPLSeq,
            Op<bmpl::_1>,
            Inserter
            >::type type;
};

}//namespace PMacc
