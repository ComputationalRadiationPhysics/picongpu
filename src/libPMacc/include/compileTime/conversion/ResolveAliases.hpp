/**
 * Copyright 2013 Rene Widera
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

#include <boost/mpl/vector.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/insert.hpp>

#include <boost/type_traits.hpp>

namespace PMacc
{

/* translate all pmacc alias types to full specialized types
 *
 * use lookup sequense to translate types
 * if type from T_MPLSeq is not in T_MPLSeqLookup a compile time error is triggered
 *
 * @tparam T_MPLSeq source sequence with types to translate
 * @tparam T_MPLSeqLookup lookup sequence to translate alieses
 */
template<
typename T_MPLSeq,
typename T_MPLSeqLookup
>
struct ResolveAliases
{
    typedef T_MPLSeq MPLSeq;
    typedef T_MPLSeqLookup MPLSeqLookup;
    typedef bmpl::back_inserter< bmpl::vector<> > Inserter;

    template<typename T_Identifier>
    struct GetKeyFromAliasAccessor
    {
        typedef typename GetKeyFromAlias_assert<MPLSeqLookup, T_Identifier>::type type;
    };

    typedef typename bmpl::transform<
        MPLSeq,
        GetKeyFromAliasAccessor<bmpl::_1>
    >::type type;
};

}//namespace PMacc
