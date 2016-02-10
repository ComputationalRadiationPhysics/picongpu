/**
 * Copyright 2015-2016 Heiko Burau
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

#include "pmacc_types.hpp"
#include "traits/HasFlag.hpp"
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/placeholders.hpp>

namespace PMacc
{
namespace particles
{
namespace traits
{

/** Return a new sequence of particle species carrying flag.
 *
 * @tparam T_MPLSeq sequence of particle species
 * @tparam T_Flag flag to be filtered
 */
template<typename T_MPLSeq, typename T_Flag>
struct FilterByFlag
{
    typedef T_MPLSeq MPLSeq;
    typedef T_Flag Flag;

    template<typename T_Species>
    struct HasFlag
    {
        typedef typename ::PMacc::traits::HasFlag<
            typename T_Species::FrameType,
            Flag>::type type;
    };

    typedef typename bmpl::copy_if<MPLSeq, HasFlag<bmpl::_> >::type type;
};

}//namespace traits
}//namespace particles
}//namespace PMacc
