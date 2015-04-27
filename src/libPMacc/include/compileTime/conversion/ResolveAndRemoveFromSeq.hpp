/**
 * Copyright 2014 Rene Widera
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

#include "types.h"

#include "compileTime/conversion/RemoveFromSeq.hpp"
#include "compileTime/conversion/ResolveAliases.hpp"

namespace PMacc
{

/* resolve and remove types from a sequence
 *
 * @tparam T_MPLSeqSrc source sequence from were we delete types
 * @tparam T_MPLSeqObjectsToRemove sequence with types which shuld be deleted (pmacc aliases are allowed)
 */
template<
typename T_MPLSeqSrc,
typename T_MPLSeqObjectsToRemove
>
struct ResolveAndRemoveFromSeq
{
    typedef T_MPLSeqSrc MPLSeqSrc;
    typedef T_MPLSeqObjectsToRemove MPLSeqObjectsToRemove;
    typedef typename ResolveAliases<MPLSeqObjectsToRemove, MPLSeqSrc>::type ResolvedSeqWithObjectsToRemove;
    typedef typename RemoveFromSeq<MPLSeqSrc, ResolvedSeqWithObjectsToRemove>::type type;
};

}//namespace PMacc
