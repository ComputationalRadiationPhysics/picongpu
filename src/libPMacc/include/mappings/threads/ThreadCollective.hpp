/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "dimensions/SuperCellDescription.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "dimensions/DataSpace.hpp"
#include "types.h"

namespace PMacc
{

template<class BlockArea_, int MaxThreads_ =  math::CT::volume<typename BlockArea_::SuperCellSize>::type::value >
class ThreadCollective
{
private:
    typedef typename BlockArea_::SuperCellSize SuperCellSize;
    typedef typename BlockArea_::FullSuperCellSize FullSuperCellSize;
    typedef typename BlockArea_::OffsetOrigin OffsetOrigin;
    BOOST_STATIC_CONSTEXPR int maxThreads=MaxThreads_;

    enum
    {
        Dim = BlockArea_::Dim
    };

public:

    DINLINE ThreadCollective(const int threadIndex) : threadId(threadIndex)
    {
    }

    DINLINE ThreadCollective(const DataSpace<Dim> threadIndex) :
    threadId(DataSpaceOperations<Dim>::template map<SuperCellSize>(threadIndex))
    {
    }

    template<class F, class P1, class P2>
    DINLINE void operator()(F& f, P1& p1, P2& p2)
    {
        for (int i = threadId; i < math::CT::volume<FullSuperCellSize>::type::value; i += maxThreads)
        {
            const DataSpace<Dim> pos(DataSpaceOperations<Dim>::template map<FullSuperCellSize > (i) - OffsetOrigin::toRT());
            f(p1(pos), p2(pos));
        }
    }

    template<class F, class P1>
    DINLINE void operator()(F& f, P1 & p1)
    {
        for (int i = threadId; i < math::CT::volume<FullSuperCellSize>::type::value; i += maxThreads)
        {
            const DataSpace<Dim> pos(DataSpaceOperations<Dim>::template map<FullSuperCellSize > (i) - OffsetOrigin::toRT());
            f(p1(pos));
        }
    }


private:
    const PMACC_ALIGN(threadId, int);

};

}//namespace
