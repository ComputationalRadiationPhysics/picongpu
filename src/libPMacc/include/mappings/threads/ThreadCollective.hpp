/**
 * Copyright 2013 Heiko Burau, René Widera
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
 
/* 
 * File:   ThreadCollective.hpp
 * Author: widera
 *
 * Created on 8. März 2012, 11:00
 */

#ifndef THREADCOLLECTIVE_HPP
#define	THREADCOLLECTIVE_HPP

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "dimensions/SuperCellDescription.hpp"

namespace PMacc
{

template<class BlockArea_, uint32_t Dim_ = BlockArea_::Dim >
class ThreadCollective
{
private:
    typedef typename BlockArea_::SuperCellSize SuperCellSize;
    typedef typename BlockArea_::FullSuperCellSize FullSuperCellSize;
    typedef typename BlockArea_::OffsetOrigin OffsetOrigin;

    enum
    {
        Dim = Dim_
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
        for (int i = threadId; i < FullSuperCellSize::elements; i += SuperCellSize::elements)
        {
            const DataSpace<Dim> pos(DataSpaceOperations<Dim>::template map<FullSuperCellSize > (i) - OffsetOrigin());
            f(p1(pos), p2(pos));
        }
    }

    template<class F, class P1>
    DINLINE void operator()(F& f, P1 & p1)
    {
        for (int i = threadId; i < FullSuperCellSize::elements; i += SuperCellSize::elements)
        {
            const DataSpace<Dim> pos(DataSpaceOperations<Dim>::template map<FullSuperCellSize > (i) - OffsetOrigin());
            f(p1(pos));
        }
    }


private:
    const PMACC_ALIGN(threadId, int);

};

}//namespace

#endif	/* THREADCOLLECTIVE_HPP */

