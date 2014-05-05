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


#ifndef SUPERCELLDESCRIPTION_HPP
#define	SUPERCELLDESCRIPTION_HPP

#include "types.h"
#include "dimensions/DataSpace.hpp"

namespace PMacc
{

    template< class SuperCellSize_, class OffsetOrigin_ = math::CT::make_Int<SuperCellSize_::dim, 0>, class OffsetEnd_ = math::CT::make_Int<SuperCellSize_::dim, 0> >
    struct SuperCellDescription
    {

        enum
        {
            Dim = SuperCellSize_::dim
        };
        typedef SuperCellSize_ SuperCellSize;
        typedef OffsetEnd_ OffsetEnd;
        typedef OffsetOrigin_ OffsetOrigin;
        typedef SuperCellDescription<SuperCellSize, OffsetOrigin_, OffsetEnd> Type;

        typedef typename OffsetOrigin::template add<typename SuperCellSize::template add<OffsetEnd>::Result >::Result FullSuperCellSize;
    };

}//namespace

#endif	/* SUPERCELLDESCRIPTION_HPP */

