/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/SharedBox.hpp"


namespace PMacc
{
    namespace intern
    {

        template< typename T_ValueType, class T_BlockDescription, uint32_t T_Id>
        class CachedBox
        {
        public:
            typedef T_BlockDescription BlockDescription;
            typedef T_ValueType ValueType;
        private:
            typedef typename BlockDescription::SuperCellSize SuperCellSize;
            typedef typename BlockDescription::FullSuperCellSize FullSuperCellSize;
            typedef typename BlockDescription::OffsetOrigin OffsetOrigin;

        public:
            typedef DataBox<SharedBox<ValueType, FullSuperCellSize,T_Id> > Type;

            HDINLINE static Type create()
            {
                DataSpace<OffsetOrigin::dim> offset(OffsetOrigin::toRT());
                Type c_box(Type::init());
                return c_box.shift(offset);
            }

        };
    }

    struct CachedBox
    {

        template<uint32_t Id_, typename ValueType_, class BlockDescription_ >
        DINLINE static typename intern::CachedBox<ValueType_, BlockDescription_, Id_ >::Type
        create(const ValueType_& value, const BlockDescription_ block)
        {
            return intern::CachedBox<ValueType_, BlockDescription_, Id_>::create();
        }

        template< uint32_t Id_, typename ValueType_, class BlockDescription_ >
        DINLINE static typename intern::CachedBox<ValueType_, BlockDescription_, Id_ >::Type
        create(const BlockDescription_ block)
        {
            return intern::CachedBox<ValueType_, BlockDescription_, Id_>::create();
        }

    };

}
