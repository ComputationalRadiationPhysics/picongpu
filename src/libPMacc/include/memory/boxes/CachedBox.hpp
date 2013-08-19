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
 

#ifndef CACHEDBOX_HPP
#define	CACHEDBOX_HPP

#include "types.h"
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/SharedBox.hpp"


namespace PMacc
{
    namespace intern
    {

        template< typename ValueType_, class BlockDescription_, uint32_t Id_>
        class CachedBox
        {
        public:
            typedef BlockDescription_ BlockDescription;
            typedef ValueType_ ValueType;
        private:
            typedef typename BlockDescription::SuperCellSize SuperCellSize;
            typedef typename BlockDescription::FullSuperCellSize FullSuperCellSize;
            typedef typename BlockDescription::OffsetOrigin OffsetOrigin;

        public:
            typedef DataBox<SharedBox<ValueType, FullSuperCellSize> > Type;

            HDINLINE static Type create()
            {
                Type c_box(Type::init());
                return c_box.shift(OffsetOrigin());
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

#endif	/* CACHEDBOX_HPP */

