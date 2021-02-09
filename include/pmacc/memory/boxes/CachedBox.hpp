/* Copyright 2013-2021 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "pmacc/types.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/SharedBox.hpp"


namespace pmacc
{
    namespace intern
    {
        template<typename T_ValueType, class T_BlockDescription, uint32_t T_Id>
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
            typedef DataBox<SharedBox<ValueType, FullSuperCellSize, T_Id>> Type;

            template<typename T_Acc>
            HDINLINE static Type create(T_Acc const& acc)
            {
                DataSpace<OffsetOrigin::dim> offset(OffsetOrigin::toRT());
                Type c_box(Type::init(acc));
                return c_box.shift(offset);
            }
        };
    } // namespace intern

    struct CachedBox
    {
        template<uint32_t Id_, typename ValueType_, class BlockDescription_, typename T_Acc>
        DINLINE static typename intern::CachedBox<ValueType_, BlockDescription_, Id_>::Type create(
            T_Acc const& acc,
            const ValueType_& value,
            const BlockDescription_ block)
        {
            return intern::CachedBox<ValueType_, BlockDescription_, Id_>::create(acc);
        }

        template<uint32_t Id_, typename ValueType_, class BlockDescription_, typename T_Acc>
        DINLINE static typename intern::CachedBox<ValueType_, BlockDescription_, Id_>::Type create(
            T_Acc const& acc,
            const BlockDescription_ block)
        {
            return intern::CachedBox<ValueType_, BlockDescription_, Id_>::create(acc);
        }
    };

} // namespace pmacc
