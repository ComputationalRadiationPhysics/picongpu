/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/SharedBox.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace CachedBox
    {
        template<uint32_t Id_, typename ValueType_, class BlockDescription_, typename T_Worker>
        DINLINE auto create(T_Worker const& worker, BlockDescription_ const block)
        {
            using OffsetOrigin = typename BlockDescription_::OffsetOrigin;
            using Type = DataBox<SharedBox<ValueType_, typename BlockDescription_::FullSuperCellSize, Id_>>;
            return Type{Type::init(worker)}.shift(DataSpace<OffsetOrigin::dim>{OffsetOrigin::toRT()});
        }

        template<uint32_t Id_, typename ValueType_, class BlockDescription_, typename T_Worker>
        DINLINE auto create(T_Worker const& worker, ValueType_ const& value, BlockDescription_ const block)
        {
            return create<Id_, ValueType_, BlockDescription_>(worker);
        }
    } // namespace CachedBox
} // namespace pmacc
