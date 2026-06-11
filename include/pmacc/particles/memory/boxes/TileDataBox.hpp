/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/particles/frame_types.hpp"

namespace pmacc
{
    template<class TYPE>
    class VectorDataBox : public DataBox<PitchedBox<TYPE, DIM1>>
    {
    public:
        using BaseType = DataBox<PitchedBox<TYPE, 1U>>;
        using type = TYPE;

        HDINLINE VectorDataBox(TYPE* pointer, DataSpace<DIM1> const& offset = {})
            : BaseType(BaseType(PitchedBox<TYPE, DIM1>(pointer)).shift(offset))
        {
        }

        HDINLINE VectorDataBox() = default;
    };

    /**
     * Specifies a one-dimensional DataBox for more convenient usage.
     *
     * @tparam TYPE type of data represented by the DataBox
     */
    template<class TYPE>
    class TileDataBox : public VectorDataBox<TYPE>
    {
    public:
        using BaseType = VectorDataBox<TYPE>;

        HDINLINE TileDataBox(TYPE* pointer, DataSpace<DIM1> const& offset = DataSpace<DIM1>(0), uint32_t size = 0)
            : BaseType(pointer, offset)
            , size(size)
        {
        }

        /**
         * Returns  size of the Box.
         *
         * @return size of this TileDataBox
         */
        HDINLINE int getSize()
        {
            return size;
        }

        /*object is not  initialized valid, copy a valid instance to this object to get a valid instance*/
        HDINLINE TileDataBox() = default;


    protected:
        PMACC_ALIGN(size, size_t);
    };


} // namespace pmacc
