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
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/particles/memory/frames/NullFrame.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"

namespace pmacc
{
    namespace privatePositionFilter
    {
        template<unsigned T_dim, class Base = NullFrame>
        class PositionFilter : public Base
        {
        public:
            static constexpr uint32_t dim = T_dim;

        protected:
            DataSpace<dim> offset;
            DataSpace<dim> max;
            DataSpace<dim> superCellIdx;

        public:
            HDINLINE PositionFilter()
            {
            }

            HDINLINE void setWindowPosition(DataSpace<dim> offset, DataSpace<dim> size)
            {
                this->offset = offset;
                this->max = offset + size;
            }

            HDINLINE void setSuperCellPosition(DataSpace<dim> superCellIdx)
            {
                this->superCellIdx = superCellIdx;
            }

            HDINLINE DataSpace<dim> getOffset()
            {
                return offset;
            }

            template<class FRAME>
            HDINLINE bool operator()(FRAME& frame, lcellId_t id)
            {
                DataSpace<dim> localCellIdx = DataSpaceOperations<dim>::template map<typename FRAME::SuperCellSize>(
                    (uint32_t)(frame[id][localCellIdx_]));
                DataSpace<dim> pos = this->superCellIdx + localCellIdx;
                bool result = true;
                for(uint32_t d = 0; d < dim; ++d)
                    result = result && (this->offset[d] <= pos[d]) && (pos[d] < this->max[d]);
                return Base::operator()(frame, id) && result;
            }
        };

    } // namespace privatePositionFilter

    /** This wrapper class is needed because for filters we are only allowed to
     * define one template parameter "base" (it is a constrain from FilterFactory)
     */
    template<class Base = NullFrame>
    class PositionFilter3D : public privatePositionFilter::PositionFilter<DIM3, Base>
    {
    };

    template<class Base = NullFrame>
    class PositionFilter2D : public privatePositionFilter::PositionFilter<DIM2, Base>
    {
    };

    template<unsigned dim>
    struct GetPositionFilter;

    template<>
    struct GetPositionFilter<DIM3>
    {
        typedef PositionFilter3D<> type;
    };

    template<>
    struct GetPositionFilter<DIM2>
    {
        typedef PositionFilter2D<> type;
    };


} // namespace pmacc
