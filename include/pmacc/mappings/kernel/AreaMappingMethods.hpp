/* Copyright 2013-2018 Felix Schmitt, Heiko Burau, Rene Widera
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
#include "pmacc/dimensions/DataSpace.hpp"

namespace pmacc
{

    /**
     * Helper class for AreaMapping.
     * Provides methods called by AreaMapping using template specialization.
     *
     * @tparam areaType the area to map to
     * @tparam DIM dimension of the mapping
     */
    template<uint32_t areaType, unsigned DIM>
    class AreaMappingMethods;

    //CORE + BORDER + GUARD

    template<unsigned DIM>
    class AreaMappingMethods<CORE + BORDER + GUARD, DIM>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base&, const DataSpace<DIM> &gBlocks)
        {
            return gBlocks;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(const Base&,
        const DataSpace<DIM>&,
        const DataSpace<DIM>& _blockIdx)
        {
            return _blockIdx;
        }
    };

    //CORE

    template<unsigned DIM>
    class AreaMappingMethods<CORE, DIM>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base &base, const DataSpace<DIM> &gBlocks)
        {
            return gBlocks - (2 * (base.getGuardingSuperCells() + base.getBorderSuperCells()));
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(const Base &base,
        const DataSpace<DIM> &gBlocks,
        const DataSpace<DIM>& _blockIdx)
        {
            return _blockIdx + (base.getGuardingSuperCells() + base.getBorderSuperCells());
        }
    };

    //CORE+BORDER

    template<unsigned DIM>
    class AreaMappingMethods<CORE + BORDER, DIM>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base &base, const DataSpace<DIM> &gBlocks)
        {
            return gBlocks - (2 * base.getGuardingSuperCells());
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(const Base &base,
        const DataSpace<DIM> &gBlocks,
        const DataSpace<DIM>& _blockIdx)
        {
            return _blockIdx + base.getGuardingSuperCells();
        }
    };


    //dim 2D

    //GUARD

    template<>
    class AreaMappingMethods<GUARD, DIM2>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM2> getGridDim(const Base &base, const DataSpace<DIM2> &gBlocks)
        {
            return DataSpace<DIM2 > (
                    gBlocks.x() +
                    gBlocks.y() - 2 * base.getGuardingSuperCells(),
                    2 * base.getGuardingSuperCells());
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2> getBlockIndex(const Base &base,
        const DataSpace<DIM2> &gBlocks,
        const DataSpace<DIM2>& _blockIdx)
        {
            if (_blockIdx.x() < gBlocks.x())
            {
                return DataSpace<DIM2 > (
                        _blockIdx.x(),
                        _blockIdx.y() / 2 +
                        (_blockIdx.y() & 1u) * (gBlocks.y() - base.getGuardingSuperCells()));
            }

            return DataSpace<DIM2 > (
                    _blockIdx.y() / 2 +
                    (_blockIdx.y() & 1u) * (gBlocks.x() - base.getGuardingSuperCells()),
                    base.getGuardingSuperCells() + _blockIdx.x() - gBlocks.x());
        }
    };

    //BORDER

    template<>
    class AreaMappingMethods<BORDER, DIM2>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM2> getGridDim(const Base &base, const DataSpace<DIM2> &gBlocks)
        {

            const uint32_t xOverhead = 2 * (base.getGuardingSuperCells());
            const uint32_t yOverhead = xOverhead + 2 * (base.getBorderSuperCells());
            return DataSpace<DIM2 > (
                    gBlocks.x() - xOverhead +
                    gBlocks.y() - yOverhead,
                    2 * base.getBorderSuperCells());
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2> getBlockIndex(const Base &base,
        const DataSpace<DIM2> &gBlocks,
        const DataSpace<DIM2>& _blockIdx)
        {
            const uint32_t width = gBlocks.x() - 2 * base.getGuardingSuperCells();
            if (_blockIdx.x() < width)
            {
                return DataSpace<DIM2 > (
                        base.getGuardingSuperCells() + _blockIdx.x(),
                        base.getGuardingSuperCells() + _blockIdx.y() / 2 +
                        (_blockIdx.y() & 1u) *
                        (gBlocks.y() - 2 * base.getGuardingSuperCells() - base.getBorderSuperCells())); /*gridBlocks.y()-2*blocksGuard-blocksBorder*/
            }

            return DataSpace<DIM2 > (
                    base.getGuardingSuperCells() + _blockIdx.y() / 2 +
                    (_blockIdx.y() & 1u) *
                    (gBlocks.x() - base.getBorderSuperCells() - 2 * base.getGuardingSuperCells()),
                    (base.getGuardingSuperCells() + base.getBorderSuperCells())
                    + _blockIdx.x() - width);

        }
    };

    template<>
    class AreaMappingMethods<GUARD, DIM3>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM3> getGridDim(const Base &base, const DataSpace<DIM3> &gBlocks)
        {
            const int x = gBlocks.x();
            const int x_ = gBlocks.x() - 2 * base.getGuardingSuperCells();
            const int y = gBlocks.y();
            const int z_ = gBlocks.z() - 2 * base.getGuardingSuperCells();


            return DataSpace<DIM3 > (x * y + z_ * y + x_*z_,
                    2 * base.getGuardingSuperCells(),
                    1);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3> getBlockIndex(const Base &base,
        const DataSpace<DIM3> &gBlocks,
        const DataSpace<DIM3>& _blockIdx)
        {
            const int x = gBlocks.x();
            const int x_ = gBlocks.x() - 2 * base.getGuardingSuperCells();
            const int y = gBlocks.y();
            const int z = gBlocks.z();
            const int z_ = gBlocks.z() - 2 * base.getGuardingSuperCells();

            if (_blockIdx.x() < (x * y))
            {
                /* area is x*y */
                const int tmp_x = _blockIdx.x();
                return DataSpace<DIM3 > (tmp_x % x,
                        tmp_x / x,
                        _blockIdx.y() / 2 +
                        (_blockIdx.y() & 1u) *
                        (z - base.getGuardingSuperCells()));
            }
            if ((_blockIdx.x() >= (x * y)) && _blockIdx.x() < (x * y + z_ * y))
            {
                /* area is z_*y */
                const int tmp_x = _blockIdx.x() - (x * y);
                return DataSpace<DIM3 > (_blockIdx.y() / 2 +
                        (_blockIdx.y() & 1u) *
                        (x - base.getGuardingSuperCells()),
                        tmp_x / z_,
                        (tmp_x % z_) + base.getGuardingSuperCells());
            }
            /* area is x_*z_ */
            const int tmp_x = _blockIdx.x() - (x * y) - (z_ * y);
            return DataSpace<DIM3 > ((tmp_x % x_) + base.getGuardingSuperCells(),
                    _blockIdx.y() / 2 +
                    (_blockIdx.y() & 1u) *
                    (y - base.getGuardingSuperCells()),
                    (tmp_x / x_) + base.getGuardingSuperCells());
        }
    };

    template<>
    class AreaMappingMethods<BORDER, DIM3>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM3> getGridDim(const Base &base, const DataSpace<DIM3> &gBlocks)
        {

            const int g = 2 * base.getGuardingSuperCells();
            const int b = 2 * base.getBorderSuperCells();
            const int x = gBlocks.x() - g;
            const int x_ = gBlocks.x() - g - b;
            const int y = gBlocks.y() - g;
            const int z_ = gBlocks.z() - g - b;


            return DataSpace<DIM3 > (x * y + z_ * y + x_*z_,
                    b,
                    1);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3> getBlockIndex(const Base &base,
        const DataSpace<DIM3> &gBlocks,
        const DataSpace<DIM3>& _blockIdx)
        {
            const int g = base.getGuardingSuperCells();
            const int b = base.getBorderSuperCells();
            const int x = gBlocks.x() - 2 * g;
            const int x_ = gBlocks.x() - 2 * g - 2 * b;
            const int y = gBlocks.y() - 2 * g;
            const int z = gBlocks.z() - 2 * g;
            const int z_ = gBlocks.z() - 2 * g - 2 * b;

            if (_blockIdx.x() < (x * y))
            {
                /* area is x*y */
                const int tmp_x = _blockIdx.x();
                return DataSpace<DIM3 > (tmp_x % x + g,
                        tmp_x / x + g,
                        g + _blockIdx.y() / 2 +
                        (_blockIdx.y() & 1u) *
                        (z - b));
            }
            if ((_blockIdx.x() >= (x * y)) && _blockIdx.x() < (x * y + z_ * y))
            {
                /* area is z_*y */
                const int tmp_x = _blockIdx.x() - (x * y);
                return DataSpace<DIM3 > (g + _blockIdx.y() / 2 +
                        (_blockIdx.y() & 1u) *
                        (x - b),
                        tmp_x / z_ + g,
                        (tmp_x % z_) + g + b);
            }
            /* area is x_*z_ */
            const int tmp_x = _blockIdx.x() - (x * y) - (z_ * y);
            return DataSpace<DIM3 > ((tmp_x % x_) + g + b,
                    g + _blockIdx.y() / 2 +
                    (_blockIdx.y() & 1u) *
                    (y - b),
                    (tmp_x / x_) + g + b);
        }
    };

} //namespace pmacc
