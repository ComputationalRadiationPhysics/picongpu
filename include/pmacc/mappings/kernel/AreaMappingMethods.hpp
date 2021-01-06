/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera
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

    // CORE + BORDER + GUARD

    template<unsigned DIM>
    class AreaMappingMethods<CORE + BORDER + GUARD, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base&, const DataSpace<DIM>& gBlocks)
        {
            return gBlocks;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(
            const Base&,
            const DataSpace<DIM>&,
            const DataSpace<DIM>& _blockIdx)
        {
            return _blockIdx;
        }
    };

    // CORE

    template<unsigned DIM>
    class AreaMappingMethods<CORE, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base& base, const DataSpace<DIM>& gBlocks)
        {
            // skip 2 x (border + guard) == 4 x guard
            return gBlocks - 4 * base.getGuardingSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(
            const Base& base,
            const DataSpace<DIM>& gBlocks,
            const DataSpace<DIM>& _blockIdx)
        {
            // skip guard + border == 2 x guard
            return _blockIdx + 2 * base.getGuardingSuperCells();
        }
    };

    // CORE+BORDER

    template<unsigned DIM>
    class AreaMappingMethods<CORE + BORDER, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base& base, const DataSpace<DIM>& gBlocks)
        {
            // remove guard + border == 2 x guard
            return gBlocks - 2 * base.getGuardingSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(
            const Base& base,
            const DataSpace<DIM>& gBlocks,
            const DataSpace<DIM>& _blockIdx)
        {
            // skip guarding supercells
            return _blockIdx + base.getGuardingSuperCells();
        }
    };


    // dim 2D

    // GUARD

    template<>
    class AreaMappingMethods<GUARD, DIM2>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM2> getGridDim(const Base& base, const DataSpace<DIM2>& gBlocks)
        {
            const int x = gBlocks.x();
            const int y_ = gBlocks.y() - 2 * base.getGuardingSuperCells().y();

            const int xArea = x * base.getGuardingSuperCells().y();
            const int y_Area = y_ * base.getGuardingSuperCells().x();

            return DataSpace<DIM2>(xArea + y_Area, 2);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2> getBlockIndex(
            const Base& base,
            const DataSpace<DIM2>& gBlocks,
            const DataSpace<DIM2>& _blockIdx)
        {
            const int x = gBlocks.x();

            const int xArea = x * base.getGuardingSuperCells().y();

            if(_blockIdx.x() < xArea)
            {
                const int tmp_x = _blockIdx.x();
                return DataSpace<DIM2>(
                    tmp_x % x,
                    tmp_x / x +
                        // if _blockIdx.y() == 1 means bottom plane
                        _blockIdx.y() * (gBlocks.y() - base.getGuardingSuperCells().y()));
            }
            else
            {
                const int tmp_x = _blockIdx.x() - xArea;
                return DataSpace<DIM2>(
                    tmp_x % base.getGuardingSuperCells().x() +
                        // if _blockIdx.y() == 1 means right plane
                        _blockIdx.y() * (gBlocks.x() - base.getGuardingSuperCells().x()),
                    tmp_x / base.getGuardingSuperCells().x() + base.getGuardingSuperCells().y());
            }
        }
    };

    // BORDER

    template<>
    class AreaMappingMethods<BORDER, DIM2>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM2> getGridDim(const Base& base, const DataSpace<DIM2>& gBlocks)
        {
            // removes the guard, than BORDER is the new GUARD and we can reuse the GUARD mapper
            const DataSpace<DIM2> sizeWithoutGuard(gBlocks - 2 * base.getGuardingSuperCells());

            return AreaMappingMethods<GUARD, DIM2>{}.getGridDim(base, sizeWithoutGuard);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2> getBlockIndex(
            const Base& base,
            const DataSpace<DIM2>& gBlocks,
            const DataSpace<DIM2>& _blockIdx)
        {
            // removes the guard, than BORDER is the new GUARD and we can reuse the GUARD mapper
            const DataSpace<DIM2> sizeWithoutGuard(gBlocks - 2 * base.getGuardingSuperCells());

            // use result of the shrinked domain and skip guarding supercells
            return AreaMappingMethods<GUARD, DIM2>{}.getBlockIndex(base, sizeWithoutGuard, _blockIdx)
                + base.getGuardingSuperCells();
        }
    };

    template<>
    class AreaMappingMethods<GUARD, DIM3>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM3> getGridDim(const Base& base, const DataSpace<DIM3>& gBlocks)
        {
            const int x = gBlocks.x();
            const int x_ = gBlocks.x() - 2 * base.getGuardingSuperCells().x();
            const int y = gBlocks.y();
            const int z_ = gBlocks.z() - 2 * base.getGuardingSuperCells().z();

            const int xyVolume = x * y * base.getGuardingSuperCells().z();
            const int z_yVolume = z_ * y * base.getGuardingSuperCells().x();
            const int z_x_Volume = z_ * x_ * base.getGuardingSuperCells().y();

            return DataSpace<DIM3>(xyVolume + z_x_Volume + z_yVolume, 2, 1);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3> getBlockIndex(
            const Base& base,
            const DataSpace<DIM3>& gBlocks,
            const DataSpace<DIM3>& _blockIdx)
        {
            const int x = gBlocks.x();
            const int x_ = gBlocks.x() - 2 * base.getGuardingSuperCells().x();
            const int y = gBlocks.y();
            const int z_ = gBlocks.z() - 2 * base.getGuardingSuperCells().z();

            const int xyVolume = x * y * base.getGuardingSuperCells().z();
            const int z_yVolume = z_ * y * base.getGuardingSuperCells().x();

            if(_blockIdx.x() < xyVolume)
            {
                /* area is x*y */
                const int xyPlane = x * y;
                const int tmp_x = _blockIdx.x();

                return DataSpace<DIM3>(
                    tmp_x % x,
                    tmp_x / x % y,
                    tmp_x / xyPlane +
                        // if _blockIdx.y() == 1 means back plane
                        _blockIdx.y() * (gBlocks.z() - base.getGuardingSuperCells().z()));
            }
            else if(_blockIdx.x() >= xyVolume && _blockIdx.x() < xyVolume + z_yVolume)
            {
                /* area is z_*y */
                const int z_yPlane = z_ * y;
                const int tmp_x = _blockIdx.x() - xyVolume;

                return DataSpace<DIM3>(
                    tmp_x / z_yPlane +
                        // if _blockIdx.y() == 1 means right plane
                        _blockIdx.y() * (gBlocks.x() - base.getGuardingSuperCells().x()),
                    tmp_x % y,
                    tmp_x / y % z_ + base.getGuardingSuperCells().z());
            }
            else
            {
                /* area is x_*z_ */
                const int x_z_Plane = x_ * z_;
                const int tmp_x = _blockIdx.x() - xyVolume - z_yVolume;
                return DataSpace<DIM3>(
                    (tmp_x % x_) + base.getGuardingSuperCells().x(),
                    tmp_x / x_z_Plane +
                        // if _blockIdx.y() == 1 means bottom plane
                        _blockIdx.y() * (gBlocks.y() - base.getGuardingSuperCells().y()),
                    tmp_x / x_ % z_ + base.getGuardingSuperCells().z());
            }
        }
    };

    template<>
    class AreaMappingMethods<BORDER, DIM3>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM3> getGridDim(const Base& base, const DataSpace<DIM3>& gBlocks)
        {
            // removes the guard, than BORDER is the new GUARD and we can reuse the GUARD mapper
            const DataSpace<DIM3> sizeWithoutGuard(gBlocks - 2 * base.getGuardingSuperCells());

            return AreaMappingMethods<GUARD, DIM3>{}.getGridDim(base, sizeWithoutGuard);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3> getBlockIndex(
            const Base& base,
            const DataSpace<DIM3>& gBlocks,
            const DataSpace<DIM3>& _blockIdx)
        {
            // removes the guard, than BORDER is the new GUARD and we can reuse the GUARD mapper
            const DataSpace<DIM3> sizeWithoutGuard(gBlocks - 2 * base.getGuardingSuperCells());

            // use result of the shrinked domain and skip guarding supercells
            return AreaMappingMethods<GUARD, DIM3>{}.getBlockIndex(base, sizeWithoutGuard, _blockIdx)
                + base.getGuardingSuperCells();
        }
    };

} // namespace pmacc
