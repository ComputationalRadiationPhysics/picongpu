/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /**
     * Helper class for AreaMapping.
     * Provides methods called by AreaMapping using template specialization.
     *
     * @tparam T_area the area to map to
     * @tparam DIM dimension of the mapping
     */
    template<uint32_t T_area, unsigned DIM>
    class AreaMappingMethods;

    // CORE + BORDER + GUARD

    template<unsigned DIM>
    class AreaMappingMethods<CORE + BORDER + GUARD, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(Base const&, DataSpace<DIM> const& gBlocks)
        {
            return gBlocks;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getSuperCellIndex(
            Base const&,
            DataSpace<DIM> const&,
            DataSpace<DIM> const& _blockIdx)
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
        HINLINE static DataSpace<DIM> getGridDim(Base const& base, DataSpace<DIM> const& gBlocks)
        {
            // skip 2 x (border + guard) == 4 x guard
            return gBlocks - 4 * base.getGuardingSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM> const& gBlocks,
            DataSpace<DIM> const& _blockIdx)
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
        HINLINE static DataSpace<DIM> getGridDim(Base const& base, DataSpace<DIM> const& gBlocks)
        {
            // remove guard + border == 2 x guard
            return gBlocks - 2 * base.getGuardingSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM> const& gBlocks,
            DataSpace<DIM> const& _blockIdx)
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
        HINLINE static DataSpace<DIM2> getGridDim(Base const& base, DataSpace<DIM2> const& gBlocks)
        {
            int const x = gBlocks.x();
            int const y_ = gBlocks.y() - 2 * base.getGuardingSuperCells().y();

            int const xArea = x * base.getGuardingSuperCells().y();
            int const y_Area = y_ * base.getGuardingSuperCells().x();

            return DataSpace<DIM2>(xArea + y_Area, 2);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM2> const& gBlocks,
            DataSpace<DIM2> const& _blockIdx)
        {
            int const x = gBlocks.x();

            int const xArea = x * base.getGuardingSuperCells().y();

            if(_blockIdx.x() < xArea)
            {
                int const tmp_x = _blockIdx.x();
                return DataSpace<DIM2>(
                    tmp_x % x,
                    tmp_x / x +
                        // if _blockIdx.y() == 1 means bottom plane
                        _blockIdx.y() * (gBlocks.y() - base.getGuardingSuperCells().y()));
            }
            else
            {
                int const tmp_x = _blockIdx.x() - xArea;
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
        HINLINE static DataSpace<DIM2> getGridDim(Base const& base, DataSpace<DIM2> const& gBlocks)
        {
            // removes the guard, than BORDER is the new GUARD and we can reuse the GUARD mapper
            DataSpace<DIM2> const sizeWithoutGuard(gBlocks - 2 * base.getGuardingSuperCells());

            return AreaMappingMethods<GUARD, DIM2>{}.getGridDim(base, sizeWithoutGuard);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM2> const& gBlocks,
            DataSpace<DIM2> const& _blockIdx)
        {
            // removes the guard, than BORDER is the new GUARD and we can reuse the GUARD mapper
            DataSpace<DIM2> const sizeWithoutGuard(gBlocks - 2 * base.getGuardingSuperCells());

            // use result of the shrinked domain and skip guarding supercells
            return AreaMappingMethods<GUARD, DIM2>{}.getSuperCellIndex(base, sizeWithoutGuard, _blockIdx)
                   + base.getGuardingSuperCells();
        }
    };

    template<>
    class AreaMappingMethods<GUARD, DIM3>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM3> getGridDim(Base const& base, DataSpace<DIM3> const& gBlocks)
        {
            int const x = gBlocks.x();
            int const x_ = gBlocks.x() - 2 * base.getGuardingSuperCells().x();
            int const y = gBlocks.y();
            int const z_ = gBlocks.z() - 2 * base.getGuardingSuperCells().z();

            int const xyVolume = x * y * base.getGuardingSuperCells().z();
            int const z_yVolume = z_ * y * base.getGuardingSuperCells().x();
            int const z_x_Volume = z_ * x_ * base.getGuardingSuperCells().y();

            return DataSpace<DIM3>(xyVolume + z_x_Volume + z_yVolume, 2, 1);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM3> const& gBlocks,
            DataSpace<DIM3> const& _blockIdx)
        {
            int const x = gBlocks.x();
            int const x_ = gBlocks.x() - 2 * base.getGuardingSuperCells().x();
            int const y = gBlocks.y();
            int const z_ = gBlocks.z() - 2 * base.getGuardingSuperCells().z();

            int const xyVolume = x * y * base.getGuardingSuperCells().z();
            int const z_yVolume = z_ * y * base.getGuardingSuperCells().x();

            if(_blockIdx.x() < xyVolume)
            {
                /* area is x*y */
                int const xyPlane = x * y;
                int const tmp_x = _blockIdx.x();

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
                int const z_yPlane = z_ * y;
                int const tmp_x = _blockIdx.x() - xyVolume;

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
                int const x_z_Plane = x_ * z_;
                int const tmp_x = _blockIdx.x() - xyVolume - z_yVolume;
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
        HINLINE static DataSpace<DIM3> getGridDim(Base const& base, DataSpace<DIM3> const& gBlocks)
        {
            // removes the guard, than BORDER is the new GUARD and we can reuse the GUARD mapper
            DataSpace<DIM3> const sizeWithoutGuard(gBlocks - 2 * base.getGuardingSuperCells());

            return AreaMappingMethods<GUARD, DIM3>{}.getGridDim(base, sizeWithoutGuard);
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM3> const& gBlocks,
            DataSpace<DIM3> const& _blockIdx)
        {
            // removes the guard, than BORDER is the new GUARD and we can reuse the GUARD mapper
            DataSpace<DIM3> const sizeWithoutGuard(gBlocks - 2 * base.getGuardingSuperCells());

            // use result of the shrinked domain and skip guarding supercells
            return AreaMappingMethods<GUARD, DIM3>{}.getSuperCellIndex(base, sizeWithoutGuard, _blockIdx)
                   + base.getGuardingSuperCells();
        }
    };

} // namespace pmacc
