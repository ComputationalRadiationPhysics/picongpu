/**
 * Copyright 2015 Heiko Burau
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "types.h"
#include "traits/GetMargin.hpp"
#include "dimensions/SuperCellDescription.hpp"
#include "cuSTL/container/compile-time/SharedBuffer.hpp"
#include "cuSTL/algorithm/cudaBlock/Foreach.hpp"
#include "vector/Int.hpp"
#include "lambda/placeholder.h"

namespace picongpu
{

using namespace PMacc;

/** Provides field-to-particle interpolation of a cached (shared memory)
 * field on device. The caching is done by this class.
 *
 * \tparam T_Buffer container type (cuSTL) of field data
 * \tparam T_FieldToParticleInterpolation interpolation algorithm
 * \tparam T_NumericalFieldPosition A type with an operator(), that returns
 *      a VectorVector indicating the numerical field position.
 * \tparam T_CudaBlockDim compile-time vector of the used blockDim
 * \tparam T_sharedMemIdx Arbitrary, unique integer. Has to be unique to
 *      every identical shared memory block in use (due to a nvcc bug).
 *
 * \see: "algorithms/FieldToParticleInterpolation.hpp"
 * \see: "fields/numericalCellTypes/GetNumericalFieldPos.hpp"
 */
template<
    typename T_Buffer,
    typename T_FieldToParticleInterpolation,
    typename T_NumericalFieldPosition,
    typename T_CudaBlockDim,
    int T_sharedMemIdx>
class Cached_F2P_Interp
{
public:
    typedef T_Buffer Buffer;
    typedef T_FieldToParticleInterpolation FieldToParticleInterpolation;
    typedef T_NumericalFieldPosition NumericalFieldPosition;
    typedef T_CudaBlockDim CudaBlockDim;
    BOOST_STATIC_CONSTEXPR int sharedMemIdx = T_sharedMemIdx;
    typedef typename Buffer::type type;
    BOOST_STATIC_CONSTEXPR int dim = Buffer::dim;

private:
    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<FieldToParticleInterpolation>::LowerMargin LowerMargin;
    typedef typename GetMargin<FieldToParticleInterpolation>::UpperMargin UpperMargin;

    /* relevant area of a block */
    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
        > BlockArea;

    typedef container::CT::SharedBuffer<type, typename BlockArea::FullSuperCellSize, sharedMemIdx> CachedBuffer;

    PMACC_ALIGN(globalFieldCursor, typename Buffer::Cursor);
    PMACC_ALIGN(cachedFieldCursor, typename CachedBuffer::Cursor);

public:
    /**
     * @param buffer cuSTL buffer of the field data
     */
    Cached_F2P_Interp(const Buffer& buffer) : globalFieldCursor(buffer.origin()) {}

    /** Fill the cache with the field data.
     *
     * @param blockCell multi-dim offset from the origin of the local domain on GPU
     *                  to the origin of the block of the in unit of cells
     * @param linearThreadIdx linear coordinate of the thread in the threadBlock
     */
    DINLINE
    void init(const math::Int<dim>& blockCell, const int linearThreadIdx)
    {
        CachedBuffer cachedBuffer; /* allocate shared memory. Valid for kernel lifetime. */
        this->cachedFieldCursor = cachedBuffer.origin();

        const math::Int<dim> lowerMargin = LowerMargin::toRT();

        using namespace lambda;
        DECLARE_PLACEHOLDERS(); // declares _1, _2, _3, ... in device code

        /* fill shared memory with global field data */
        algorithm::cudaBlock::Foreach<CudaBlockDim> foreach(linearThreadIdx);
        foreach(
            CachedBuffer::Zone(), /* super cell size + margins */
            this->cachedFieldCursor,
            this->globalFieldCursor(blockCell - lowerMargin),
            _1 = _2);

        /* jump to the origin of the SuperCell */
        this->cachedFieldCursor = this->cachedFieldCursor(lowerMargin);
    }

    /** Perform the field-to-particle interpolation.
     *
     * @param particlePos particle position within cell
     * @param localCell multi-dim coordinate of the local cell inside the super cell
     * @return interpolated field value
     */
    DINLINE
    type operator()(const floatD_X& particlePos, const math::Int<dim> localCell)
    {
        /* interpolation of the field */
        return FieldToParticleInterpolation()
            (this->cachedFieldCursor(localCell), particlePos, NumericalFieldPosition()());
    }
};

} // namespace picongpu
