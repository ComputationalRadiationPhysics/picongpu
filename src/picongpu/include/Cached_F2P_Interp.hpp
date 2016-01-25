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

#include "traits/GetMargin.hpp"
#include "dimensions/SuperCellDescription.hpp"
#include "cuSTL/container/compile-time/SharedBuffer.hpp"
#include "cuSTL/algorithm/cudaBlock/Foreach.hpp"
#include "math/vector/Int.hpp"
#include "lambda/placeholder.h"
#include "types.h"

namespace picongpu
{

using namespace PMacc;

/** Provides cached (shared memory) field-to-particle interpolation for a global field on device.
 * The caching operation is included in this class.
 *
 * \tparam T_Buffer container type (cuSTL) of field data
 * \tparam T_FieldToParticleInterpolation interpolation algorithm
 * \tparam T_NumericalFieldPosition A type with an operator(), that returns
 *      a VectorVector indicating the numerical field position.
 * \tparam T_CudaBlockDim compile-time vector of blockDim
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
    int T_sharedMemIdx>
class Cached_F2P_Interp
{
public:
    typedef T_Buffer Buffer;
    typedef T_FieldToParticleInterpolation FieldToParticleInterpolation;
    typedef T_NumericalFieldPosition NumericalFieldPosition;
    BOOST_STATIC_CONSTEXPR int sharedMemIdx = T_sharedMemIdx;

    typedef typename Buffer::type type;
    BOOST_STATIC_CONSTEXPR int dim = Buffer::dim;

private:
    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<FieldToParticleInterpolation>::LowerMargin LowerMargin;
    typedef typename GetMargin<FieldToParticleInterpolation>::UpperMargin UpperMargin;

    /* super cell size + margins */
    typedef typename SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
        >::FullSuperCellSize FullSuperCellSize;

    typedef container::CT::SharedBuffer<type, FullSuperCellSize, sharedMemIdx> CachedBuffer;
    typedef typename CachedBuffer::Cursor CCursor;

    PMACC_ALIGN(globalFieldCursor, typename Buffer::Cursor);
    PMACC_ALIGN(cachedFieldCursor, typename CachedBuffer::SafeCursor);
    //typename Buffer::Cursor globalFieldCursor;
    //typename CachedBuffer::SafeCursor cachedFieldCursor;

public:
    /**
     * @param buffer cuSTL buffer of the field data
     */
    Cached_F2P_Interp(const Buffer& buffer) :
        globalFieldCursor(buffer.origin()),
        cachedFieldCursor(CCursor(NULL)) /* gets valid in `init()` */
        /*cachedFieldCursor(NULL)*/ /* gets valid in `init()` */
        {}

    /** Fill shared memory cache with global memory field data.
     *
     * @param blockCell multi-dim offset from the origin of the local domain on GPU
     *                  to the origin of the block of the in unit of cells
     * @param linearThreadIdx linear coordinate of the thread in the threadBlock
     */
    DINLINE
    void init(const PMacc::math::Int<dim>& blockCell, const int linearThreadIdx)
    {
        CachedBuffer cachedBuffer; /* \TODO: allocate shared memory. Valid for kernel lifetime. */
        /*__shared__ type buf[600];
        if(linearThreadIdx == 0)
        {
            printf("buf: %x\n", buf);
            printf("buf+600: %x\n", buf+600);
        }
        this->cachedFieldCursor = CachedBuffer::Cursor(reinterpret_cast<type*>(buf));*/
        this->cachedFieldCursor = cachedBuffer.originSafe();

        const PMacc::math::Int<dim> lowerMargin = LowerMargin::toRT();

        using namespace lambda;
        DECLARE_PLACEHOLDERS(); // declares _1, _2, _3, ... in device code

        /* fill shared memory with global field data */
        algorithm::cudaBlock::Foreach<MappingDesc::SuperCellSize> foreach(linearThreadIdx);
        foreach(
            CachedBuffer::Zone(), /* super cell size + margins */
            cachedFieldCursor,
            this->globalFieldCursor(blockCell - lowerMargin),
            _1 = _2);

        __syncthreads();

        /* jump to origin of the SuperCell */
        //this->cachedFieldCursor = this->cachedFieldCursor(lowerMargin);
        if(linearThreadIdx == 0)
        {
            printf("cached marker: %x\n", this->cachedFieldCursor.marker);
        }
        PMACC_AUTO(shCursor, this->cachedFieldCursor(lowerMargin));
        if(linearThreadIdx == 0)
        {
            /*printf("shCursor marker: %x\n", shCursor.marker);
            printf("shCursor adress: %x\n", &(*shCursor));*/
            //printf("shCursor back-shifted: %x\n", &((*shCursor(-1,-1,-1))[2]));
            printf("cached adress: %x\n", &((*this->cachedFieldCursor(0,0,0))[2]));

            /*printf("fieldB global A: %f, %f, %f\n", (*this->globalFieldCursor(blockCell - lowerMargin)(0,0,0))[0],
                                                    (*this->globalFieldCursor(blockCell - lowerMargin)(0,0,0))[1],
                                                    (*this->globalFieldCursor(blockCell - lowerMargin)(0,0,0))[2]);
            printf("fieldB global B: %f, %f, %f\n", (*this->globalFieldCursor(blockCell - lowerMargin)(4,5,3))[0],
                                                    (*this->globalFieldCursor(blockCell - lowerMargin)(4,5,3))[1],
                                                    (*this->globalFieldCursor(blockCell - lowerMargin)(4,5,3))[2]);
            printf("fieldB global C: %f, %f, %f\n", (*this->globalFieldCursor(blockCell - lowerMargin)(9,9,5))[0],
                                                    (*this->globalFieldCursor(blockCell - lowerMargin)(9,9,5))[1],
                                                    (*this->globalFieldCursor(blockCell - lowerMargin)(9,9,5))[2]);*/

            ((*this->cachedFieldCursor(0,0,0))[2])=42;
            float_X* data = &((*shCursor(-1,-1,-1))[2]);
            printf("data adress: %x\n", data);
            printf("data value: %f\n", *data);

            printf("fieldB shared A: %f, %f, %f\n", (*shCursor(0,0,0))[2],
                                                    (*shCursor(0,0,0))[1],
                                                    (*shCursor(0,0,0))[0]);
            printf("fieldB shared B: %f, %f, %f\n", *data,
                                                    (*shCursor(-1,-1,-1))[1],
                                                    (*shCursor(-1,-1,-1))[0]);
            printf("fieldB shared C: %f, %f, %f\n", (*shCursor(8,8,4))[2],
                                                    (*shCursor(8,8,4))[1],
                                                    (*shCursor(8,8,4))[0]);

            printf("fieldB shared D: %f, %f, %f\n", (*this->cachedFieldCursor(0,0,0))[2],
                                                    (*this->cachedFieldCursor(0,0,0))[1],
                                                    (*this->cachedFieldCursor(0,0,0))[0]);
            printf("fieldB shared E: %f, %f, %f\n", (*this->cachedFieldCursor(4,5,3))[2],
                                                    (*this->cachedFieldCursor(4,5,3))[1],
                                                    (*this->cachedFieldCursor(4,5,3))[0]);
            printf("fieldB shared F: %f, %f, %f\n", (*this->cachedFieldCursor(9,9,5))[2],
                                                    (*this->cachedFieldCursor(9,9,5))[1],
                                                    (*this->cachedFieldCursor(9,9,5))[0]);
        }
    }

    /** Perform field-to-particle interpolation.
     *
     * @param particlePos particle position within cell
     * @param localCell multi-dim coordinate of the local cell inside the super cell
     * @return interpolated field value
     */
    DINLINE
    type operator()(const floatD_X& particlePos, const PMacc::math::Int<dim> localCell)
    {
        if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0)
        {
            printf("op(), cached shifted: %x\n", &(*this->cachedFieldCursor));
            /*printf("op(), fieldB shared A: %f, %f, %f\n", (*this->cachedFieldCursor(0,0,0))[0],
                                                          (*this->cachedFieldCursor(0,0,0))[1],
                                                          (*this->cachedFieldCursor(0,0,0))[2]);
            printf("op(), fieldB shared B: %f, %f, %f\n", (*this->cachedFieldCursor(-1,-1,-1))[0],
                                                          (*this->cachedFieldCursor(-1,-1,-1))[1],
                                                          (*this->cachedFieldCursor(-1,-1,-1))[2]);
            printf("op(), fieldB shared C: %f, %f, %f\n", (*this->cachedFieldCursor(8,8,4))[0],
                                                          (*this->cachedFieldCursor(8,8,4))[1],
                                                          (*this->cachedFieldCursor(8,8,4))[2]);*/
        }
        return *this->cachedFieldCursor(0,0,0);/* +
               *this->cachedFieldCursor(-1,-1,-1) +
               *this->cachedFieldCursor(8,8,4);*/
        /* interpolation of the field */
        /*return FieldToParticleInterpolation()(this->cachedFieldCursor(localCell),
                                              particlePos,
                                              fieldSolver::NumericalCellType::getBFieldPosition() NumericalFieldPosition()());*/
    }
};

} // namespace picongpu
