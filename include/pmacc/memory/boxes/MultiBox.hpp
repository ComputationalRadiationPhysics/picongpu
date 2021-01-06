/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"

namespace pmacc
{
    namespace mutiBoxAccess
    {
        template<typename Type>
        class MutiBoxAccess
        {
        public:
            typedef Type ValueType;
            typedef ValueType& RefValueType;

            HDINLINE MutiBoxAccess(ValueType* ptr, const size_t offset) : offset(offset), ptr((char*) ptr)
            {
            }

            HDINLINE RefValueType operator[](const uint32_t idx)
            {
                return *((ValueType*) (ptr + (idx * offset)));
            }

            HDINLINE RefValueType operator[](const uint32_t idx) const
            {
                return *((ValueType*) (ptr + (idx * offset)));
            }

        private:
            PMACC_ALIGN(offset, const size_t);
            PMACC_ALIGN(ptr, const char*);
        };

    } // namespace mutiBoxAccess

    template<typename Type, unsigned DIM>
    class MultiBox;

    template<typename Type>
    class MultiBox<Type, DIM1>
    {
    private:
        typedef DataBox<PitchedBox<Type, DIM1>> DataBoxType;

    public:
        enum
        {
            Dim = DIM1
        };
        typedef mutiBoxAccess::MutiBoxAccess<Type> ValueType;
        typedef mutiBoxAccess::MutiBoxAccess<Type> RefValueType;
        typedef MultiBox<Type, DIM1> ReducedType;

        HDINLINE DataBoxType getDataBox(uint32_t nameId)
        {
            return DataBoxType(PitchedBox<Type, DIM1>((Type*) ((char*) fixedPointer + attributePitch * nameId)));
        }

        HDINLINE RefValueType operator[](const int idx)
        {
            return RefValueType(fixedPointer + idx, attributePitch);
        }

        HDINLINE RefValueType operator[](const int idx) const
        {
            return RefValueType(fixedPointer + idx, attributePitch);
        }

        HDINLINE MultiBox(Type* pointer, const DataSpace<DIM1>& offset, const DataSpace<DIM1>&, const size_t pitch)
            : attributePitch(pitch)
            , fixedPointer(pointer + offset[0])
        {
        }

        HDINLINE MultiBox(Type* pointer, const size_t attributePitch)
            : attributePitch(attributePitch)
            , fixedPointer(pointer)
        {
        }

        /*Object must init by copy a valid instance*/
        HDINLINE MultiBox()
        {
        }

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return RefValueType(fixedPointer, attributePitch);
        }

        HDINLINE Type const* getPointer() const
        {
            return fixedPointer;
        }
        HDINLINE Type* getPointer()
        {
            return fixedPointer;
        }


    protected:
        PMACC_ALIGN(attributePitch, size_t);
        PMACC_ALIGN(fixedPointer, Type*);
    };

    template<typename Type>
    class MultiBox<Type, DIM2>
    {
    private:
        typedef DataBox<PitchedBox<Type, DIM2>> DataBoxType;

    public:
        enum
        {
            Dim = DIM2
        };
        typedef mutiBoxAccess::MutiBoxAccess<Type> ValueType;
        typedef mutiBoxAccess::MutiBoxAccess<Type> RefValueType;
        typedef MultiBox<Type, DIM1> ReducedType;

        HDINLINE DataBoxType getDataBox(uint32_t nameId)
        {
            return DataBoxType(
                PitchedBox<Type, DIM2>((Type*) ((char*) fixedPointer + attributePitch * nameId), pitch));
        }

        HDINLINE MultiBox(
            Type* pointer,
            const DataSpace<DIM2>& offset,
            const DataSpace<DIM2>& memSize,
            const size_t pitch)
            : pitch(pitch)
            , attributePitch(pitch * memSize.y())
            , fixedPointer((Type*) ((char*) pointer + offset[1] * pitch) + offset[0])
        {
        }

        /*Object must init by copy a valid instance*/
        HDINLINE MultiBox()
        {
        }

        HDINLINE ReducedType operator[](const int idx)
        {
            return ReducedType((Type*) ((char*) this->fixedPointer + idx * pitch), attributePitch);
        }

        HDINLINE ReducedType operator[](const int idx) const
        {
            return ReducedType((Type*) ((char*) this->fixedPointer + idx * pitch), attributePitch);
        }

        HDINLINE MultiBox(Type* pointer, size_t pitch, size_t attributePitch)
            : pitch(pitch)
            , attributePitch(attributePitch)
            , fixedPointer(pointer)
        {
        }

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return RefValueType(fixedPointer, attributePitch);
        }

        HDINLINE Type const* getPointer() const
        {
            return fixedPointer;
        }
        HDINLINE Type* getPointer()
        {
            return fixedPointer;
        }

    protected:
        PMACC_ALIGN(pitch, size_t);
        PMACC_ALIGN(attributePitch, size_t);
        PMACC_ALIGN(fixedPointer, Type*);
    };

    template<typename Type>
    class MultiBox<Type, DIM3>
    {
    private:
        typedef DataBox<PitchedBox<Type, DIM3>> DataBoxType;

    public:
        enum
        {
            Dim = DIM3
        };
        typedef mutiBoxAccess::MutiBoxAccess<Type> ValueType;
        typedef mutiBoxAccess::MutiBoxAccess<Type> RefValueType;
        typedef MultiBox<Type, DIM2> ReducedType;

        HDINLINE DataBoxType getDataBox(uint32_t nameId)
        {
            return DataBoxType(
                PitchedBox<Type, DIM3>((Type*) ((char*) fixedPointer + attributePitch * nameId), pitch, pitch2D));
        }

        HDINLINE ReducedType operator[](const int idx)
        {
            return ReducedType((Type*) ((char*) (this->fixedPointer) + idx * pitch2D), pitch, attributePitch);
        }

        HDINLINE ReducedType operator[](const int idx) const
        {
            return ReducedType((Type*) ((char*) (this->fixedPointer) + idx * pitch2D), pitch, attributePitch);
        }

        /** constructor
         *
         * @param pointer pointer to the origin of the physical memory
         * @param offset offset (in elements)
         * @param memSize size of the physical memory (in elements)
         * @param pitch number of bytes in one line (first dimension)
         */
        HDINLINE MultiBox(
            Type* pointer,
            const DataSpace<DIM3>& offset,
            const DataSpace<DIM3>& memSize,
            const size_t pitch)
            : pitch(pitch)
            , pitch2D(memSize.y() * pitch)
            , attributePitch((memSize.y() * pitch) * size.z())
            , fixedPointer(
                  (Type*) ((char*) pointer + offset[2] * (memSize.y() * pitch) + offset[1] * pitch) + offset[0])
        {
        }

        /*Object must init by copy a valid instance*/
        HDINLINE MultiBox()
        {
        }

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return RefValueType(fixedPointer, attributePitch);
        }

        HDINLINE Type const* getPointer() const
        {
            return fixedPointer;
        }
        HDINLINE Type* getPointer()
        {
            return fixedPointer;
        }


        PMACC_ALIGN(pitch, size_t);
        PMACC_ALIGN(pitch2D, size_t);
        PMACC_ALIGN(attributePitch, size_t);
        PMACC_ALIGN(fixedPointer, Type*);
    };


} // namespace pmacc
