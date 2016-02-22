/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc_types.hpp"

/**
 * \todo
 */
#define TO_BITS(x) (~((-1) << (x)))

namespace PMacc
{

    /**
     * Represents single bits of data.
     *
     * @param TYPE
     * @param NUMBITS
     */
    template<class TYPE, unsigned NUMBITS>
    class BitData
    {
    protected:
        TYPE* data;
        uint16_t bit;
    public:

        /**
         * constructor
         * @param data
         * @param bit
         */
        HDINLINE BitData(TYPE* data, uint16_t bit) : data(data), bit(bit)
        {
        }

        HDINLINE operator TYPE() const;

        /**
         * set a value to a bit field
         * WARNING: if rhs is greater than number of bits,
         * other bit fields are destroyed
         */
        HDINLINE void operator=(const TYPE &rhs);

        /**
         * set represented bit(s) to 0 (null)
         */
        HDINLINE void setBitsToNull();

        /**
         * add any value to the bits
         * WARNING: if sum of bits is greater than number of bits,
         * other bit fields are destroyed
         * @param rhs
         */
        HDINLINE void operator+=(const TYPE &rhs);

        /**
         * container  operation
         * @param value set data to value
         */
        HDINLINE void setData(TYPE value);

        /**
         * container operation
         */
        HDINLINE void copyDataFrom(const BitData<TYPE, NUMBITS>& rhs);

        HDINLINE void operator=(const BitData<TYPE, NUMBITS>& rhs);

    };

    /**
     * Specialized BitData with NUMBITS=1u
     */
    template<class TYPE>
    class BitData<TYPE, 1u >
    {
    public:

        HDINLINE BitData(TYPE* data, uint16_t bit) : data(data), bit(bit)
        {
        }
        HDINLINE void operator=(const BitData<TYPE, 1u > & rhs);
        HDINLINE operator TYPE() const;
        HDINLINE void operator=(const TYPE &rhs);
        HDINLINE void setBitsToNull();
        HDINLINE void operator+=(const TYPE &rhs);
        HDINLINE void setData(TYPE value);
        HDINLINE void copyDataFrom(const BitData<TYPE, 1u > & rhs);
    private:
        TYPE* data;
        uint16_t bit;
    };

    template<class TYPE, unsigned NUMBITS>
    HDINLINE void BitData<TYPE, NUMBITS>::operator=(const BitData<TYPE, NUMBITS>& rhs)
    {
        *this = (TYPE) (rhs);
    }

    template<class TYPE>
    HDINLINE void BitData<TYPE, 1u > ::operator=(const BitData<TYPE, 1u > & rhs)
    {
        *this = (TYPE) (rhs);
    }

    template<class TYPE, unsigned NUMBITS>
    HDINLINE BitData<TYPE, NUMBITS>::operator TYPE() const
    {
        return (*this->data >> bit) & TO_BITS(NUMBITS);
    }

    template<class TYPE>
    HDINLINE BitData<TYPE, 1u > ::operator TYPE() const
    {
        return (*this->data >> bit) & 1u;
    }

    template<class TYPE, unsigned NUMBITS>
    HDINLINE void BitData<TYPE, NUMBITS>::operator+=(const TYPE &rhs)
    {
#if !defined(__CUDA_ARCH__) // Host code path
        *(this->data) += (rhs << this->bit);
#else
        atomicAdd(this->data, rhs << this->bit);
#endif
    }

    template<class TYPE, unsigned NUMBITS>
    HDINLINE void BitData<TYPE, NUMBITS>::setBitsToNull()
    {
#if !defined(__CUDA_ARCH__) // Host code path
        *(this->data) &= ~(TO_BITS(NUMBITS) << this->bit);
#else
        atomicAnd(this->data, ~(TO_BITS(NUMBITS) << this->bit));
#endif
    }

    template<class TYPE, unsigned NUMBITS>
    HDINLINE void BitData<TYPE, NUMBITS>::operator=(const TYPE &rhs)
    {
#if !defined(__CUDA_ARCH__) // Host code path
        setBitsToNull();
        *(this->data) |= (rhs << this->bit);
#else
        setBitsToNull();
        atomicOr(this->data, (rhs << this->bit));
#endif
    }

    template<class TYPE>
    HDINLINE void BitData<TYPE, 1u > ::operator=(const TYPE &rhs)
    {
#if !defined(__CUDA_ARCH__) // Host code path
        if (rhs)
            *(this->data) |= (1u << this->bit);
        else
            *(this->data) &= (~(1u << this->bit));
#else
        if (rhs)
            atomicOr(this->data, 1u << this->bit);
        else
            atomicAnd(this->data, ~(1u << this->bit));
#endif
    }

    template<class TYPE, unsigned NUMBITS>
    HDINLINE void BitData<TYPE, NUMBITS>::copyDataFrom(const BitData<TYPE, NUMBITS>& rhs)
    {
        *data = *(rhs.data);
    }

    template<class TYPE, unsigned NUMBITS>
    HDINLINE void BitData<TYPE, NUMBITS>::setData(TYPE value)
    {
        *data = value;
    }


}
