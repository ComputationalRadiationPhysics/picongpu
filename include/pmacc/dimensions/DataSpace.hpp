/* Copyright 2013-2019 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
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

#include "pmacc/math/Vector.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{

    /**
     * A DIM-dimensional data space.
     *
     * DataSpace describes a DIM-dimensional data space with a specific size for each dimension.
     * It only describes the space and does not hold any actual data.
     *
     * @tparam DIM dimension (1-3) of the dataspace
     */
    template <unsigned DIM>
    class DataSpace : public math::Vector<int,DIM>
    {
    public:

        static constexpr int Dim=DIM;
        typedef math::Vector<int,DIM> BaseType;

        /**
         * default constructor.
         * Sets size of all dimensions to 0.
         */
        HDINLINE DataSpace()
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                (*this)[i] = 0;
            }
        }

        /**
         * constructor.
         * Sets size of all dimensions from cuda dim3.
         */
        HDINLINE DataSpace(dim3 value)
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                (*this)[i] = *(&(value.x) + i);
            }
        }

        /**
         * constructor.
         * Sets size of all dimensions from cuda uint3 (e.g. threadIdx/blockIdx)
         */
        HDINLINE DataSpace(uint3 value)
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                (*this)[i] = *(&(value.x) + i);
            }
        }

        HDINLINE DataSpace(const DataSpace<DIM>& value) : BaseType(value)
        {
        }

        /**
         * Constructor for DIM1-dimensional DataSpace.
         *
         * @param x size of first dimension
         */
        HDINLINE DataSpace(int x) : BaseType(x)
        {
        }

        /**
         * Constructor for DIM2-dimensional DataSpace.
         *
         * @param x size of first dimension
         * @param y size of second dimension
         */
        HDINLINE DataSpace(int x, int y) : BaseType(x, y)
        {
        }

        /**
         * Constructor for DIM3-dimensional DataSpace.
         *
         * @param x size of first dimension
         * @param y size of second dimension
         * @param z size of third dimension
         */
        HDINLINE DataSpace(int x, int y, int z) : BaseType(x, y, z)
        {
        }

        HDINLINE DataSpace(const BaseType& vec) : BaseType(vec)
        {
        }

        HDINLINE DataSpace(const math::Size_t<DIM>& vec)
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                (*this)[i] = vec[i];
            }
        }

        /**
         * Give DataSpace where all dimensions set to init value
         *
         * @param value value which is setfor all dimensions
         * @return the new DataSpace
         */
        HDINLINE static DataSpace<DIM> create(int value = 1)
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = value;
            }
            return tmp;
        }

        /**
         * Returns number of dimensions (DIM) of this DataSpace.
         *
         * @return number of dimensions
         */
        HDINLINE int getDim() const
        {
            return DIM;
        }

        /**
         * Evaluates if one dimension is greater than the respective dimension of other.
         *
         * @param other DataSpace to compare with
         * @return true if one dimension is greater, false otherwise
         */
        HINLINE bool isOneDimensionGreaterThan(const DataSpace<DIM>& other) const
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                if ((*this)[i] > other[i])
                    return true;
            }
            return false;
        }

        HDINLINE operator math::Size_t<DIM>() const
        {
            math::Size_t<DIM> result;
            for (uint32_t i = 0; i < DIM; i++)
                result[i] = (size_t) (*this)[i];
            return result;
        }

        HDINLINE operator dim3() const
        {
            return this->toDim3();
        }

    };

} //namespace pmacc

#include "pmacc/dimensions/DataSpace.tpp"
