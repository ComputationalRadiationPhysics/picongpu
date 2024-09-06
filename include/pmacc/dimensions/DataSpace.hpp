/* Copyright 2013-2023 Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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

#include <type_traits>

namespace pmacc
{
    /** A T_dim-dimensional vector of MemIdxType.
     *
     * The object is used to describe extents/number of elements for buffer allocations.
     * It only describes the space and does not hold any actual data.
     *
     * @tparam T_dim dimension N of the dataspace
     */
    template<uint32_t T_dim>
    using MemSpace = math::Vector<MemIdxType, T_dim>;

    /** A T_dim-dimensional vector of integers.
     *
     * The object is used to describe indices of threads in the kernel.
     * @attention Currently this object is also used to describe memory extents. This should be avoided and is a
     * technical dept from the past and limits the index space to 2 giga elements.
     *
     * DataSpace describes a T_dim-dimensional data space with a specific size for each dimension.
     * It only describes the space and does not hold any actual data.
     *
     * @tparam T_dim dimension N of the dataspace
     */
    template<unsigned T_dim>
    class DataSpace : public math::Vector<int, T_dim>
    {
    public:
        static constexpr uint32_t Dim = T_dim;
        using BaseType = math::Vector<int, T_dim>;

        /** default constructor.
         *
         * Sets size of all dimensions to 0.
         */
        constexpr DataSpace() : BaseType(BaseType::create(0))
        {
        }

        constexpr DataSpace(const DataSpace&) = default;

        HDINLINE constexpr DataSpace& operator=(const DataSpace&) = default;

        /** constructor.
         *
         * Sets size of all dimensions from alpaka.
         */
        template<typename T_MemberType>
        HDINLINE explicit DataSpace(alpaka::Vec<::alpaka::DimInt<T_dim>, T_MemberType> const& value)
        {
            for(uint32_t i = 0u; i < T_dim; i++)
                (*this)[T_dim - 1 - i] = value[i];
        }

        /** Constructor for N-dimensional DataSpace.
         *
         * @attention This constructor allows implicit casts.
         *
         * @param args size of each dimension, x,y,z,...
         */
        template<typename... T_Args, typename = std::enable_if_t<(std::is_convertible_v<T_Args, int> && ...)>>
        constexpr DataSpace(T_Args&&... args) : BaseType(std::forward<T_Args>(args)...)
        {
            static_assert(sizeof...(T_Args) == T_dim, "Number of arguments must be equal to the DataSpace dimension.");
        }

        constexpr DataSpace(const BaseType& vec) : BaseType(vec)
        {
        }

        HDINLINE DataSpace(const math::Size_t<T_dim>& vec)
        {
            for(uint32_t i = 0; i < T_dim; ++i)
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
        HDINLINE static DataSpace<T_dim> create(int value = 1)
        {
            DataSpace<T_dim> tmp;
            for(uint32_t i = 0; i < T_dim; ++i)
            {
                tmp[i] = value;
            }
            return tmp;
        }

        /**
         * Returns number of dimensions (T_dim) of this DataSpace.
         *
         * @return number of dimensions
         */
        HDINLINE int getDim() const
        {
            return T_dim;
        }

        /**
         * Evaluates if one dimension is greater than the respective dimension of other.
         *
         * @param other DataSpace to compare with
         * @return true if one dimension is greater, false otherwise
         */
        HINLINE bool isOneDimensionGreaterThan(const MemSpace<T_dim>& other) const
        {
            for(uint32_t i = 0; i < T_dim; ++i)
            {
                if((*this)[i] > other[i])
                    return true;
            }
            return false;
        }

        HDINLINE operator math::Size_t<T_dim>() const
        {
            math::Size_t<T_dim> result;
            for(uint32_t i = 0; i < T_dim; i++)
                result[i] = static_cast<size_t>((*this)[i]);
            return result;
        }
    };

} // namespace pmacc

#include "pmacc/dimensions/DataSpace.tpp"
