/* Copyright 2013-2023 Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund, Axel Huebl
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

#include "pmacc/algorithms/math.hpp"
#include "pmacc/math/vector/accessor/StandardAccessor.hpp"
#include "pmacc/math/vector/navigator/StandardNavigator.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/static_assert.hpp"
#include "pmacc/types.hpp"

#include <boost/call_traits.hpp>

#include <iostream>
#include <type_traits>

namespace pmacc
{
    namespace math
    {
        namespace detail
        {
            template<typename T_Type, uint32_t T_dim>
            struct Vector_components
            {
                static constexpr bool isConst = false;
                static constexpr uint32_t dim = T_dim;
                using type = T_Type;

                HDINLINE
                constexpr Vector_components()
                {
                }

                HDINLINE
                constexpr Vector_components& operator=(const Vector_components&) = default;

                HDINLINE
                constexpr Vector_components(const Vector_components&) = default;

                /*align full vector*/
                PMACC_ALIGN(v[dim], type);

                HDINLINE
                type& operator[](const uint32_t idx)
                {
                    return v[idx];
                }

                HDINLINE
                const type& operator[](const uint32_t idx) const
                {
                    return v[idx];
                }
            };

        } // namespace detail

        namespace tag
        {
            struct Vector;
        }

        template<
            typename T_Type,
            uint32_t T_dim,
            typename T_Accessor = StandardAccessor,
            typename T_Navigator = StandardNavigator,
            typename T_Storage = detail::Vector_components<T_Type, T_dim>>
        struct Vector
            : private T_Storage
            , protected T_Accessor
            , protected T_Navigator
        {
            using Storage = T_Storage;
            using type = typename Storage::type;
            static constexpr uint32_t dim = Storage::dim;
            using tag = tag::Vector;
            using Accessor = T_Accessor;
            using Navigator = T_Navigator;
            using ParamType = typename boost::call_traits<type>::param_type;

            /*Vectors without elements are not allowed*/
            PMACC_CASSERT_MSG(math_Vector__with_DIM_0_is_not_allowed, dim > 0u);

            HDINLINE
            constexpr Vector()
            {
            }

            /** Constructor for N-dimensional vector
             *
             * @attention This constructor allows implicit casts.
             *
             * @param args value of each dimension, x,y,z,...
             */
            template<typename... T_Args, typename = std::enable_if_t<(std::is_convertible_v<T_Args, T_Type> && ...)>>
            HDINLINE constexpr Vector(T_Args... args)
            {
                static_assert(sizeof...(T_Args) == dim, "Number of arguments must be equal to the vector dimension.");
                int i = 0;
                ([&] { (*this)[i++] = args; }(), ...);
            }

            HDINLINE
            constexpr Vector(const Vector& other) = default;

            template<typename T_OtherAccessor, typename T_OtherNavigator, typename T_OtherStorage>
            HDINLINE Vector(const Vector<T_Type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
                : Storage{}
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] = other[i];
            }

            template<
                typename T_OtherType,
                typename T_OtherAccessor,
                typename T_OtherNavigator,
                typename T_OtherStorage>
            HDINLINE explicit Vector(
                const Vector<T_OtherType, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] = static_cast<type>(other[i]);
            }

            /** Allow static_cast / explicit cast to member type for 1D vector */
            template<uint32_t T_deferDim = T_dim, typename = typename std::enable_if<T_deferDim == 1u>::type>
            HDINLINE explicit operator type()
            {
                return (*this)[0];
            }

            /**
             * Creates a Vector where all dimensions are set to the same value
             *
             * @param value Value which is set for all dimensions
             * @return new Vector<...>
             */
            HDINLINE
            static Vector create(ParamType value)
            {
                Vector result;
                for(uint32_t i = 0u; i < dim; i++)
                    result[i] = value;

                return result;
            }

            HDINLINE const Vector& toRT() const
            {
                return *this;
            }

            HDINLINE Vector& toRT()
            {
                return *this;
            }

            HDINLINE Vector revert()
            {
                Vector invertedVector;
                for(uint32_t i = 0u; i < dim; i++)
                    invertedVector[dim - 1 - i] = (*this)[i];

                return invertedVector;
            }

            HDINLINE Vector& operator=(const Vector&) = default;

            template<typename T_OtherAccessor, typename T_OtherNavigator, typename T_OtherStorage>
            HDINLINE Vector& operator=(const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] = rhs[i];
                return *this;
            }

            HDINLINE
            type& operator[](const uint32_t idx)
            {
                return Accessor::operator()(Storage::operator[](Navigator::operator()(idx)));
            }

            HDINLINE
            const type& operator[](const uint32_t idx) const
            {
                return Accessor::operator()(Storage::operator[](Navigator::operator()(idx)));
            }

            HDINLINE type& x()
            {
                return (*this)[0];
            }

            HDINLINE type& y()
            {
                PMACC_CASSERT_MSG(math_Vector__access_to_y_is_not_allowed_for_DIM_lesser_than_2, dim >= 2u);
                return (*this)[1];
            }

            HDINLINE type& z()
            {
                PMACC_CASSERT_MSG(math_Vector__access_to_z_is_not_allowed_for_DIM_lesser_than_3, dim >= 3u);
                return (*this)[2];
            }

            HDINLINE const type& x() const
            {
                return (*this)[0];
            }

            HDINLINE const type& y() const
            {
                PMACC_CASSERT_MSG(math_Vector__access_to_y_is_not_allowed_for_DIM_lesser_than_2, dim >= 2u);
                return (*this)[1];
            }

            HDINLINE const type& z() const
            {
                PMACC_CASSERT_MSG(math_Vector__access_to_z_is_not_allowed_for_DIM_lesser_than_3, dim >= 3u);
                return (*this)[2];
            }

            /** Shrink the number of elements of a vector.
             *
             * @tparam T_numElements New dimension of the vector.
             * @return First T_numElements elements of the origin vector
             */
            template<uint32_t T_numElements>
            HDINLINE Vector<type, T_numElements, Accessor, Navigator> shrink() const
            {
                PMACC_CASSERT_MSG(
                    math_Vector__T_numElements_must_be_lesser_or_equal_to_Vector_DIM,
                    T_numElements <= dim);
                Vector<type, T_numElements, Accessor, Navigator> result;
                for(uint32_t i = 0u; i < T_numElements; i++)
                    result[i] = (*this)[i];
                return result;
            }

            /** Shrink the number of elements of a vector.
             *
             * @tparam T_numElements New dimension of the vector.
             * @param startIdx Index within the origin vector which will be the first element in the result.
             * @return T_numElements elements of the origin vector starting with the index startIdx.
             *         Indexing will wrapp around when the end of the origin vector is reached.
             */
            template<uint32_t T_numElements>
            HDINLINE Vector<type, T_numElements, Accessor, Navigator> shrink(const int startIdx) const
            {
                PMACC_CASSERT_MSG(
                    math_Vector_T_numElements_must_be_lesser_or_equal_to_Vector_DIM,
                    T_numElements <= dim);
                Vector<type, T_numElements, Accessor, Navigator> result;
                for(uint32_t i = 0u; i < T_numElements; i++)
                    result[i] = (*this)[(startIdx + i) % dim];
                return result;
            }

            /** Removes a component
             *
             * It is not allowed to call this method on a vector with the dimensionality of one.
             *
             * @tparam dimToRemove index which shall be removed; range: [ 0; dim - 1 ]
             * @return vector with `dim - 1` elements
             */
            template<uint32_t dimToRemove>
            HDINLINE Vector<type, dim - 1, Accessor, Navigator> remove() const
            {
                PMACC_CASSERT_MSG(__math_Vector__dim_must_be_greater_than_1__, dim > 1u);
                PMACC_CASSERT_MSG(__math_Vector__dimToRemove_must_be_lesser_than_dim__, dimToRemove < dim);
                Vector<type, dim - 1, Accessor, Navigator> result;
                for(uint32_t i = 0u; i < dim - 1; ++i)
                {
                    // skip component which must be deleted
                    const int sourceIdx = i >= dimToRemove ? i + 1 : i;
                    result[i] = (*this)[sourceIdx];
                }
                return result;
            }

            /** Returns product of all components.
             *
             * @return product of components
             */
            HDINLINE type productOfComponents() const
            {
                type result = (*this)[0];
                for(uint32_t i = 1u; i < dim; i++)
                    result *= (*this)[i];
                return result;
            }

            /** Returns sum of all components.
             *
             * @return sum of components
             */
            HDINLINE type sumOfComponents() const
            {
                type result = (*this)[0];
                for(uint32_t i = 1u; i < dim; i++)
                    result += (*this)[i];
                return result;
            }

            /*! += operator
             * @param other instance with same type and dimension like the left instance
             * @return reference to manipulated left instance
             */
            template<typename T_OtherAccessor, typename T_OtherNavigator, typename T_OtherStorage>
            HDINLINE Vector& operator+=(
                const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] += other[i];
                return *this;
            }

            /*! -= operator
             * @param other instance with same type and dimension like the left instance
             * @return reference to manipulated left instance
             */
            template<typename T_OtherAccessor, typename T_OtherNavigator, typename T_OtherStorage>
            HDINLINE Vector& operator-=(
                const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] -= other[i];
                return *this;
            }

            /*! *= operator
             * @param other instance with same type and dimension like the left instance
             * @return reference to manipulated left instance
             */
            template<typename T_OtherAccessor, typename T_OtherNavigator, typename T_OtherStorage>
            HDINLINE Vector& operator*=(
                const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] *= other[i];
                return *this;
            }

            /*! /= operator
             * @param other instance with same type and dimension like the left instance
             * @return reference to manipulated left instance
             */
            template<typename T_OtherAccessor, typename T_OtherNavigator, typename T_OtherStorage>
            HDINLINE Vector& operator/=(
                const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] /= other[i];
                return *this;
            }

            HDINLINE Vector& operator+=(ParamType other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] += other;
                return *this;
            }

            HDINLINE Vector& operator-=(ParamType other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] -= other;
                return *this;
            }

            HDINLINE Vector& operator*=(ParamType other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] *= other;
                return *this;
            }

            HDINLINE Vector& operator/=(ParamType other)
            {
                for(uint32_t i = 0u; i < dim; i++)
                    (*this)[i] /= other;
                return *this;
            }

            /**
             * == comparison operator.
             *
             * Compares sizes of two DataSpaces.
             *
             * @param other Vector to compare to
             * @return true if all components in both vectors are equal, else false
             */
            HDINLINE bool operator==(const Vector& rhs) const
            {
                bool result = true;
                for(uint32_t i = 0u; i < dim; i++)
                    result = result && ((*this)[i] == rhs[i]);
                return result;
            }

            /**
             * != comparison operator.
             *
             * Compares sizes of two DataSpaces.
             *
             * @param other Vector to compare to
             * @return true if one component in both vectors are not equal, else false
             */
            HDINLINE bool operator!=(const Vector& rhs) const
            {
                return !((*this) == rhs);
            }

            /** create string out of the vector
             *
             * @param separator string to separate components of the vector
             * @param enclosings string with size 2 to enclose vector
             *                   size == 0 ? no enclose symbols
             *                   size == 1 ? means enclose symbol begin and end are equal
             *                   size >= 2 ? letter[0] = begin enclose symbol
             *                               letter[1] = end enclose symbol
             *
             * example:
             * .toString(";","|")     -> |x;...;z|
             * .toString(",","[]")    -> [x,...,z]
             */
            std::string toString(const std::string separator = ",", const std::string enclosings = "{}") const
            {
                std::string locale_enclosing_begin;
                std::string locale_enclosing_end;
                size_t enclosing_size = enclosings.size();

                if(enclosing_size > 0)
                {
                    /* % avoid out of memory access */
                    locale_enclosing_begin = enclosings[0 % enclosing_size];
                    locale_enclosing_end = enclosings[1 % enclosing_size];
                }

                std::stringstream stream;
                stream << locale_enclosing_begin << (*this)[0];

                for(uint32_t i = 1u; i < dim; ++i)
                    stream << separator << (*this)[i];
                stream << locale_enclosing_end;
                return stream.str();
            }

            HDINLINE cupla::dim3 toDim3() const
            {
                cupla::dim3 result;
                unsigned int* ptr = &result.x;
                for(uint32_t d = 0u; d < dim; ++d)
                    ptr[d] = (*this)[d];
                return result;
            }
        };

        template<
            std::size_t I,
            typename T_Type,
            uint32_t T_dim,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage>
        auto get(const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& v)
        {
            return v[I];
        }

        template<
            std::size_t I,
            typename T_Type,
            uint32_t T_dim,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage>
        auto& get(Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& v)
        {
            return v[I];
        }

        template<typename Type>
        struct Vector<Type, 0>
        {
            using type = Type;
            static constexpr uint32_t dim = 0;

            template<typename OtherType>
            HDINLINE operator Vector<OtherType, 0>() const
            {
                return Vector<OtherType, 0>();
            }

            /**
             * == comparison operator.
             *
             * Returns always true
             */
            HDINLINE bool operator==(const Vector& rhs) const
            {
                return true;
            }

            /**
             * != comparison operator.
             *
             * Returns always false
             */
            HDINLINE bool operator!=(const Vector& rhs) const
            {
                return false;
            }

            HDINLINE
            static Vector create(Type)
            {
                /* this method should never be actually called,
                 * it exists only for Visual Studio to handle pmacc::math::Size_t< 0 >
                 */
                PMACC_CASSERT_MSG(Vector_dim_0_create_cannot_be_called, sizeof(Type) != 0 && false);
            }
        };

        template<typename Type, uint32_t dim, typename Accessor, typename Navigator>
        std::ostream& operator<<(std::ostream& s, const Vector<Type, dim, Accessor, Navigator>& vec)
        {
            return s << vec.toString();
        }

        template<
            typename T_Type,
            uint32_t T_dim,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage>
        HDINLINE Vector<T_Type, T_dim> operator+(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(lhs);
            result += rhs;
            return result;
        }

        template<typename T_Type, uint32_t T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
        HDINLINE Vector<T_Type, T_dim> operator+(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            typename Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>::ParamType rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(lhs);
            result += rhs;
            return result;
        }

        template<
            typename T_Type,
            uint32_t T_dim,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage>
        HDINLINE Vector<T_Type, T_dim> operator-(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(lhs);
            result -= rhs;
            return result;
        }

        template<typename T_Type, uint32_t T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
        HDINLINE Vector<T_Type, T_dim> operator-(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            typename Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>::ParamType rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(lhs);
            result -= rhs;
            return result;
        }

        template<
            typename T_Type,
            uint32_t T_dim,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage>
        HDINLINE Vector<T_Type, T_dim> operator*(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(lhs);
            result *= rhs;
            return result;
        }

        template<
            typename T_Type,
            uint32_t T_dim,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage>
        HDINLINE Vector<T_Type, T_dim> operator/(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(lhs);
            result /= rhs;
            return result;
        }

        template<typename T_Type, uint32_t T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
        HDINLINE Vector<T_Type, T_dim> operator*(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            typename Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>::ParamType rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(lhs);
            result *= rhs;
            return result;
        }

        template<typename T_Type, uint32_t T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
        HDINLINE Vector<T_Type, T_dim> operator*(
            typename boost::call_traits<T_Type>::param_type lhs,
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(rhs);
            result *= lhs;
            return result;
        }

        template<typename T_Type, uint32_t T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
        HDINLINE Vector<T_Type, T_dim> operator/(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            typename Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>::ParamType rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(lhs);
            result /= rhs;
            return result;
        }

        template<typename T_Type, uint32_t T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
        HDINLINE Vector<T_Type, T_dim> operator-(const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& vec)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_dim> result(vec);

            for(uint32_t i = 0u; i < T_dim; i++)
                result[i] = -result[i];
            return result;
        }

        template<
            typename T_Type,
            uint32_t T_dim,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage>
        HDINLINE Vector<bool, T_dim> operator>=(
            const Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<bool, T_dim> result;
            for(uint32_t i = 0u; i < T_dim; ++i)
                result[i] = (lhs[i] >= rhs[i]);
            return result;
        }

        /** Give the linear index of an N-dimensional index within an N-dimensional index space.
         *
         * @tparam T_IntegralType vector data type (must be an integral type)
         * @tparam T_dim dimension of the vector, should be >= 2
         * @param size N-dimensional size of the index space (N can be one dimension less compared to idx)
         * @param idx N-dimensional index within the index space
         *            @attention behaviour is undefined for negative index
         *            @attention if idx is outside of size the result will be outside of the the index domain too
         * @return linear index within the index domain
         *
         * @{
         */
        template<
            typename T_IntegralType,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage,
            uint32_t T_dim,
            typename = std::enable_if_t<std::is_integral_v<T_IntegralType> && T_dim >= DIM2>>
        HDINLINE T_IntegralType linearize(
            const Vector<T_IntegralType, T_dim - 1u, T_Accessor, T_Navigator, T_Storage>& size,
            const Vector<T_IntegralType, T_dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& idx)
        {
            T_IntegralType linearIdx{idx[T_dim - 1u]};
            for(int d = T_dim - 2; d >= 0; --d)
                linearIdx = linearIdx * size[d] + idx[d];

            return linearIdx;
        }

        template<
            typename T_IntegralType,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage,
            uint32_t T_dim,
            typename = std::enable_if_t<std::is_integral_v<T_IntegralType>>>
        HDINLINE T_IntegralType linearize(
            const Vector<T_IntegralType, T_dim, T_Accessor, T_Navigator, T_Storage>& size,
            const Vector<T_IntegralType, T_dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& idx)
        {
            return linearize(size.template shrink<T_dim - 1u>(), idx);
        }

        /** @} */

        /** Maps a linear index to an N-dimensional index
         *
         * @tparam T_IntegralType vector data type (must be an integral type)
         * @param size N-dimensional index space
         * @param linearIdx Linear index within size.
         *        @attention If linearIdx is an index outside of size the result will be outside of the index domain
         * too.
         * @return N-dimensional index
         *
         * @{
         */
        template<
            typename T_IntegralType,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            uint32_t T_dim,
            typename = std::enable_if_t<std::is_integral_v<T_IntegralType> && T_dim >= DIM2>>
        HDINLINE auto mapToND(
            const Vector<T_IntegralType, T_dim, T_Accessor, T_Navigator, T_Storage>& size,
            T_IntegralType linearIdx)
        {
            Vector<T_IntegralType, T_dim - 1u, T_Accessor, T_Navigator, T_Storage> pitchExtents;
            pitchExtents[0] = size[0];
            for(uint32_t d = 1u; d < T_dim - 1u; ++d)
                pitchExtents[d] = size[d] * pitchExtents[d - 1u];

            Vector<T_IntegralType, T_dim> result;
            for(uint32_t d = T_dim - 1u; d >= 1u; --d)
            {
                result[d] = linearIdx / pitchExtents[d - 1];
                linearIdx -= pitchExtents[d - 1] * result[d];
            }
            result[0] = linearIdx;
            return result;
        }

        template<
            typename T_IntegralType,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename = std::enable_if_t<std::is_integral_v<T_IntegralType>>>
        HDINLINE auto mapToND(
            const Vector<T_IntegralType, DIM1, T_Accessor, T_Navigator, T_Storage>& size,
            T_IntegralType linearIdx)
        {
            return linearIdx;
        }

        /** @} */

        template<typename Lhs, typename Rhs>
        HDINLINE Lhs operator%(const Lhs& lhs, const Rhs& rhs)
        {
            Lhs result;

            for(uint32_t i = 0u; i < Lhs::dim; i++)
                result[i] = lhs[i] % rhs[i];
            return result;
        }

        /** Get the unit basis vector of the given type along the given direction
         *
         * In case 0 <= T_direction < T_Vector::dim, return the basis vector with value
         * 1 in component T_direction and 0 in other components, otherwise return the
         * zero vector.
         *
         * @tparam T_Vector result type
         * @tparam T_direction index of the basis vector direction
         */
        template<typename T_Vector, uint32_t T_direction>
        HDINLINE T_Vector basisVector();

    } // namespace math
} // namespace pmacc

namespace std
{
    template<typename T_Type, uint32_t T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
    struct tuple_size<pmacc::math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>>
    {
        static constexpr std::size_t value = T_dim;
    };

    template<
        std::size_t I,
        typename T_Type,
        uint32_t T_dim,
        typename T_Accessor,
        typename T_Navigator,
        typename T_Storage>
    struct tuple_element<I, pmacc::math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>>
    {
        using type = T_Type;
    };
} // namespace std
