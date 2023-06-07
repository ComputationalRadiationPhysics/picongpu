/* Copyright 2013-2022 Heiko Burau, Rene Widera, Benjamin Worpitz,
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

#include <llama/llama.hpp>

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

            template<typename T_Type, int T_Dim, typename RecordRef>
            struct VectorLlamaRecordRefStorage
            {
                static_assert(llama::isRecordRef<RecordRef>);

                inline static constexpr bool isConst = false;
                inline static constexpr int dim = T_Dim;
                using type = T_Type;

                RecordRef rr;

                HDINLINE
                type& operator[](const int idx)
                {
                    return mp_with_index<T_Dim>(
                        idx,
                        [&](auto ic) LLAMA_LAMBDA_INLINE -> decltype(auto)
                        { return rr(llama::RecordCoord<decltype(ic)::value>{}); });
                }

                HDINLINE
                const type& operator[](const int idx) const
                {
                    return mp_with_index<T_Dim>(
                        idx,
                        [&](auto ic) LLAMA_LAMBDA_INLINE -> decltype(auto)
                        { return rr(llama::RecordCoord<decltype(ic)::value>{}); });
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

            HDINLINE
            constexpr explicit Vector(Storage s) : Storage{std::move(s)}
            {
            }

            HDINLINE
            constexpr Vector(const type x)
            {
                PMACC_CASSERT_MSG(math_Vector__constructor_is_only_allowed_for_DIM1, dim == 1u);
                (*this)[0] = x;
            }

            HDINLINE
            constexpr Vector(const type x, const type y)
            {
                PMACC_CASSERT_MSG(math_Vector__constructor_is_only_allowed_for_DIM2, dim == 2u);
                (*this)[0] = x;
                (*this)[1] = y;
            }

            HDINLINE
            constexpr Vector(const type x, const type y, const type z)
            {
                PMACC_CASSERT_MSG(math_Vector__constructor_is_only_allowed_for_DIM3, dim == 3u);
                (*this)[0] = x;
                (*this)[1] = y;
                (*this)[2] = z;
            }

            HDINLINE
            constexpr Vector(const Vector& other) = default;

            template<typename T_OtherAccessor, typename T_OtherNavigator>
            HDINLINE Vector(const Vector<T_Type, dim, T_OtherAccessor, T_OtherNavigator, Storage>& other)
                : Storage{static_cast<const Storage&>(other)}
            {
            }

            template<typename T_OtherAccessor, typename T_OtherNavigator, typename T_OtherStorage>
            HDINLINE Vector(const Vector<T_Type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
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

            template<uint32_t shrinkedDim>
            HDINLINE Vector<type, shrinkedDim, Accessor, Navigator> shrink(const int startIdx = 0) const
            {
                PMACC_CASSERT_MSG(
                    math_Vector__shrinkedDim_DIM_must_be_lesser_or_equal_to_Vector_DIM,
                    shrinkedDim <= dim);
                Vector<type, shrinkedDim, Accessor, Navigator> result;
                for(uint32_t i = 0u; i < shrinkedDim; i++)
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

        template<typename T>
        inline constexpr bool isVector = false;

        template<typename T_Type, int T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
        inline constexpr bool isVector<pmacc::math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>> = true;

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

        template<
            typename T_Type,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage>
        HDINLINE T_Type linearize(
            const Vector<T_Type, 1, T_Accessor, T_Navigator, T_Storage>& size,
            const Vector<T_Type, 2, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& pos)
        {
            return pos.y() * size.x() + pos.x();
        }

        template<
            typename T_Type,
            typename T_Accessor,
            typename T_Navigator,
            typename T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            typename T_OtherStorage>
        HDINLINE T_Type linearize(
            const Vector<T_Type, 2, T_Accessor, T_Navigator, T_Storage>& size,
            const Vector<T_Type, 3, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& pos)
        {
            return pos.z() * size.x() * size.y() + pos.y() * size.x() + pos.x();
        }


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

        /** Creates a \ref pmacc::math::Vector backed by a LLAMA RecordRef as storage. All other properties of the
         * Vector are taken from ProtoVec.
         */
        template<typename ProtoVec, typename RecordRef>
        HDINLINE auto makeVectorWithLlamaStorage(RecordRef rr)
        {
            return Vector<
                typename ProtoVec::type,
                ProtoVec::dim,
                typename ProtoVec::Accessor,
                typename ProtoVec::Navigator,
                detail::VectorLlamaRecordRefStorage<typename ProtoVec::type, ProtoVec::dim, RecordRef>>{{rr}};
        }

        namespace detail
        {
            template<typename T>
            struct ReplaceVectorByArrayImpl
            {
                using type = T;
            };

            template<typename T_Type, int T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
            struct ReplaceVectorByArrayImpl<Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>>
            {
                using type = T_Type[T_dim];
            };
        } // namespace detail

        /** If T is a \ref pmacc::math::Vector, replaced it by an equally sized and typed array. Otherwise, just passes
         * the type through.
         */
        template<typename T>
        using ReplaceVectorByArray = typename detail::ReplaceVectorByArrayImpl<T>::type;
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
