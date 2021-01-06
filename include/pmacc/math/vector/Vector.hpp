/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz,
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

#include "pmacc/math/vector/accessor/StandardAccessor.hpp"
#include "pmacc/math/vector/navigator/StandardNavigator.hpp"
#include "pmacc/result_of_Functor.hpp"
#include "pmacc/static_assert.hpp"
#include "pmacc/types.hpp"
#include "pmacc/algorithms/math.hpp"

#include <boost/mpl/size.hpp>
#include <boost/call_traits.hpp>
#include <iostream>
#include <type_traits>

namespace pmacc
{
    namespace math
    {
        namespace detail
        {
            template<typename T_Type, int T_Dim>
            struct Vector_components
            {
                static constexpr bool isConst = false;
                static constexpr int dim = T_Dim;
                using type = T_Type;

                HDINLINE
                constexpr Vector_components()
                {
                }

                constexpr Vector_components& operator=(const Vector_components&) = default;

                /*align full vector*/
                PMACC_ALIGN(v[dim], type);

                HDINLINE
                type& operator[](const int idx)
                {
                    return v[idx];
                }

                HDINLINE
                const type& operator[](const int idx) const
                {
                    return v[idx];
                }
            };


            /** functor to copy a object element-wise
             *
             * @tparam isDestConst define if destination is const (not copyable) object
             */
            template<bool isDestConst>
            struct CopyElementWise
            {
                /** copy object element-wise
                 *
                 * @tparam T_Dest destination object type
                 * @tparam T_Src source object type
                 */
                template<typename T_Dest, typename T_Src>
                HDINLINE void operator()(T_Dest& dest, const T_Src& src) const
                {
                    PMACC_CASSERT_MSG(
                        CopyElementWise_destination_and_source_had_different_dimension,
                        T_Dest::dim == T_Src::dim);
                    for(int d = 0; d < T_Dest::dim; d++)
                        dest[d] = src[d];
                }
            };

            /** specialization for constant destination
             *
             * the constant storage is already available and set in the destination
             */
            template<>
            struct CopyElementWise<true>
            {
                template<typename T_Dest, typename T_Src>
                HDINLINE void operator()(T_Dest& dest, const T_Src& src) const
                {
                }
            };

        } // namespace detail

        namespace tag
        {
            struct Vector;
        }

        template<
            typename T_Type,
            int T_dim,
            typename T_Accessor = StandardAccessor,
            typename T_Navigator = StandardNavigator,
            template<typename, int> class T_Storage = detail::Vector_components>
        struct Vector
            : private T_Storage<T_Type, T_dim>
            , protected T_Accessor
            , protected T_Navigator
        {
            using Storage = T_Storage<T_Type, T_dim>;
            using type = typename Storage::type;
            static constexpr int dim = Storage::dim;
            using tag = tag::Vector;
            using Accessor = T_Accessor;
            using Navigator = T_Navigator;
            using This = Vector<type, dim, Accessor, Navigator, T_Storage>;
            using ParamType = typename boost::call_traits<type>::param_type;

            /*Vectors without elements are not allowed*/
            PMACC_CASSERT_MSG(math_Vector__with_DIM_0_is_not_allowed, dim > 0);

            template<class>
            struct result;

            template<class F, typename T>
            struct result<F(T)>
            {
                using type = typename F::type&;
            };

            template<class F, typename T>
            struct result<const F(T)>
            {
                using type = const typename F::type&;
            };

            HDINLINE
            constexpr Vector()
            {
            }

            HDINLINE
            constexpr Vector(const type x)
            {
                PMACC_CASSERT_MSG(math_Vector__constructor_is_only_allowed_for_DIM1, dim == 1);
                (*this)[0] = x;
            }

            HDINLINE
            constexpr Vector(const type x, const type y)
            {
                PMACC_CASSERT_MSG(math_Vector__constructor_is_only_allowed_for_DIM2, dim == 2);
                (*this)[0] = x;
                (*this)[1] = y;
            }

            HDINLINE
            constexpr Vector(const type x, const type y, const type z)
            {
                PMACC_CASSERT_MSG(math_Vector__constructor_is_only_allowed_for_DIM3, dim == 3);
                (*this)[0] = x;
                (*this)[1] = y;
                (*this)[2] = z;
            }

            HDINLINE
            constexpr Vector(const This& other)
            {
                detail::CopyElementWise<Storage::isConst>()(*this, other);
            }

            template<
                typename T_OtherType,
                typename T_OtherAccessor,
                typename T_OtherNavigator,
                template<typename, int>
                class T_OtherStorage>
            HDINLINE explicit Vector(
                const Vector<T_OtherType, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] = static_cast<type>(other[i]);
            }

            /** Allow static_cast / explicit cast to member type for 1D vector */
            template<int T_deferDim = T_dim, typename = typename std::enable_if<T_deferDim == 1>::type>
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
            static This create(ParamType value)
            {
                This result;
                for(int i = 0; i < dim; i++)
                    result[i] = value;

                return result;
            }

            HDINLINE const This& toRT() const
            {
                return *this;
            }

            HDINLINE This& toRT()
            {
                return *this;
            }

            HDINLINE This revert()
            {
                This invertedVector;
                for(int i = 0; i < dim; i++)
                    invertedVector[dim - 1 - i] = (*this)[i];

                return invertedVector;
            }

            constexpr HDINLINE Vector& operator=(const Vector&) = default;

            template<typename T_OtherAccessor, typename T_OtherNavigator, template<typename, int> class T_OtherStorage>
            HDINLINE This& operator=(const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] = rhs[i];
                return *this;
            }

            HDINLINE
            type& operator[](const int idx)
            {
                return Accessor::operator()(Storage::operator[](Navigator::operator()(idx)));
            }

            HDINLINE
            const type& operator[](const int idx) const
            {
                return Accessor::operator()(Storage::operator[](Navigator::operator()(idx)));
            }

            HDINLINE type& x()
            {
                return (*this)[0];
            }

            HDINLINE type& y()
            {
                PMACC_CASSERT_MSG(math_Vector__access_to_y_is_not_allowed_for_DIM_lesser_than_2, dim >= 2);
                return (*this)[1];
            }

            HDINLINE type& z()
            {
                PMACC_CASSERT_MSG(math_Vector__access_to_z_is_not_allowed_for_DIM_lesser_than_3, dim >= 3);
                return (*this)[2];
            }

            HDINLINE const type& x() const
            {
                return (*this)[0];
            }

            HDINLINE const type& y() const
            {
                PMACC_CASSERT_MSG(math_Vector__access_to_y_is_not_allowed_for_DIM_lesser_than_2, dim >= 2);
                return (*this)[1];
            }

            HDINLINE const type& z() const
            {
                PMACC_CASSERT_MSG(math_Vector__access_to_z_is_not_allowed_for_DIM_lesser_than_3, dim >= 3);
                return (*this)[2];
            }

            template<int shrinkedDim>
            HDINLINE Vector<type, shrinkedDim, Accessor, Navigator> shrink(const int startIdx = 0) const
            {
                PMACC_CASSERT_MSG(
                    math_Vector__shrinkedDim_DIM_must_be_lesser_or_equal_to_Vector_DIM,
                    shrinkedDim <= dim);
                Vector<type, shrinkedDim, Accessor, Navigator> result;
                for(int i = 0; i < shrinkedDim; i++)
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
            template<int dimToRemove>
            HDINLINE Vector<type, dim - 1, Accessor, Navigator> remove() const
            {
                PMACC_CASSERT_MSG(__math_Vector__dim_must_be_greater_than_1__, dim > 1);
                PMACC_CASSERT_MSG(__math_Vector__dimToRemove_must_be_lesser_than_dim__, dimToRemove < dim);
                Vector<type, dim - 1, Accessor, Navigator> result;
                for(int i = 0; i < dim - 1; ++i)
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
                for(int i = 1; i < dim; i++)
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
                for(int i = 1; i < dim; i++)
                    result += (*this)[i];
                return result;
            }

            /*! += operator
             * @param other instance with same type and dimension like the left instance
             * @return reference to manipulated left instance
             */
            template<typename T_OtherAccessor, typename T_OtherNavigator, template<typename, int> class T_OtherStorage>
            HDINLINE This& operator+=(
                const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] += other[i];
                return *this;
            }

            /*! -= operator
             * @param other instance with same type and dimension like the left instance
             * @return reference to manipulated left instance
             */
            template<typename T_OtherAccessor, typename T_OtherNavigator, template<typename, int> class T_OtherStorage>
            HDINLINE This& operator-=(
                const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] -= other[i];
                return *this;
            }

            /*! *= operator
             * @param other instance with same type and dimension like the left instance
             * @return reference to manipulated left instance
             */
            template<typename T_OtherAccessor, typename T_OtherNavigator, template<typename, int> class T_OtherStorage>
            HDINLINE This& operator*=(
                const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] *= other[i];
                return *this;
            }

            /*! /= operator
             * @param other instance with same type and dimension like the left instance
             * @return reference to manipulated left instance
             */
            template<typename T_OtherAccessor, typename T_OtherNavigator, template<typename, int> class T_OtherStorage>
            HDINLINE This& operator/=(
                const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& other)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] /= other[i];
                return *this;
            }

            HDINLINE This& operator+=(ParamType other)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] += other;
                return *this;
            }

            HDINLINE This& operator-=(ParamType other)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] -= other;
                return *this;
            }

            HDINLINE This& operator*=(ParamType other)
            {
                for(int i = 0; i < dim; i++)
                    (*this)[i] *= other;
                return *this;
            }

            HDINLINE This& operator/=(ParamType other)
            {
                for(int i = 0; i < dim; i++)
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
            HDINLINE bool operator==(const This& rhs) const
            {
                bool result = true;
                for(int i = 0; i < dim; i++)
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
            HDINLINE bool operator!=(const This& rhs) const
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

                for(int i = 1; i < dim; ++i)
                    stream << separator << (*this)[i];
                stream << locale_enclosing_end;
                return stream.str();
            }

            HDINLINE cupla::dim3 toDim3() const
            {
                cupla::dim3 result;
                unsigned int* ptr = &result.x;
                for(int d = 0; d < dim; ++d)
                    ptr[d] = (*this)[d];
                return result;
            }
        };

        template<typename Type>
        struct Vector<Type, 0>
        {
            using type = Type;
            static constexpr int dim = 0;

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

        template<typename Type, int dim, typename Accessor, typename Navigator>
        std::ostream& operator<<(std::ostream& s, const Vector<Type, dim, Accessor, Navigator>& vec)
        {
            return s << vec.toString();
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            template<typename, int>
            class T_OtherStorage>
        HDINLINE Vector<T_Type, T_Dim> operator+(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(lhs);
            result += rhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage>
        HDINLINE Vector<T_Type, T_Dim> operator+(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            typename Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>::ParamType rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(lhs);
            result += rhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            template<typename, int>
            class T_OtherStorage>
        HDINLINE Vector<T_Type, T_Dim> operator-(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(lhs);
            result -= rhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage>
        HDINLINE Vector<T_Type, T_Dim> operator-(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            typename Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>::ParamType rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(lhs);
            result -= rhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            template<typename, int>
            class T_OtherStorage>
        HDINLINE Vector<T_Type, T_Dim> operator*(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(lhs);
            result *= rhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            template<typename, int>
            class T_OtherStorage>
        HDINLINE Vector<T_Type, T_Dim> operator/(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(lhs);
            result /= rhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage>
        HDINLINE Vector<T_Type, T_Dim> operator*(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            typename Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>::ParamType rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(lhs);
            result *= rhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage>
        HDINLINE Vector<T_Type, T_Dim> operator*(
            typename boost::call_traits<T_Type>::param_type lhs,
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(rhs);
            result *= lhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage>
        HDINLINE Vector<T_Type, T_Dim> operator/(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            typename Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>::ParamType rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(lhs);
            result /= rhs;
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage>
        HDINLINE Vector<T_Type, T_Dim> operator-(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& vec)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<T_Type, T_Dim> result(vec);

            for(int i = 0; i < T_Dim; i++)
                result[i] = -result[i];
            return result;
        }

        template<
            typename T_Type,
            int T_Dim,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            template<typename, int>
            class T_OtherStorage>
        HDINLINE Vector<bool, T_Dim> operator>=(
            const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
            const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
        {
            /* to avoid allocation side effects the result is always a vector
             * with default policies*/
            Vector<bool, T_Dim> result;
            for(int i = 0; i < T_Dim; ++i)
                result[i] = (lhs[i] >= rhs[i]);
            return result;
        }

        template<
            typename T_Type,
            typename T_Accessor,
            typename T_Navigator,
            template<typename, int>
            class T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            template<typename, int>
            class T_OtherStorage>
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
            template<typename, int>
            class T_Storage,
            typename T_OtherAccessor,
            typename T_OtherNavigator,
            template<typename, int>
            class T_OtherStorage>
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

            for(int i = 0; i < Lhs::dim; i++)
                result[i] = lhs[i] % rhs[i];
            return result;
        }

        struct Abs
        {
            template<typename Type, int dim>
            HDINLINE Type operator()(const Vector<Type, dim>& vec)
            {
                return cupla::math::abs(vec);
            }
        };

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

    namespace result_of
    {
        template<typename TVector>
        struct Functor<math::Abs, TVector>
        {
            using type = typename TVector::type;
        };

    } // namespace result_of
} // namespace pmacc
