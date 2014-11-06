/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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

#include "result_of_Functor.hpp"
#include <builtin_types.h>
#include <cuda_runtime.h>
#include <boost/static_assert.hpp>
#include <boost/mpl/size.hpp>
#include <types.h>
#include <iostream>
#include <lambda/Expression.hpp>
#include <math/vector/accessor/StandartAccessor.hpp>
#include <math/vector/navigator/StandartNavigator.hpp>

namespace PMacc
{
namespace math
{
namespace detail
{

template<typename T_Type, int T_Dim>
struct Vector_components
{
    static const int dim = T_Dim;
    typedef T_Type type;

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

} //namespace detail

namespace tag
{
struct Vector;
}

template<typename T_Type, int T_dim,
typename T_Accessor = StandartAccessor,
typename T_Navigator = StandartNavigator,
template <typename, int> class T_Storage = detail::Vector_components>
struct Vector : private T_Storage<T_Type, T_dim>, protected T_Accessor, protected T_Navigator
{
    typedef T_Storage<T_Type, T_dim> Storage;
    typedef typename Storage::type type;
    static const int dim = Storage::dim;
    typedef tag::Vector tag;
    typedef T_Accessor Accessor;
    typedef T_Navigator Navigator;
    typedef Vector<type, dim, Accessor, Navigator, T_Storage> This;

    /*Vectors without elements are not allowed*/
    PMACC_CASSERT_MSG(math_Vector__with_DIM_0_is_not_allowed,dim > 0);

    template<class> struct result;

    template<class F, typename T>
    struct result < F(T)>
    {
        typedef typename F::type& type;
    };

    template<class F, typename T>
    struct result < const F(T)>
    {
        typedef const typename F::type& type;
    };

    HDINLINE Vector()
    {
    }

    HDINLINE
    Vector(const type x, const type y)
    {
        PMACC_CASSERT_MSG(math_Vector__constructor_is_only_allowed_for_DIM2,dim == 2);
        (*this)[0] = x;
        (*this)[1] = y;
    }

    HDINLINE
    Vector(const type x, const type y, const type z)
    {
        PMACC_CASSERT_MSG(math_Vector__constructor_is_only_allowed_for_DIM3,dim == 3);
        (*this)[0] = x;
        (*this)[1] = y;
        (*this)[2] = z;
    }

    HDINLINE
    Vector(const T_Type& value)
    {
        for (int i = 0; i < dim; i++)
            (*this)[i] = value;
    }

    HDINLINE Vector(const This& other)
    {
        for (int i = 0; i < dim; i++)
            (*this)[i] = other[i];
    }

    template<
    typename T_OtherType,
    typename T_OtherAccessor,
    typename T_OtherNavigator,
    template <typename, int> class T_OtherStorage>
    HDINLINE explicit Vector(const Vector<
                             T_OtherType,
                             dim,
                             T_OtherAccessor,
                             T_OtherNavigator,
                             T_OtherStorage
                             >&
                             other)
    {
        for (int i = 0; i < dim; i++)
            (*this)[i] = static_cast<type> (other[i]);
    }

    HDINLINE const This& toRT() const
    {
        return *this;
    }

    HDINLINE This& toRT()
    {
        return *this;
    }

    template<
    typename T_OtherAccessor,
    typename T_OtherNavigator,
    template <typename, int> class T_OtherStorage>
    HDINLINE This&
    operator=(const Vector<type, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
    {
        for (int i = 0; i < dim; i++)
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

    HDINLINE type & x()
    {
        return (*this)[0];
    }

    HDINLINE type & y()
    {
        PMACC_CASSERT_MSG(math_Vector__access_to_y_is_not_allowed_for_DIM_lesser_than_2,dim >= 2);
        return (*this)[1];
    }

    HDINLINE type & z()
    {
        PMACC_CASSERT_MSG(math_Vector__access_to_z_is_not_allowed_for_DIM_lesser_than_3,dim >= 3);
        return (*this)[2];
    }

    HDINLINE const type & x() const
    {
        return (*this)[0];
    }

    HDINLINE const type & y() const
    {
        PMACC_CASSERT_MSG(math_Vector__access_to_y_is_not_allowed_for_DIM_lesser_than_2,dim >= 2);
        return (*this)[1];
    }

    HDINLINE const type & z() const
    {
        PMACC_CASSERT_MSG(math_Vector__access_to_z_is_not_allowed_for_DIM_lesser_than_3,dim >= 3);
        return (*this)[2];
    }

    template<int shrinkedDim >
    HDINLINE Vector<type, shrinkedDim, Accessor, Navigator, T_Storage> shrink(const int startIdx = 0) const
    {
        PMACC_CASSERT_MSG(math_Vector__shrinkedDim_DIM_must_be_lesser_or_equal_to_Vector_DIM,shrinkedDim <= dim);
        Vector<type, shrinkedDim, Accessor, Navigator> result;
        for (int i = 0; i < shrinkedDim; i++)
            result[i] = (*this)[(startIdx + i) % dim];
        return result;
    }

    /**
     * Returns product of all components.
     *
     * @return product of components
     */
    HDINLINE type productOfComponents() const
    {
        type result = (*this)[0];
        for (int i = 1; i < dim; i++)
            result *= (*this)[i];
        return result;
    }

    /*! += operator
     * @param other instance with same type and dimension like the left instance
     * @return reference to manipulated left instance
     */
    template<
    typename T_OtherAccessor,
    typename T_OtherNavigator,
    template <typename, int> class T_OtherStorage>
    HDINLINE This&
    operator+=(const Vector<
               type, dim,
               T_OtherAccessor, T_OtherNavigator, T_OtherStorage>&
               other)
    {
        for (int i = 0; i < dim; i++)
            (*this)[i] += other[i];
        return *this;
    }

    /*! -= operator
     * @param other instance with same type and dimension like the left instance
     * @return reference to manipulated left instance
     */
    template<
    typename T_OtherAccessor,
    typename T_OtherNavigator,
    template <typename, int> class T_OtherStorage>
    HDINLINE This&
    operator-=(const Vector<
               type, dim,
               T_OtherAccessor, T_OtherNavigator, T_OtherStorage>&
               other)
    {
        for (int i = 0; i < dim; i++)
            (*this)[i] -= other[i];
        return *this;
    }

    /*! *= operator
     * @param other instance with same type and dimension like the left instance
     * @return reference to manipulated left instance
     */
    template<
    typename T_OtherAccessor,
    typename T_OtherNavigator,
    template <typename, int> class T_OtherStorage>
    HDINLINE This&
    operator*=(const Vector<
               type, dim,
               T_OtherAccessor, T_OtherNavigator, T_OtherStorage>&
               other)
    {

        for (int i = 0; i < dim; i++)
            (*this)[i] *= other[i];
        return *this;
    }

    /*! /= operator
     * @param other instance with same type and dimension like the left instance
     * @return reference to manipulated left instance
     */
    template<
    typename T_OtherAccessor,
    typename T_OtherNavigator,
    template <typename, int> class T_OtherStorage>
    HDINLINE This&
    operator/=(const Vector<
               type, dim,
               T_OtherAccessor, T_OtherNavigator, T_OtherStorage>&
               other)
    {

        for (int i = 0; i < dim; i++)
            (*this)[i] /= other[i];
        return *this;
    }

    HDINLINE This& operator+=(const type & other)
    {

        for (int i = 0; i < dim; i++)
            (*this)[i] += other;
        return *this;
    }

    HDINLINE This& operator-=(const type & other)
    {

        for (int i = 0; i < dim; i++)
            (*this)[i] -= other;
        return *this;
    }

    HDINLINE This& operator*=(const type & other)
    {

        for (int i = 0; i < dim; i++)
            (*this)[i] *= other;
        return *this;
    }

    HDINLINE This& operator/=(const type & other)
    {

        for (int i = 0; i < dim; i++)
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
        for (int i = 0; i < dim; i++)

            if ((*this)[i] != rhs[i]) return false;
        return true;
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
        size_t enclosing_size=enclosings.size();

        if(enclosing_size > 0)
        {
            /* % avoid out of memory access */
            locale_enclosing_begin=enclosings[0%enclosing_size];
            locale_enclosing_end=enclosings[1%enclosing_size];
        }

        std::stringstream stream;
        stream << locale_enclosing_begin << (*this)[0];

        for (int i = 1; i < dim; ++i)
            stream << separator << (*this)[i];
        stream << locale_enclosing_end;
        return stream.str();
    }

    HDINLINE dim3 toDim3() const
    {
        dim3 result;
        unsigned int* ptr = &result.x;
        for (int d = 0; d < dim; ++d)
            ptr[d] = (*this)[d];
        return result;
    }
};

template<typename Type>
struct Vector<Type, 0 >
{
    typedef Type type;
    static const int dim = 0;

    template<typename OtherType >
    HDINLINE operator Vector<OtherType, 0 > () const
    {

        return Vector<OtherType, 0 > ();
    }
};

template<typename Type, int dim, typename Accessor, typename Navigator>
std::ostream& operator<<(std::ostream& s, const Vector<Type, dim, Accessor, Navigator>& vec)
{

    return s << vec.toString();
}

template<typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage,
typename T_OtherAccessor,
typename T_OtherNavigator,
template <typename, int> class T_OtherStorage
>
HDINLINE Vector<T_Type, T_Dim>
operator+(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
          const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(lhs);
    result += rhs;
    return result;
}

template<typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage
>
HDINLINE Vector<T_Type, T_Dim>
operator+(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
          const T_Type& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(lhs);
    result += rhs;
    return result;
}

template<typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage,
typename T_OtherAccessor,
typename T_OtherNavigator,
template <typename, int> class T_OtherStorage
>
HDINLINE Vector<T_Type, T_Dim>
operator-(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
          const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(lhs);
    result -= rhs;
    return result;
}

template<typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage
>
HDINLINE Vector<T_Type, T_Dim>
operator-(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
          const T_Type& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(lhs);
    result -= rhs;
    return result;
}

template<typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage,
typename T_OtherAccessor,
typename T_OtherNavigator,
template <typename, int> class T_OtherStorage
>
HDINLINE Vector<T_Type, T_Dim>
operator*(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
          const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(lhs);
    result *= rhs;
    return result;
}

template<
typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage,
typename T_OtherAccessor,
typename T_OtherNavigator,
template <typename, int> class T_OtherStorage
>
HDINLINE Vector<T_Type, T_Dim>
operator/(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
          const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(lhs);
    result /= rhs;
    return result;
}

template<
typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage
>
HDINLINE Vector<T_Type, T_Dim>
operator*(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs, const T_Type& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(lhs);
    result *= rhs;
    return result;
}

template<
typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage
>
HDINLINE Vector<T_Type, T_Dim>
operator*(const T_Type& lhs, const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(rhs);
    result *= lhs;
    return result;
}

template<
typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage
>
HDINLINE Vector<T_Type, T_Dim>
operator/(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs, const T_Type& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(lhs);
    result /= rhs;
    return result;
}

template<
typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage
>
HDINLINE Vector<T_Type, T_Dim>
operator-(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& vec)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<T_Type, T_Dim> result(vec);

    for (int i = 0; i < T_Dim; i++)
        result[i] = -result[i];
    return result;
}

template<
typename T_Type, int T_Dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage,
typename T_OtherAccessor,
typename T_OtherNavigator,
template <typename, int> class T_OtherStorage
>
HDINLINE Vector<bool, T_Dim>
operator>=(const Vector<T_Type, T_Dim, T_Accessor, T_Navigator, T_Storage>& lhs,
           const Vector<T_Type, T_Dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& rhs)
{
    /* to avoid allocation side effects the result is always a vector
     * with default policies*/
    Vector<bool, T_Dim > result;
    for (int i = 0; i < T_Dim; ++i)
        result[i] = (lhs[i] >= rhs[i]);
    return result;
}

template<
typename T_Type,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage,
typename T_OtherAccessor,
typename T_OtherNavigator,
template <typename, int> class T_OtherStorage
>
HDINLINE T_Type
linearize(const Vector<T_Type, 1, T_Accessor, T_Navigator, T_Storage >& size,
          const Vector<T_Type, 2, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& pos)
{
    return pos.y() * size.x() + pos.x();
}

template<
typename T_Type,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage,
typename T_OtherAccessor,
typename T_OtherNavigator,
template <typename, int> class T_OtherStorage
>
HDINLINE T_Type
linearize(const Vector<T_Type, 2, T_Accessor, T_Navigator, T_Storage >& size,
          const Vector<T_Type, 3, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& pos)
{
    return pos.z() * size.x() * size.y() + pos.y() * size.x() + pos.x();
}

template<typename Vector>
HDINLINE Vector floor(const Vector& vector)
{
    Vector result;

    for (int i = 0; i < Vector::dim; i++)
        result[i] = floorf(vector[i]);
    return result;
}

template<typename Lhs, typename Rhs>
HDINLINE Lhs operator%(const Lhs& lhs, const Rhs& rhs)
{
    Lhs result;

    for (int i = 0; i < Lhs::dim; i++)
        result[i] = lhs[i] % rhs[i];
    return result;
}

template<typename Type, int dim>
HDINLINE Type abs2(const Vector<Type, dim>& vec)
{
    Type result = vec.x() * vec.x();

    for (int i = 1; i < dim; i++)
        result += vec[i] * vec[i];
    return result;
}

template<typename Type, int dim>
HDINLINE Type abs(const Vector<Type, dim>& vec)
{

    return sqrtf(abs2(vec));
}

template<typename Type, int dim>
HDINLINE
Type dot(const Vector<Type, dim>& a, const Vector<Type, dim>& b)
{
    Type result = a.x() * b.x();

    for (int i = 1; i < dim; i++)
        result += a[i] * b[i];
    return result;
}

struct Abs2
{

    template<typename Type, int dim >
    HDINLINE Type operator()(const Vector<Type, dim>& vec)
    {

        return abs2(vec);
    }
};

struct Abs
{

    template<typename Type, int dim >
    HDINLINE Type operator()(const Vector<Type, dim>& vec)
    {
        return abs(vec);
    }
};

} //namespace math

namespace result_of
{

template<typename TVector>
struct Functor<math::Abs2, TVector>
{
    typedef typename TVector::type type;
};

template<typename TVector>
struct Functor<math::Abs, TVector>
{
    typedef typename TVector::type type;
};

} //namespace result_of
} //namespace PMacc
