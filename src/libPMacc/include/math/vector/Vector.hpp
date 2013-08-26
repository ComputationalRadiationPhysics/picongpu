/**
 * Copyright 2013 Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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

#ifndef VECTOR_HPP
#define VECTOR_HPP

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

template<typename Type, int dim>
struct Vector_components;

template<typename Type>
struct Vector_components<Type, 0 >
{
};

template<typename Type>
struct Vector_components<Type, 1 >
{
    PMACC_ALIGN(_x, Type);

    HDINLINE Vector_components()
    {
    }

    HDINLINE
    Vector_components(const Type & x) : _x(x)
    {
    }
};

template<typename Type>
struct Vector_components<Type, 2 >
{
    PMACC_ALIGN(_x, Type);
    PMACC_ALIGN(_y, Type);

    HDINLINE Vector_components()
    {
    }

    HDINLINE
    Vector_components(const Type& x, const Type & y) : _x(x), _y(y)
    {
    }
};

template<typename Type>
struct Vector_components<Type, 3 >
{
    //Type x, y, z;
    PMACC_ALIGN(_x, Type);
    PMACC_ALIGN(_y, Type);
    PMACC_ALIGN(_z, Type);

    HDINLINE Vector_components()
    {
    }

    HDINLINE
    Vector_components(const Type& x, const Type& y, const Type & z) : _x(x), _y(y), _z(z)
    {
    }
};

}

namespace tag
{
struct Vector;
}

template<typename Type, int _dim,
typename _Accessor = StandartAccessor,
typename _Navigator = StandartNavigator>
//__optimal_align__((dim==0) ? 1 : dim * sizeof(Type))
struct Vector : public detail::Vector_components<Type, _dim>, _Accessor, _Navigator
{
    typedef Type type;
    static const int dim = _dim;
    typedef tag::Vector tag;
    typedef _Accessor Accessor;
    typedef _Navigator Navigator;
    typedef Vector<Type, dim, Accessor, Navigator> This;

    template<class> struct result;

    template<class F,typename T>
    struct result < F(T)>
    {
        typedef typename F::type& type;
    };

    HDINLINE Vector()
    {
    }

    HDINLINE
    Vector(const Type x, const Type y)
    {
        (*this)[0] = x;
        (*this)[1] = y;
    }

    HDINLINE
    Vector(const Type x, const Type y, const Type z)
    {
        (*this)[0] = x;
        (*this)[1] = y;
        (*this)[2] = z;
    }

    HDINLINE
    Vector(const Type & value)
    {
        //BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] = value;
    }

    template<typename OtherType, typename OtherAccessor, typename OtherNavigator >
        HDINLINE Vector(Vector<OtherType, dim, OtherAccessor, OtherNavigator>& other)
    {
        //BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] = (Type) other[i];
    }

    template<typename OtherType, typename OtherAccessor, typename OtherNavigator >
        HDINLINE Vector(const Vector<OtherType, dim, OtherAccessor, OtherNavigator>& other)
    {
        //BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] = (Type) other[i];
    }

        /*
        template<typename OtherType >
        HDINLINE operator Vector<OtherType, dim > () const
        {
            Vector<OtherType, dim> result;
            for (int i = 0; i < dim; i++)
                result[i] = (OtherType) ((*this)[i]);

            return result;
        }
    
        template<typename OtherType>
        HDINLINE operator OtherType() const
        {
            BOOST_STATIC_ASSERT((boost::is_same<typename OtherType::tag, tag::Vector>::value));
            return (vector<typename OtherType::type, OtherType::dim>)*this;
        }*/

    HDINLINE const Vector<type, dim>& vec() const
    {
        return *this;
    }

    HDINLINE Vector<type, dim>& vec()
    {
        return *this;
    }

    template<typename OtherAccessor, typename OtherNavigator >
        HDINLINE This&
        operator=(Vector<Type, dim, OtherAccessor, OtherNavigator>& rhs)
    {
        for (int i = 0; i < dim; i++)
            (*this)[i] = rhs[i];
        return *this;
    }

    template<typename OtherAccessor, typename OtherNavigator >
        HDINLINE This&
        operator=(const Vector<Type, dim, OtherAccessor, OtherNavigator>& rhs)
    {
        for (int i = 0; i < dim; i++)
            (*this)[i] = rhs[i];
        return *this;
    }

    HDINLINE
    Type& operator[](const int idx)
    {
        BOOST_STATIC_ASSERT(dim > 0);
        return Accessor::operator()((&this->_x)[Navigator::operator()(idx)]);
    }

    HDINLINE
        const Type& operator[](const int idx) const
        {
            BOOST_STATIC_ASSERT(dim > 0);
            return Accessor::operator()((&this->_x)[Navigator::operator()(idx)]);
        }

    HDINLINE Type & x()
    {
        return (*this)[0];
    }

    HDINLINE Type & y()
    {
        return (*this)[1];
    }

    HDINLINE Type & z()
    {
        return (*this)[2];
    }

    HDINLINE const Type & x() const
    {
        return (*this)[0];
    }

    HDINLINE const Type & y() const
    {
        return (*this)[1];
    }

    HDINLINE const Type & z() const
    {
        return (*this)[2];
    }

    template<int shrinkedDim >
        HDINLINE
        Vector<Type, shrinkedDim, Accessor, Navigator> shrink(const int startIdx = 0) const
    {
        BOOST_STATIC_ASSERT(shrinkedDim <= dim);
        Vector<Type, shrinkedDim, Accessor, Navigator> result;
        for (int i = 0; i < shrinkedDim; i++)
            result[i] = (*this)[(startIdx + i) % dim];
        return result;
    }

    HDINLINE
    Type volume() const
    {
        BOOST_STATIC_ASSERT(dim > 0);
        Type result = (*this)[0];
        for (int i = 1; i < dim; i++)
            result *= (*this)[i];
        return result;
    }

    HDINLINE
    Vector<Type, dim>& operator+=(const Vector<Type, dim>& other)
    {
        BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] += other[i];
        return *this;
    }

    HDINLINE
    Vector<Type, dim>& operator-=(const Vector<Type, dim>& other)
    {
        BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] -= other[i];
        return *this;
    }

    HDINLINE
    Vector<Type, dim>& operator*=(const Vector<Type, dim>& other)
    {
        BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] *= other[i];
        return *this;
    }

    HDINLINE
    Vector<Type, dim>& operator/=(const Vector<Type, dim>& other)
    {
        BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] /= other[i];
        return *this;
    }

    HDINLINE
    Vector<Type, dim>& operator*=(const Type & other)
    {
        BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] *= other;
        return *this;
    }

    HDINLINE
    Vector<Type, dim>& operator/=(const Type & other)
    {
        BOOST_STATIC_ASSERT(dim > 0);
        for (int i = 0; i < dim; i++)
            (*this)[i] /= other;
        return *this;
    }

    HDINLINE bool operator==(const Vector<Type, dim, Accessor, Navigator>& rhs) const
    {
        for (int i = 0; i < dim; i++)
            if ((*this)[i] != rhs[i]) return false;
        return true;
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
    for (int i = 0; i < dim - 1; i++)
        s << vec[i] << ", ";
    return s << vec[dim - 1];
}

template<typename Type, int dim, typename Accessor, typename Navigator>
HDINLINE
Vector<Type, dim> operator+(const Vector<Type, dim, Accessor, Navigator>& lhs, const Vector<Type, dim, Accessor, Navigator>& rhs)
{
    Vector<Type, dim> result(lhs);
    result += rhs;
    return result;
}

template<typename Type, int dim, typename Accessor, typename Navigator>
HDINLINE
Vector<Type, dim> operator-(const Vector<Type, dim, Accessor, Navigator>& lhs, const Vector<Type, dim, Accessor, Navigator>& rhs)
{
    Vector<Type, dim> result(lhs);
    result -= rhs;
    return result;
}

template<typename Type, int dim, typename Accessor, typename Navigator>
HDINLINE
Vector<Type, dim> operator*(const Vector<Type, dim, Accessor, Navigator>& lhs, const Vector<Type, dim, Accessor, Navigator>& rhs)
{
    Vector<Type, dim> result(lhs);
    result *= rhs;
    return result;
}

template<typename Type, int dim, typename Accessor, typename Navigator>
HDINLINE
Vector<Type, dim> operator/(const Vector<Type, dim, Accessor, Navigator>& lhs, const Vector<Type, dim, Accessor, Navigator>& rhs)
{
    Vector<Type, dim> result(lhs);
    result /= rhs;
    return result;
}

template<typename Type, int dim, typename Accessor, typename Navigator>
HDINLINE
Vector<Type, dim> operator*(const Vector<Type, dim, Accessor, Navigator>& lhs, const Type& rhs)
{
    Vector<Type, dim> result(lhs);
    result *= rhs;
    return result;
}

template<typename Type, int dim, typename Accessor, typename Navigator>
HDINLINE
Vector<Type, dim> operator*(const Type& lhs, const Vector<Type, dim, Accessor, Navigator>& rhs)
{
    Vector<Type, dim> result(rhs);
    result *= lhs;
    return result;
}

template<typename Type, int dim, typename Accessor, typename Navigator>
HDINLINE
Vector<Type, dim> operator/(const Vector<Type, dim, Accessor, Navigator>& lhs, const Type& rhs)
{
    Vector<Type, dim> result(lhs);
    result /= rhs;
    return result;
}

template<typename Type, int dim, typename Accessor, typename Navigator>
HDINLINE
Vector<Type, dim> operator-(const Vector<Type, dim, Accessor, Navigator>& vec)
{
    Vector<Type, dim> result(vec);
    for (int i = 0; i < dim; i++)
        result[i] = -result[i];
    return result;
}

template<typename Type>
HDINLINE Type linearize(const Vector<Type, 1 > & size, const Vector<Type, 2 > & pos)
{
    return pos.y() * size.x() + pos.x();
}

template<typename Type>
HDINLINE Type linearize(const Vector<Type, 2 > & size, const Vector<Type, 3 > & pos)
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
    BOOST_STATIC_ASSERT(dim > 0);
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

//lambda::Expression<lambda::exprTypes::terminal, mpl::vector<Abs2> > _abs2;
//lambda::Expression<lambda::exprTypes::terminal, mpl::vector<Abs> > _abs;

} // math

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

} // result_of
} // PMacc

#endif // VECTOR_HPP
