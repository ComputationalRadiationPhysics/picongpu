/**
 * Copyright 2013-2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
 
#pragma once

#include "dimensions/DataSpace.hpp"
#include "math/vector/compile-time/Int.hpp"

namespace PMacc
{

    /**
     * 3-dimensional vector type whose values are set using template arguments.
     *
     * @param X value for 1. dimension
     * @param Y value for 2. dimension. 0 by default
     * @param Z value for 3. dimension. 0 by default
     */
    template<uint32_t X, uint32_t Y = 0, uint32_t Z = 0 >
            class TVec
    {
    public:

        enum
        {
            x = X,
            y = Y,
            z = Z,
            elements = X * Y * Z,
            dim = DIM3
        };

        typedef TVec<X> TVec1D;
        typedef TVec<X, Y> TVec2D;
        typedef TVec<X, Y, Z> TVec3D;

        HDINLINE static DataSpace<DIM3> getDataSpace()
        {
            return DataSpace<DIM3 > (X, Y, Z);
        }

        HDINLINE operator DataSpace<DIM3>() const
        {
            return DataSpace<DIM3 > (x, y, z);
        }

        HDINLINE operator DataSpace<DIM2>() const
        {
            return DataSpace<DIM2 > (x, y);
        }

        HDINLINE operator DataSpace<DIM1>() const
        {
            return DataSpace<DIM1 > (x);
        }

        template<class TVector>
        struct add
        {
            typedef TVec < X + TVector::x, Y + TVector::y, Z + TVector::z> Result;
        };
        
        template<class TVector>
        struct max
        {
            typedef TVec <
                ( X > TVector::x ) ? X : TVector::x,
                ( Y > TVector::y ) ? Y : TVector::y,
                ( Z > TVector::z ) ? Z : TVector::z> Result;
        };
    };

    template<uint32_t X>
    class TVec<X, 0, 0 >
    {
    public:

        enum
        {
            x = X,
            y = 0,
            z = 0,
            elements = X,
            dim = DIM1
        };

        typedef TVec<X> TVec1D;
        typedef TVec<X, 0 > TVec2D;
        typedef TVec<X, 0, 0 > TVec3D;

        HDINLINE static DataSpace<DIM1> getDataSpace()
        {
            return DataSpace<DIM1 > (X);
        }

        HDINLINE operator DataSpace<DIM3>() const
        {
            return DataSpace<DIM3 > (x, y, z);
        }

        HDINLINE operator DataSpace<DIM2>() const
        {
            return DataSpace<DIM2 > (x, y);
        }

        HDINLINE operator DataSpace<DIM1>() const
        {
            return DataSpace<DIM1 > (x);
        }

        template<class TVector>
        struct add
        {
            typedef TVec < X + TVector::x, TVector::y, TVector::z> Result;
        };
        
        template<class TVector>
        struct max
        {
            typedef TVec <
                ( X > TVector::x ) ? X : TVector::x,
                TVector::y,
                TVector::z> Result;
        };

    };

    template<uint32_t X, uint32_t Y>
    class TVec<X, Y, 0 >
    {
    public:

        enum
        {
            x = X,
            y = Y,
            z = 0,
            elements = X * Y,
            dim = DIM2
        };

        typedef TVec<X> TVec1D;
        typedef TVec<X, Y> TVec2D;
        typedef TVec<X, Y, 0 > TVec3D;

        HDINLINE static DataSpace<DIM2> getDataSpace()
        {
            return DataSpace<DIM2 > (X, Y);
        }

        HDINLINE operator DataSpace<DIM3>() const
        {
            return DataSpace<DIM3 > (x, y, z);
        }

        HDINLINE operator DataSpace<DIM2>() const
        {
            return DataSpace<DIM2 > (x, y);
        }

        HDINLINE operator DataSpace<DIM1>() const
        {
            return DataSpace<DIM1 > (x);
        }

        template<class TVector>
        struct add
        {
            typedef TVec < X + TVector::x, Y + TVector::y, TVector::z> Result;
        };
        
        template<class TVector>
        struct max
        {
            typedef TVec <
                ( X > TVector::x ) ? X : TVector::x,
                ( Y > TVector::y ) ? Y : TVector::y,
                TVector::z> Result;
        };
    };

    template< class Any, uint32_t Id >
    struct Get;

    template<template<uint32_t, uint32_t, uint32_t> class Any, uint32_t X, uint32_t Y, uint32_t Z >
    struct Get < Any<X, Y, Z>, 0 >
    {

        enum
        {
            value = X
        };
    };

    template<template<uint32_t, uint32_t, uint32_t> class Any, uint32_t X, uint32_t Y, uint32_t Z >
    struct Get < Any<X, Y, Z>, 1 >
    {

        enum
        {
            value = Y
        };
    };

    template<template<uint32_t, uint32_t, uint32_t> class Any, uint32_t X, uint32_t Y, uint32_t Z >
    struct Get < Any<X, Y, Z>, 2 >
    {

        enum
        {
            value = Z
        };
    };
    
    template<typename TVec_>
    struct toVector;
    
    template<uint32_t X, uint32_t Y, uint32_t Z >
    struct toVector<TVec<X,Y,Z> >
    {
        typedef typename math::CT::Int<X,Y,Z> type;
    };
    
    namespace detail
    {
        template<uint32_t T_Dim,typename T_Type >
        struct toTVec;
    
        template<int X, int Y, int Z >
        struct toTVec<3u,math::CT::Int<X,Y,Z> >
        {
            typedef TVec<(uint32_t)X, (uint32_t)Y, (uint32_t)Z> type;
        };

        template<typename X, typename Y, typename Z >
        struct toTVec<3u,math::CT::Vector<X,Y,Z> >
        {
            typedef TVec<X::value,Y::value,Z::value> type;
        };
        
        template<int X, int Y, int Z >
        struct toTVec<2u,math::CT::Int<X,Y,Z> >
        {
            typedef TVec<(uint32_t)X, (uint32_t)Y> type;
        };

        template<typename X, typename Y, typename Z >
        struct toTVec<2u,math::CT::Vector<X,Y,Z> >
        {
            typedef TVec<X::value,Y::value> type;
        };
    }
    
    template<typename Vector>
    struct toTVec
    {
        typedef typename detail::toTVec<Vector::dim,Vector>::type type;
    };
    

}
