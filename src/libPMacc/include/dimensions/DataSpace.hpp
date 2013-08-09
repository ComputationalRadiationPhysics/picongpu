/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera, Wolfgang Hoenig
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
 
/*
 * File:   DataSpace.hpp
 * Author: widera
 *
 * Created on 22. März 2010, 16:04
 */

#ifndef _DATASPACE_HPP
#define	_DATASPACE_HPP

#include <cassert>
#include <stdexcept>
#include <stdio.h>
#include <math.h>
#include <iostream>

#include "types.h"
#include "math/vector/Int.hpp"
#include "math/vector/Size_t.hpp"


namespace PMacc
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
class DataSpace : public math::Int<DIM>
    {
    public:
        
        static const int Dim=DIM;

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

        HDINLINE DataSpace(const DataSpace<DIM>& value)
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                (*this)[i] = value[i];
            }
        }

        /**
         * Constructor for DIM1-dimensional DataSpace.
         *
         * @param x size of first dimension
         */
        HDINLINE DataSpace(int x);

        /**
         * Constructor for DIM2-dimensional DataSpace.
         *
         * @param x size of first dimension
         * @param y size of second dimension
         */
        HDINLINE DataSpace(int x, int y);

        /**
         * Constructor for DIM3-dimensional DataSpace.
         *
         * @param x size of first dimension
         * @param y size of second dimension
         * @param z size of third dimension
         */
        HDINLINE DataSpace(int x, int y, int z);
        
        HDINLINE int& x() {return (*this)[0];}
        HDINLINE int& y() {return (*this)[1];}
        HDINLINE int& z() {return (*this)[2];}
    
        HDINLINE int x() const {return (*this)[0];}
        HDINLINE int y() const {return (*this)[1];}
        HDINLINE int z() const {return (*this)[2];}
        
        HDINLINE DataSpace(const math::Int<DIM>& vec)
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                (*this)[i] = vec[i];
            }
        }
        HDINLINE DataSpace(const math::Size_t<DIM>& vec)
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                (*this)[i] = vec[i];
            }
        }

        /**
         * Destructor.
         */
        HDINLINE ~DataSpace()
        {
        }

        /**
         * == comparison operator.
         *
         * Compares sizes of two DataSpaces.
         *
         * @param other DataSpace to compare to
         * @return if both DataSpaces are equal
         */
        HDINLINE bool operator==(DataSpace<DIM> const& other) const
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                if ((*this)[i] != other[i]) return false;
            }
            return true;
        }

        /**
         * != comparison operator.
         *
         * Compares sizes of two DataSpaces.
         *
         * @param other DataSpace to compare to
         * @return if DataSpaces are different
         */
        HDINLINE bool operator!=(DataSpace<DIM> const& other) const
        {
            return !((*this) == other);
        }

        /**
         * Adds dimensions of two DataSpaces dimension-wise.
         *
         * @param other DataSpace to add to this one
         * @return the new DataSpace
         */
        HDINLINE DataSpace<DIM> operator+(DataSpace<DIM> const& other) const
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = (*this)[i] + other[i];
            }
            return tmp;
        }

        /**
         * Adds dimensions of two DataSpaces dimension-wise.
         *
         * @param other DataSpace to add to "this"
         * @return the manipulated DataSpace
         */
        HDINLINE DataSpace<DIM> operator+=(DataSpace<DIM> const& other)
        {
            for (uint32_t i = 0; i < DIM; ++i)
            {
                (*this)[i] += other[i];
            }
            return (*this);
        }

        /**
         * Adds each dimension with value.
         *
         * @param value value for addition
         * @return the new DataSpace
         */
        HDINLINE DataSpace<DIM> operator+(int value) const
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = (*this)[i] + value;
            }
            return tmp;
        }

        /**
         * Subtracts dimensions of two DataSpaces dimension-wise.
         *
         * @param other DataSpace to subtract from this one
         * @return the new DataSpace
         */
        HDINLINE DataSpace<DIM> operator-(DataSpace<DIM> const& other) const
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = (*this)[i] - other[i];
            }
            return tmp;
        }

        /**
         * Subtracts each dimension with value.
         *
         * @param value value for subtraction
         * @return the new DataSpace
         */
        HDINLINE DataSpace<DIM> operator-(int value) const
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = (*this)[i] - value;
            }
            return tmp;
        }

        /**
         * Divides dimensions of two DataSpaces dimension-wise.
         *
         * @param other DataSpace to divide by
         * @return the new DataSpace
         */
        HDINLINE DataSpace<DIM> operator/(DataSpace<DIM> const& other) const
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = (*this)[i] / other[i];
            }
            return tmp;
        }

        /**
         * Divides each dimension with value.
         *
         * @param value value for division
         * @return the new DataSpace
         */
        HDINLINE DataSpace<DIM> operator/(int value) const
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = (*this)[i] / value;
            }
            return tmp;
        }

        /**
         * Multiplies dimensions of two DataSpaces dimension-wise.
         *
         * @param other DataSpace to multiply with
         * @return the new DataSpace
         */
        HDINLINE DataSpace<DIM> operator*(DataSpace<DIM> const& other) const
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = (*this)[i] * other[i];
            }
            return tmp;
        }

        /**
         * Multiplies each dimension with value.
         *
         * @param value value for multiplication
         * @return the new DataSpace
         */
        HDINLINE DataSpace<DIM> operator*(int value) const
        {
            DataSpace<DIM> tmp;
            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp[i] = (*this)[i] * value;
            }
            return tmp;
        }

        /**
         * Give DataSpace were all dimensions set to init value
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
         * Returns total count of elements by multiplying all dimensions.
         *
         * @return total count of elements
         */
        HDINLINE int getElementCount() const
        {
            int tmp = 1;

            for (uint32_t i = 0; i < DIM; ++i)
            {
                tmp *= (*this)[i];
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

        HDINLINE operator dim3() const;

    };

    //######METHODS#####

    template <>
    HDINLINE DataSpace<DIM3>::operator dim3() const
    {
        return dim3(x(), y(), z());
    }

    template <>
    HDINLINE DataSpace<DIM2>::operator dim3() const
    {
        return dim3(x(), y());
    }

    template <>
    HDINLINE DataSpace<DIM1>::operator dim3() const
    {
        return dim3(x());
    }

    template <>
    inline DataSpace<DIM1>::DataSpace(int x)
    {
        this->x() = x;
    }

    template <>
    inline DataSpace<DIM2>::DataSpace(int x, int y)
    {
        this->x() = x;
        this->y() = y;
    }

    template <>
    inline DataSpace<DIM3>::DataSpace(int x, int y, int z)
    {
        this->x() = x;
        this->y() = y;
        this->z() = z;
    }

} //namespace PMacc


#endif	/* _DATASPACE_HPP */

