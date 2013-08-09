/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera
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
 * File:   MappingDescription.h
 * Author: burau
 *
 * Created on 13. Dezember 2010, 14:29
 */

#ifndef CUDAGRIDDIMRESTRICTIONS_HPP
#define	CUDAGRIDDIMRESTRICTIONS_HPP

#include "types.h"
#include <stdexcept>
#include "dimensions/DataSpace.hpp"

namespace PMacc
{

/*! Handles cuda restriction that gridDim.z()  != 1 is not allowed
 */
template<unsigned DIM>
class CudaGridDimRestrictions
{
protected:
    /*! Create a mapping of any gridDim to a allowed gridDim where gridDim.z() =1
     * @param value a gridDim
     * @return a valid cuda gridDim as DataSpace
     */
    HDINLINE DataSpace<DIM> reduce(const DataSpace<DIM> &value);

    /*! Is the toward operation of reduce
     * - call this with cuda blockIdx than the real 3D block index is returned
     * - call this with cuda gridDim than the real 3D grid dimension is returnd
     * @param value a gridDim
     * @return gridDim whith size before reduce was called.
     */
    HDINLINE DataSpace<DIM> extend(const DataSpace<DIM> &value);
};

template<>
class CudaGridDimRestrictions<DIM2>
{
protected:

    HDINLINE DataSpace<DIM2> reduce(const DataSpace<DIM2> &value)
    {
        return value;
    }

    HDINLINE DataSpace<DIM2> extend(const DataSpace<DIM2> &value)
    {
        return value;
    }
};

template<>
class CudaGridDimRestrictions<DIM3>
{
protected:

    CudaGridDimRestrictions()
    : z(1)
    {
    }

    HDINLINE DataSpace<DIM3> reduce(const DataSpace<DIM3> &value)
    {
        z = value.z();
        return DataSpace<DIM3 > (value.x() * z, value.y(), 1);
    }

    HDINLINE DataSpace<DIM3> extend(const DataSpace<DIM3> &value)
    {
        return DataSpace<DIM3 > (value.x() / z, value.y(), value.x() % z);
    }
private:

    PMACC_ALIGN(z, int);


};

} // namespace PMacc



#endif	/* CUDAGRIDDIMRESTRICTIONS_HPP */

