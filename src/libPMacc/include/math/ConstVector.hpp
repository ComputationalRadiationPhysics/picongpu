/**
 * Copyright 2014 Rene Widera
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

#include "types.h"
#include "math/vector/Vector.tpp"
#include "ppFunctions.hpp"

/* select namespace depending on __CUDA_ARCH__ compiler flag*/
#ifdef __CUDA_ARCH__ //we are on gpu
#define PMACC_USING_STATIC_CONST_VECTOR_NAMESPACE(id) using namespace PMACC_JOIN(pmacc_static_const_vector_device,id)
#else
#define PMACC_USING_STATIC_CONST_VECTOR_NAMESPACE(id) using namespace PMACC_JOIN(pmacc_static_const_vector_host,id)
#endif

/** @see PMACC_CONST_VECTOR documentation, only unique "id" is added
 *
 * @param id unique precompiler id to create unique namespaces
 */
#define PMACC_STATIC_CONST_VECTOR_DIM(id,Name,Type,Dim,count,...)               \
namespace PMACC_JOIN(pmacc_static_const_storage,id)                            \
{                                                                              \
    namespace PMACC_JOIN(pmacc_static_const_vector_device,id)                  \
    {                                                                          \
       /* store all values in a const C array on device*/                      \
        __constant__ const Type PMACC_JOIN(Name,_data)[]={__VA_ARGS__};        \
    } /*namespace pmacc_static_const_vector_device + id */                     \
    namespace PMACC_JOIN( pmacc_static_const_vector_host,id)                   \
    {                                                                          \
        /* store all values in a const C array on device*/                     \
        const Type PMACC_JOIN(Name,_data)[]={__VA_ARGS__};                     \
    } /* namespace pmacc_static_const_vector_host + id  */                     \
    /* select host or device namespace depending on __CUDA_ARCH__ compiler flag*/ \
    PMACC_USING_STATIC_CONST_VECTOR_NAMESPACE(id);                             \
    template<typename T_Type, int T_Dim>                                       \
    struct ConstArrayStorage                                                   \
    {                                                                          \
        typedef T_Type type;                                                   \
        static const int dim=T_Dim;                                            \
        HDINLINE type& operator[](const int idx)                               \
        {                                                                      \
            /*access const C array with the name of array*/                    \
            return PMACC_JOIN(Name,_data)[idx];                                \
        }                                                                      \
        HDINLINE const type& operator[](const int idx) const                   \
        {                                                                      \
        /*access const C array with the name of array*/                    \
            return PMACC_JOIN(Name,_data)[idx];                                \
        }                                                                      \
    };                                                                         \
    /*define a const vector type, ConstArrayStorage is used as Storage policy*/\
    typedef PMacc::math::Vector<                                               \
        Type,                                                                  \
        Dim,                                                                   \
        PMacc::math::StandartAccessor,                                         \
        PMacc::math::StandartNavigator,                                        \
        ConstArrayStorage > PMACC_JOIN(Name,_t);                               \
    namespace PMACC_JOIN(pmacc_static_const_vector_device,id)                  \
    {                                                                          \
        /* create const instance on device */                                  \
        __constant__ const PMACC_JOIN(Name,_t) Name;                           \
    } /* namespace pmacc_static_const_vector_device + id */                    \
    namespace PMACC_JOIN( pmacc_static_const_vector_host,id)                   \
    {                                                                          \
        /* create const instance on host*/                                     \
        const PMACC_JOIN(Name,_t) Name;                                        \
    } /* namespace pmacc_static_const_vector_host + id  */                     \
} /* namespace pmacc_static_const_storage + id */                              \
using namespace PMACC_JOIN(pmacc_static_const_storage,id)


/** Create global constant math::Vector with compile time values which can be
 *  used on device and host
 *
 * Support all native C/C++ types (e.g. int, float, double,...) and structs with
 * default constructor
 *
 * @param type type of vector
 * @param dim count of components of the vector
 * @param name name of created vector instance
 * @param ... values for component x,y,z
 *            If dim is 2 the parameter z is optional
 *
 * e.g. PMACC_CONST_VECTOR(float,2,myVector,2.1,4.2);
 *      create math:Vector<float,2> myVector(2.1,4.2); //as global const vector
 *      The type of the created vector is "name_t" -> in this case "myVector_t"
 */
#define PMACC_CONST_VECTOR(type,dim,name,...)                                   \
    PMACC_STATIC_CONST_VECTOR_DIM(__COUNTER__,name,type,dim,PMACC_COUNT_ARGS(__VA_ARGS__),__VA_ARGS__)

