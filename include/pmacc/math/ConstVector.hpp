/* Copyright 2014-2021 Rene Widera, Benjamin Worpitz
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

#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/ppFunctions.hpp"
#include "pmacc/types.hpp"

/* select namespace depending on __CUDA_ARCH__ compiler flag*/
#if(CUPLA_DEVICE_COMPILE == 1) // we are on gpu
#    define PMACC_USING_STATIC_CONST_VECTOR_NAMESPACE(id)                                                             \
        using namespace PMACC_JOIN(pmacc_static_const_vector_device, id)
#else
#    define PMACC_USING_STATIC_CONST_VECTOR_NAMESPACE(id)                                                             \
        using namespace PMACC_JOIN(pmacc_static_const_vector_host, id)
#endif

#if defined(__CUDACC__) || BOOST_COMP_HIP
#    define PMACC_STATIC_CONST_VECTOR_DIM_DEF_CUDA(id, Name, Type, ...)                                               \
        namespace PMACC_JOIN(pmacc_static_const_vector_device, id)                                                    \
        {                                                                                                             \
            /* store all values in a const C array on device*/                                                        \
            __constant__ const Type PMACC_JOIN(Name, _data)[] = {__VA_ARGS__};                                        \
        } /*namespace pmacc_static_const_vector_device + id */
#else
#    define PMACC_STATIC_CONST_VECTOR_DIM_DEF_CUDA(id, Name, Type, ...)
#endif

#define PMACC_PRAGMA_QUOTE(x) _Pragma(#x)
#define PMACC_PRAGMA_OACC_DECLARE_ARRAY(name, count)
#define PMACC_PRAGMA_OMP_TARGET_BEGIN_DECLARE
#define PMACC_PRAGMA_OMP_TARGET_END_DECLARE
#define PMACC_TARGET_CONSTEXPR constexpr

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED
#    undef PMACC_PRAGMA_OACC_DECLARE_ARRAY(name, count)
#    undef PMACC_TARGET_CONSTEXPR
// might need to remove parentheses from macro argument count to clean up copyin clause, but works with NVHPC
#    define PMACC_PRAGMA_OACC_DECLARE_ARRAY(name, count) PMACC_PRAGMA_QUOTE(acc declare copyin(name))
#    define PMACC_TARGET_CONSTEXPR
#elif defined ALPAKA_ACC_ANY_BT_OMP5_ENABLED
#    undef PMACC_PRAGMA_OMP_TARGET_BEGIN_DECLARE
#    undef PMACC_PRAGMA_OMP_TARGET_END_DECLARE
// the single-pragma declare (more like the OpenACC version above) does not work with clang 11
#    define PMACC_PRAGMA_OMP_TARGET_BEGIN_DECLARE _Pragma("omp declare target")
#    define PMACC_PRAGMA_OMP_TARGET_END_DECLARE _Pragma("omp end declare target")
#endif

/** define a const vector
 *
 * create type definition `Name_t`
 * @see PMACC_CONST_VECTOR documentation, only unique "id" is added
 *
 * @param id unique precompiler id to create unique namespaces
 */
#define PMACC_STATIC_CONST_VECTOR_DIM_DEF(id, Name, Type, Dim, count, ...)                                            \
    namespace PMACC_JOIN(pmacc_static_const_storage, id)                                                              \
    {                                                                                                                 \
        PMACC_STATIC_CONST_VECTOR_DIM_DEF_CUDA(id, Name, Type, __VA_ARGS__);                                          \
        namespace PMACC_JOIN(pmacc_static_const_vector_host, id)                                                      \
        {                                                                                                             \
            /* store all values in a const C array on host*/                                                          \
            PMACC_PRAGMA_OMP_TARGET_BEGIN_DECLARE                                                                     \
            PMACC_TARGET_CONSTEXPR Type PMACC_JOIN(Name, _data)[] = {__VA_ARGS__};                                    \
            PMACC_PRAGMA_OMP_TARGET_END_DECLARE                                                                       \
            PMACC_PRAGMA_OACC_DECLARE_ARRAY(PMACC_JOIN(Name, _data), count)                                           \
        } /* namespace pmacc_static_const_vector_host + id  */                                                        \
        /* select host or device namespace depending on __CUDA_ARCH__ compiler flag*/                                 \
        PMACC_USING_STATIC_CONST_VECTOR_NAMESPACE(id);                                                                \
        template<typename T_Type, int T_Dim>                                                                          \
        struct ConstArrayStorage                                                                                      \
        {                                                                                                             \
            PMACC_CASSERT_MSG(                                                                                        \
                __PMACC_CONST_VECTOR_dimension_needs_to_be_less_than_or_equal_to_the_number_of_arguments__,           \
                Dim <= count);                                                                                        \
            static constexpr bool isConst = true;                                                                     \
            typedef T_Type type;                                                                                      \
            static constexpr int dim = T_Dim;                                                                         \
                                                                                                                      \
            HDINLINE const type& operator[](const int idx) const                                                      \
            {                                                                                                         \
                /*access const C array with the name of array*/                                                       \
                return PMACC_JOIN(Name, _data)[idx];                                                                  \
            }                                                                                                         \
        };                                                                                                            \
        /*define a const vector type, ConstArrayStorage is used as Storage policy*/                                   \
        typedef const pmacc::math::                                                                                   \
            Vector<Type, Dim, pmacc::math::StandardAccessor, pmacc::math::StandardNavigator, ConstArrayStorage>       \
                PMACC_JOIN(Name, _t);                                                                                 \
    } /* namespace pmacc_static_const_storage + id */                                                                 \
    using namespace PMACC_JOIN(pmacc_static_const_storage, id)

#if defined(__CUDACC__) || BOOST_COMP_HIP
#    define PMACC_STATIC_CONST_VECTOR_DIM_INSTANCE_CUDA(Name, id)                                                     \
        namespace PMACC_JOIN(pmacc_static_const_vector_device, id)                                                    \
        {                                                                                                             \
            /* create const instance on device */                                                                     \
            __constant__ const PMACC_JOIN(Name, _t) Name;                                                             \
        } /* namespace pmacc_static_const_vector_device + id */
#else
#    define PMACC_STATIC_CONST_VECTOR_DIM_INSTANCE_CUDA(Name, id)
#endif

/** create a instance of type `Name_t` with the name `Name`
 */
#define PMACC_STATIC_CONST_VECTOR_DIM_INSTANCE(id, Name, Type, Dim, count, ...)                                       \
    namespace PMACC_JOIN(pmacc_static_const_storage, id)                                                              \
    {                                                                                                                 \
        /* Conditionally define the instance on CUDA devices */                                                       \
        PMACC_STATIC_CONST_VECTOR_DIM_INSTANCE_CUDA(Name, id)                                                         \
        namespace PMACC_JOIN(pmacc_static_const_vector_host, id)                                                      \
        {                                                                                                             \
            /* create const instance on host*/                                                                        \
            constexpr PMACC_JOIN(Name, _t) Name;                                                                      \
        } /* namespace pmacc_static_const_vector_host + id  */                                                        \
    } /* namespace pmacc_static_const_storage + id */

/** @see PMACC_CONST_VECTOR documentation, only unique "id" is added
 *
 * @param id unique precompiler id to create unique namespaces
 */
#define PMACC_STATIC_CONST_VECTOR_DIM(id, Name, Type, Dim, count, ...)                                                \
    PMACC_STATIC_CONST_VECTOR_DIM_DEF(id, Name, Type, Dim, count, __VA_ARGS__);                                       \
    PMACC_STATIC_CONST_VECTOR_DIM_INSTANCE(id, Name, Type, Dim, count, __VA_ARGS__)


/** define a const vector
 *
 * for description @see PMACC_CONST_VECTOR
 *
 * create type definition `name_t`
 */
#define PMACC_CONST_VECTOR_DEF(type, dim, name, ...)                                                                  \
    PMACC_STATIC_CONST_VECTOR_DIM_DEF(__COUNTER__, name, type, dim, PMACC_COUNT_ARGS(type, __VA_ARGS__), __VA_ARGS__)

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
#define PMACC_CONST_VECTOR(type, dim, name, ...)                                                                      \
    PMACC_STATIC_CONST_VECTOR_DIM(__COUNTER__, name, type, dim, PMACC_COUNT_ARGS(type, __VA_ARGS__), __VA_ARGS__)
