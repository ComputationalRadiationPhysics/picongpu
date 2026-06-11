/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once


#include "pmacc/algorithms/PromoteType.hpp"
#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/algorithms/math.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/mpi/GetMPI_StructAsArray.hpp"
#include "pmacc/traits/GetComponentsType.hpp"
#include "pmacc/traits/GetNComponents.hpp"

#include <utility>

#include "pmacc/math/vector/compile-time/Vector.hpp"

namespace pmacc
{
    namespace traits
    {
        template<typename T_DataType, uint32_t T_dim, typename T_Storage>
        struct GetComponentsType<pmacc::math::Vector<T_DataType, T_dim, T_Storage>, false>
        {
            using type = typename pmacc::math::Vector<T_DataType, T_dim, T_Storage>::type;
        };

        template<typename T_DataType, uint32_t T_dim, typename T_Storage>
        struct GetNComponents<pmacc::math::Vector<T_DataType, T_dim, T_Storage>, false>
        {
            static constexpr uint32_t value = (uint32_t) pmacc::math::Vector<T_DataType, T_dim, T_Storage>::dim;
        };
    } // namespace traits
} // namespace pmacc

namespace pmacc
{
    namespace math
    {
        /*! Specialisation of cross where base is a vector with three components */
        template<typename Type, typename T_StorageLeft, typename T_StorageRight>
        struct Cross<
            ::pmacc::math::Vector<Type, DIM3, T_StorageLeft>,
            ::pmacc::math::Vector<Type, DIM3, T_StorageRight>>
        {
            using myType = ::pmacc::math::Vector<Type, DIM3>;
            using result = myType;

            HDINLINE myType operator()(myType const& lhs, myType const& rhs)
            {
                return myType(
                    lhs.y() * rhs.z() - lhs.z() * rhs.y(),
                    lhs.z() * rhs.x() - lhs.x() * rhs.z(),
                    lhs.x() * rhs.y() - lhs.y() * rhs.x());
            }
        };

        /*! Specialisation of Dot where base is a vector */
        template<typename Type, uint32_t dim, typename T_StorageLeft, typename T_StorageRight>
        struct Dot<::pmacc::math::Vector<Type, dim, T_StorageLeft>, ::pmacc::math::Vector<Type, dim, T_StorageRight>>
        {
            using myType = ::pmacc::math::Vector<Type, dim>;
            using result = Type;

            HDINLINE result operator()(myType const& a, myType const& b)
            {
                PMACC_CASSERT(dim > 0);
                result tmp = a.x() * b.x();
                for(uint32_t i = 1; i < dim; i++)
                    tmp += a[i] * b[i];
                return tmp;
            }
        };

        /** specialize l2norm2 algorithm
         */
        template<typename Type, uint32_t dim, typename T_Storage>
        struct L2norm2<::pmacc::math::Vector<Type, dim, T_Storage>>
        {
            using result = typename ::pmacc::math::Vector<Type, dim>::type;

            HDINLINE result operator()(::pmacc::math::Vector<Type, dim, T_Storage> const& vector)
            {
                result tmp = pmacc::math::norm(vector.x());
                for(uint32_t i = 1; i < dim; ++i)
                    tmp += pmacc::math::norm(vector[i]);
                return tmp;
            }
        };

        /** specialize l2norm algorithm
         */
        template<typename Type, uint32_t dim, typename T_Storage>
        struct L2norm<::pmacc::math::Vector<Type, dim, T_Storage>>
        {
            using result = typename ::pmacc::math::Vector<Type, dim>::type;

            HDINLINE result operator()(::pmacc::math::Vector<Type, dim> const& vector)
            {
                result tmp = pmacc::math::l2norm2(vector);
                return pmacc::math::sqrt(tmp);
            }
        };

        template<typename T_Vector, uint32_t T_direction>
        HDINLINE T_Vector basisVector()
        {
            using Result = typename CT::make_BasisVector<T_Vector::dim, T_direction, typename T_Vector::type>::type;
            return Result::toRT();
        }

    } // namespace math
} // namespace pmacc

namespace alpaka
{
    namespace math
    {
        namespace trait
        {
            /* Specialise min/max for vectors so that `pmacc::math::min`/`max` operate element-wise.
             *
             * Unlike the other math functions, these cannot be left to the element-wise overloads in
             * VectorOps.hpp: the generic alpaka Min/Max trait falls back to the unconstrained
             * `std::min`/`std::max` templates, which match whole vectors and break (ambiguity on the host,
             * `Vector<bool>`-to-`bool` conversion on CUDA). Specialising the trait keeps the generic
             * `pmacc::math::min`/`max` well-formed for vector arguments.
             */
            template<
                typename T_Ctx,
                typename T_ScalarType1,
                typename T_ScalarType2,
                uint32_t T_dim,
                typename T_Storage1,
                typename T_Storage2>
            struct Min<
                T_Ctx,
                ::pmacc::math::Vector<T_ScalarType1, T_dim, T_Storage1>,
                ::pmacc::math::Vector<T_ScalarType2, T_dim, T_Storage2>,
                void>
            {
                using ScalarResultType = std::decay_t<decltype(alpaka::math::min(
                    std::declval<T_Ctx>(),
                    std::declval<T_ScalarType1>(),
                    std::declval<T_ScalarType2>()))>;
                using ResultType = ::pmacc::math::Vector<ScalarResultType, T_dim>;

                ALPAKA_FN_HOST_ACC auto operator()(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Vector<T_ScalarType1, T_dim, T_Storage1> const& vector1,
                    ::pmacc::math::Vector<T_ScalarType1, T_dim, T_Storage2> const& vector2) -> ResultType
                {
                    PMACC_CASSERT(T_dim > 0);
                    ResultType tmp;
                    for(uint32_t i = 0; i < T_dim; ++i)
                        tmp[i] = alpaka::math::min(mathConcept, vector1[i], vector2[i]);
                    return tmp;
                }
            };

            template<
                typename T_Ctx,
                typename T_ScalarType1,
                typename T_ScalarType2,
                uint32_t T_dim,
                typename T_Storage1,
                typename T_Storage2>
            struct Max<
                T_Ctx,
                ::pmacc::math::Vector<T_ScalarType1, T_dim, T_Storage1>,
                ::pmacc::math::Vector<T_ScalarType2, T_dim, T_Storage2>,
                void>
            {
                using ScalarResultType = std::decay_t<decltype(alpaka::math::max(
                    std::declval<T_Ctx>(),
                    std::declval<T_ScalarType1>(),
                    std::declval<T_ScalarType2>()))>;
                using ResultType = ::pmacc::math::Vector<ScalarResultType, T_dim>;

                ALPAKA_FN_HOST_ACC auto operator()(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Vector<T_ScalarType1, T_dim, T_Storage1> const& vector1,
                    ::pmacc::math::Vector<T_ScalarType1, T_dim, T_Storage2> const& vector2) -> ResultType
                {
                    PMACC_CASSERT(T_dim > 0);
                    ResultType tmp;
                    for(uint32_t i = 0; i < T_dim; ++i)
                        tmp[i] = alpaka::math::max(mathConcept, vector1[i], vector2[i]);
                    return tmp;
                }
            };
        } // namespace trait
    } // namespace math

    namespace trait
    {
        //! dimension get trait specialization
        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct DimType<pmacc::math::Vector<T_Type, T_dim, T_Storage>>
        {
            using type = ::alpaka::DimInt<T_dim>;
        };

        //! element type trait specialization
        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct ElemType<pmacc::math::Vector<T_Type, T_dim, T_Storage>>
        {
            using type = T_Type;
        };

        //! extent get trait specialization
        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct GetExtents<pmacc::math::Vector<T_Type, T_dim, T_Storage>, std::enable_if_t<std::is_integral_v<T_Type>>>
        {
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(pmacc::math::Vector<T_Type, T_dim, T_Storage> const& extents)
                -> Vec<::alpaka::DimInt<T_dim>, T_Type>
            {
                Vec<::alpaka::DimInt<T_dim>, T_Type> result;
                for(uint32_t i = 0u; i < T_dim; i++)
                    result[T_dim - 1 - i] = extents[i];
                return result;
            }
        };

        //! offset get trait specialization
        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct GetOffsets<pmacc::math::Vector<T_Type, T_dim, T_Storage>, std::enable_if_t<std::is_integral_v<T_Type>>>
        {
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(pmacc::math::Vector<T_Type, T_dim, T_Storage> const& offsets)
                -> Vec<::alpaka::DimInt<T_dim>, T_Type>
            {
                Vec<::alpaka::DimInt<T_dim>, T_Type> result;
                for(uint32_t i = 0u; i < T_dim; i++)
                    result[T_dim - 1 - i] = offsets[i];
                return result;
            }
        };

        //! size type trait specialization.
        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct IdxType<pmacc::math::Vector<T_Type, T_dim, T_Storage>, std::enable_if_t<std::is_integral_v<T_Type>>>
        {
            using type = T_Type;
        };

    } // namespace trait

} // namespace alpaka

namespace pmacc
{
    namespace algorithms
    {
        namespace precisionCast
        {
            template<typename CastToType, uint32_t dim, typename T_Storage>
            struct TypeCast<CastToType, ::pmacc::math::Vector<CastToType, dim, T_Storage>>
            {
                // do not change the storage policy
                using result = ::pmacc::math::Vector<CastToType, dim, T_Storage>;
                using ParamType = ::pmacc::math::Vector<CastToType, dim, T_Storage>;

                constexpr result operator()(ParamType const& vector) const
                {
                    return vector;
                }
            };

            template<typename CastToType, typename OldType, uint32_t dim, typename T_Storage>
            struct TypeCast<CastToType, ::pmacc::math::Vector<OldType, dim, T_Storage>>
            {
                using result = ::pmacc::math::Vector<CastToType, dim>;
                using ParamType = ::pmacc::math::Vector<OldType, dim, T_Storage>;

                constexpr result operator()(ParamType const& vector) const
                {
                    return result(vector);
                }
            };

        } // namespace precisionCast
    } // namespace algorithms
} // namespace pmacc

namespace pmacc
{
    namespace algorithms
    {
        namespace promoteType
        {
            template<typename PromoteToType, typename OldType, uint32_t dim, typename T_Storage>
            struct promoteType<PromoteToType, ::pmacc::math::Vector<OldType, dim, T_Storage>>
            {
                using PartType = typename promoteType<OldType, PromoteToType>::type;
                using type = ::pmacc::math::Vector<PartType, dim>;
            };

        } // namespace promoteType
    } // namespace algorithms
} // namespace pmacc

namespace pmacc
{
    namespace mpi
    {
        namespace def
        {
            template<uint32_t T_dim>
            struct GetMPI_StructAsArray<::pmacc::math::Vector<float, T_dim>>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_FLOAT, T_dim};
                }
            };

            template<uint32_t T_dim, int T_N>
            struct GetMPI_StructAsArray<::pmacc::math::Vector<float, T_dim>[T_N]>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_FLOAT, T_dim * T_N};
                }
            };

            template<uint32_t T_dim>
            struct GetMPI_StructAsArray<::pmacc::math::Vector<double, T_dim>>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_DOUBLE, T_dim};
                }
            };

            template<uint32_t T_dim, int T_N>
            struct GetMPI_StructAsArray<::pmacc::math::Vector<double, T_dim>[T_N]>
            {
                MPI_StructAsArray operator()() const
                {
                    return {MPI_DOUBLE, T_dim * T_N};
                }
            };

        } // namespace def
    } // namespace mpi
} // namespace pmacc
