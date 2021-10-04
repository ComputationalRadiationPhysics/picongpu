/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Sergei Bastrakov
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


#include "pmacc/algorithms/PromoteType.hpp"
#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/algorithms/math.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/mpi/GetMPI_StructAsArray.hpp"
#include "pmacc/traits/GetComponentsType.hpp"
#include "pmacc/traits/GetInitializedInstance.hpp"
#include "pmacc/traits/GetNComponents.hpp"

#include <utility>

#include "pmacc/math/vector/compile-time/Vector.hpp"

namespace pmacc
{
    namespace traits
    {
        template<typename T_DataType, uint32_t T_dim>
        struct GetComponentsType<pmacc::math::Vector<T_DataType, T_dim>, false>
        {
            using type = typename pmacc::math::Vector<T_DataType, T_dim>::type;
        };

        template<typename T_DataType, uint32_t T_dim>
        struct GetNComponents<pmacc::math::Vector<T_DataType, T_dim>, false>
        {
            static constexpr uint32_t value = (uint32_t) pmacc::math::Vector<T_DataType, T_dim>::dim;
        };

        template<typename T_Type, uint32_t T_dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
        struct GetInitializedInstance<math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>>
        {
            using Type = math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>;
            using ValueType = typename Type::type;

            HDINLINE Type operator()(const ValueType value) const
            {
                return Type::create(value);
            }
        };

    } // namespace traits
} // namespace pmacc


namespace pmacc
{
    namespace math
    {
        /*! Specialisation of cross where base is a vector with three components */
        template<typename Type>
        struct Cross<::pmacc::math::Vector<Type, DIM3>, ::pmacc::math::Vector<Type, DIM3>>
        {
            using myType = ::pmacc::math::Vector<Type, DIM3>;
            using result = myType;

            HDINLINE myType operator()(const myType& lhs, const myType& rhs)
            {
                return myType(
                    lhs.y() * rhs.z() - lhs.z() * rhs.y(),
                    lhs.z() * rhs.x() - lhs.x() * rhs.z(),
                    lhs.x() * rhs.y() - lhs.y() * rhs.x());
            }
        };

        /*! Specialisation of Dot where base is a vector */
        template<typename Type, uint32_t dim>
        struct Dot<::pmacc::math::Vector<Type, dim>, ::pmacc::math::Vector<Type, dim>>
        {
            using myType = ::pmacc::math::Vector<Type, dim>;
            using result = Type;

            HDINLINE result operator()(const myType& a, const myType& b)
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
        template<typename Type, uint32_t dim, typename Storage>
        struct L2norm2<::pmacc::math::Vector<Type, dim, StandardAccessor, StandardNavigator, Storage>>
        {
            using result = typename ::pmacc::math::Vector<Type, dim>::type;

            HDINLINE result operator()(const ::pmacc::math::Vector<Type, dim>& vector)
            {
                result tmp = pmacc::math::norm(vector.x());
                for(uint32_t i = 1; i < dim; ++i)
                    tmp += pmacc::math::norm(vector[i]);
                return tmp;
            }
        };

        /** specialize l2norm algorithm
         */
        template<typename Type, uint32_t dim, typename Storage>
        struct L2norm<::pmacc::math::Vector<Type, dim, StandardAccessor, StandardNavigator, Storage>>
        {
            using result = typename ::pmacc::math::Vector<Type, dim, StandardAccessor, StandardNavigator, Storage>::type;

            HDINLINE result operator()(const ::pmacc::math::Vector<Type, dim, StandardAccessor, StandardNavigator, Storage>& vector)
            {
                result tmp = pmacc::math::l2norm2(vector);
                return cupla::math::sqrt(tmp);
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


/* Using the free alpaka functions `alpaka::math::*` will result into `__host__ __device__`
 * errors, therefore the alpaka math trait must be used.
 */
#define PMACC_UNARY_APAKA_MATH_SPECIALIZATION(functionName, alpakaMathTrait)                                          \
    template<typename T_Ctx, typename T_ScalarType, uint32_t T_dim>                                                   \
    struct alpakaMathTrait<T_Ctx, ::pmacc::math::Vector<T_ScalarType, T_dim>, void>                                   \
    {                                                                                                                 \
        using ResultType = ::pmacc::math::Vector<T_ScalarType, T_dim>;                                                \
                                                                                                                      \
        ALPAKA_FN_ACC auto operator()(                                                                                \
            T_Ctx const& mathConcept,                                                                                 \
            ::pmacc::math::Vector<T_ScalarType, T_dim> const& vector) -> ResultType                                   \
        {                                                                                                             \
            PMACC_CASSERT(T_dim > 0);                                                                                 \
                                                                                                                      \
            ResultType tmp;                                                                                           \
            for(uint32_t i = 0; i < T_dim; ++i)                                                                       \
                tmp[i] = alpaka::math::functionName(mathConcept, vector[i]);                                          \
            return tmp;                                                                                               \
        }                                                                                                             \
    }

namespace alpaka
{
    namespace math
    {
        namespace trait
        {
            /*! Specialisation of pow where base is a vector and exponent is a scalar
             *
             * Create pow separately for every component of the vector.
             */
            template<typename T_Ctx, typename T_ScalarType, uint32_t T_dim>
            struct Pow<T_Ctx, ::pmacc::math::Vector<T_ScalarType, T_dim>, T_ScalarType, void>
            {
                using ResultType = typename ::pmacc::math::Vector<T_ScalarType, T_dim>::type;

                ALPAKA_FN_HOST_ACC auto operator()(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Vector<T_ScalarType, T_dim> const& vector,
                    T_ScalarType const& exponent) -> ResultType
                {
                    PMACC_CASSERT(T_dim > 0);
                    ResultType tmp;
                    for(uint32_t i = 0; i < T_dim; ++i)
                        tmp[i] = cupla::pow(vector[i], exponent);
                    return tmp;
                }
            };

            template<typename T_Ctx, typename T_ScalarType1, typename T_ScalarType2, uint32_t T_dim>
            struct Min<
                T_Ctx,
                ::pmacc::math::Vector<T_ScalarType1, T_dim>,
                ::pmacc::math::Vector<T_ScalarType2, T_dim>,
                void>
            {
                using ScalarResultType = ALPAKA_DECAY_T(decltype(alpaka::math::min(
                    std::declval<T_Ctx>(),
                    std::declval<T_ScalarType1>(),
                    std::declval<T_ScalarType2>())));
                using ResultType = ::pmacc::math::Vector<ScalarResultType, T_dim>;

                ALPAKA_FN_HOST_ACC auto operator()(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Vector<T_ScalarType1, T_dim> const& vector1,
                    ::pmacc::math::Vector<T_ScalarType1, T_dim> const& vector2) -> ResultType
                {
                    PMACC_CASSERT(T_dim > 0);
                    ResultType tmp;
                    for(uint32_t i = 0; i < T_dim; ++i)
                        tmp[i] = alpaka::math::min(mathConcept, vector1[i], vector2[i]);
                    return tmp;
                }
            };

            template<typename T_Ctx, typename T_ScalarType1, typename T_ScalarType2, uint32_t T_dim>
            struct Max<
                T_Ctx,
                ::pmacc::math::Vector<T_ScalarType1, T_dim>,
                ::pmacc::math::Vector<T_ScalarType2, T_dim>,
                void>
            {
                using ScalarResultType = ALPAKA_DECAY_T(decltype(alpaka::math::max(
                    std::declval<T_Ctx>(),
                    std::declval<T_ScalarType1>(),
                    std::declval<T_ScalarType2>())));
                using ResultType = ::pmacc::math::Vector<ScalarResultType, T_dim>;

                ALPAKA_FN_HOST_ACC auto operator()(
                    T_Ctx const& mathConcept,
                    ::pmacc::math::Vector<T_ScalarType1, T_dim> const& vector1,
                    ::pmacc::math::Vector<T_ScalarType1, T_dim> const& vector2) -> ResultType
                {
                    PMACC_CASSERT(T_dim > 0);
                    ResultType tmp;
                    for(uint32_t i = 0; i < T_dim; ++i)
                        tmp[i] = alpaka::math::max(mathConcept, vector1[i], vector2[i]);
                    return tmp;
                }
            };

            // Exp specialization
            PMACC_UNARY_APAKA_MATH_SPECIALIZATION(exp, Exp);

            // Floor specialization
            PMACC_UNARY_APAKA_MATH_SPECIALIZATION(floor, Floor);

            // Abs specialization
            PMACC_UNARY_APAKA_MATH_SPECIALIZATION(abs, Abs);

        } // namespace trait
    } // namespace math
} // namespace alpaka

namespace pmacc
{
    namespace algorithms
    {
        namespace precisionCast
        {
            template<typename CastToType, uint32_t dim, typename T_Accessor, typename T_Navigator, typename T_Storage>
            struct TypeCast<CastToType, ::pmacc::math::Vector<CastToType, dim, T_Accessor, T_Navigator, T_Storage>>
            {
                using result = ::pmacc::math::Vector<CastToType, dim>;
                using ParamType = ::pmacc::math::Vector<CastToType, dim, T_Accessor, T_Navigator, T_Storage>;

                HDINLINE result operator()(ParamType const& vector) const
                {
                    return vector;
                }
            };

            template<
                typename CastToType,
                typename OldType,
                uint32_t dim,
                typename T_Accessor,
                typename T_Navigator,
                typename T_Storage>
            struct TypeCast<CastToType, ::pmacc::math::Vector<OldType, dim, T_Accessor, T_Navigator, T_Storage>>
            {
                using result = ::pmacc::math::Vector<CastToType, dim>;
                using ParamType = ::pmacc::math::Vector<OldType, dim, T_Accessor, T_Navigator, T_Storage>;

                HDINLINE result operator()(const ParamType& vector) const
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
            template<typename PromoteToType, typename OldType, uint32_t dim>
            struct promoteType<PromoteToType, ::pmacc::math::Vector<OldType, dim>>
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
