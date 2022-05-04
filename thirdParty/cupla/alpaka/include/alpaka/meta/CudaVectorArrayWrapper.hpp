/* Copyright 2022 Jiří Vyskočil, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

#    include <alpaka/core/Common.hpp>

#    include <functional>
#    include <initializer_list>
#    include <numeric>
#    include <type_traits>

namespace alpaka::meta
{
    namespace detail
    {
        template<typename TScalar, unsigned N>
        struct CudaVectorArrayTypeTraits;

        template<>
        struct CudaVectorArrayTypeTraits<float, 1>
        {
            using type = float1;
        };

        template<>
        struct CudaVectorArrayTypeTraits<float, 2>
        {
            using type = float2;
        };

        template<>
        struct CudaVectorArrayTypeTraits<float, 3>
        {
            using type = float3;
        };

        template<>
        struct CudaVectorArrayTypeTraits<float, 4>
        {
            using type = float4;
        };

        template<>
        struct CudaVectorArrayTypeTraits<double, 1>
        {
            using type = double1;
        };

        template<>
        struct CudaVectorArrayTypeTraits<double, 2>
        {
            using type = double2;
        };

        template<>
        struct CudaVectorArrayTypeTraits<double, 3>
        {
            using type = double3;
        };

        template<>
        struct CudaVectorArrayTypeTraits<double, 4>
        {
            using type = double4;
        };

        template<>
        struct CudaVectorArrayTypeTraits<unsigned, 1>
        {
            using type = uint1;
        };

        template<>
        struct CudaVectorArrayTypeTraits<unsigned, 2>
        {
            using type = uint2;
        };

        template<>
        struct CudaVectorArrayTypeTraits<unsigned, 3>
        {
            using type = uint3;
        };

        template<>
        struct CudaVectorArrayTypeTraits<unsigned, 4>
        {
            using type = uint4;
        };

        template<>
        struct CudaVectorArrayTypeTraits<int, 1>
        {
            using type = int1;
        };

        template<>
        struct CudaVectorArrayTypeTraits<int, 2>
        {
            using type = int2;
        };

        template<>
        struct CudaVectorArrayTypeTraits<int, 3>
        {
            using type = int3;
        };

        template<>
        struct CudaVectorArrayTypeTraits<int, 4>
        {
            using type = int4;
        };
    } // namespace detail

    /// Helper struct providing [] subscript access to CUDA vector types
    template<typename TScalar, unsigned N>
    struct CudaVectorArrayWrapper;

    template<typename TScalar>
    struct CudaVectorArrayWrapper<TScalar, 4> : public detail::CudaVectorArrayTypeTraits<TScalar, 4>::type
    {
        using value_type = TScalar;
        constexpr static unsigned size = 4;
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE CudaVectorArrayWrapper(std::initializer_list<TScalar> init)
        {
            auto it = std::begin(init);
            this->x = *it++;
            this->y = *it++;
            this->z = *it++;
            this->w = *it++;
        }
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr value_type& operator[](int const k) noexcept
        {
            assert(k >= 0 && k < 4);
            return k == 0 ? this->x : (k == 1 ? this->y : (k == 2 ? this->z : this->w));
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const value_type& operator[](int const k) const noexcept
        {
            assert(k >= 0 && k < 4);
            return k == 0 ? this->x : (k == 1 ? this->y : (k == 2 ? this->z : this->w));
        }
    };

    template<typename TScalar>
    struct CudaVectorArrayWrapper<TScalar, 3> : public detail::CudaVectorArrayTypeTraits<TScalar, 3>::type
    {
        using value_type = TScalar;
        constexpr static unsigned size = 3;
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE CudaVectorArrayWrapper(std::initializer_list<TScalar> init)
        {
            auto it = std::begin(init);
            this->x = *it++;
            this->y = *it++;
            this->z = *it++;
        }
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr value_type& operator[](int const k) noexcept
        {
            assert(k >= 0 && k < 3);
            return k == 0 ? this->x : (k == 1 ? this->y : this->z);
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const value_type& operator[](int const k) const noexcept
        {
            assert(k >= 0 && k < 3);
            return k == 0 ? this->x : (k == 1 ? this->y : this->z);
        }
    };

    template<typename TScalar>
    struct CudaVectorArrayWrapper<TScalar, 2> : public detail::CudaVectorArrayTypeTraits<TScalar, 2>::type
    {
        using value_type = TScalar;
        constexpr static unsigned size = 2;
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE CudaVectorArrayWrapper(std::initializer_list<TScalar> init)
        {
            auto it = std::begin(init);
            this->x = *it++;
            this->y = *it++;
        }
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr value_type& operator[](int const k) noexcept
        {
            assert(k >= 0 && k < 2);
            return k == 0 ? this->x : this->y;
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const value_type& operator[](int const k) const noexcept
        {
            assert(k >= 0 && k < 2);
            return k == 0 ? this->x : this->y;
        }
    };

    template<typename TScalar>
    struct CudaVectorArrayWrapper<TScalar, 1> : public detail::CudaVectorArrayTypeTraits<TScalar, 1>::type
    {
        using value_type = TScalar;
        constexpr static unsigned size = 1;
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE CudaVectorArrayWrapper(std::initializer_list<TScalar> init)
        {
            auto it = std::begin(init);
            this->x = *it;
        }
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr value_type& operator[]([[maybe_unused]] int const k) noexcept
        {
            assert(k == 0);
            return this->x;
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const value_type& operator[](
            [[maybe_unused]] int const k) const noexcept
        {
            assert(k == 0);
            return this->x;
        }
    };
} // namespace alpaka::meta

namespace std
{
    /// Specialization of std::tuple_size for \a float4_array
    template<typename T, unsigned N>
    struct tuple_size<alpaka::meta::CudaVectorArrayWrapper<T, N>> : integral_constant<size_t, N>
    {
    };
} // namespace std

#endif
