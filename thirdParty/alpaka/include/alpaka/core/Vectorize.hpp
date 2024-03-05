/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#include <cstddef>
#include <cstdint>

//! Suggests vectorization of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_VECTORIZE_HINT
//!  for(...){...}`
// \TODO: Implement for other compilers.
// See: http://stackoverflow.com/questions/2706286/pragmas-swp-ivdep-prefetch-support-in-various-compilers
/*#if BOOST_COMP_HPACC
    #define ALPAKA_VECTORIZE_HINT(...)  _Pragma("ivdep")
#elif BOOST_COMP_PGI
    #define ALPAKA_VECTORIZE_HINT(...)  _Pragma("vector")
#elif BOOST_COMP_MSVC
    #define ALPAKA_VECTORIZE_HINT(...)  __pragma(loop(ivdep))
#elif BOOST_COMP_GNUC
    #define ALPAKA_VECTORIZE_HINT(...)  _Pragma("GCC ivdep")
#else
    #define ALPAKA_VECTORIZE_HINT(...)
#endif*/

namespace alpaka::core::vectorization
{
    // The alignment required to enable optimal performance dependant on the target architecture.
    constexpr std::size_t defaultAlignment =
#if defined(__AVX512BW__) || defined(__AVX512F__) || defined(__MIC__)
        64u
#elif defined(__AVX__) || defined(__AVX2__)
        32u
#else
        16u
#endif
        ;

    // Number of elements of the given type that can be processed in parallel in a vector register.
    // By default there is no vectorization.
    template<typename TElem>
    struct GetVectorizationSizeElems
    {
        static constexpr std::size_t value = 1u;
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<double>
    {
        static constexpr std::size_t value =
#if defined(__AVX512F__) || defined(__MIC__)
            // addition (AVX512F,KNC): vaddpd / _mm512_add_pd
            // subtraction (AVX512F,KNC): vsubpd / _mm512_sub_pd
            // multiplication (AVX512F,KNC): vmulpd / _mm512_mul_pd
            8u;
#elif defined(__AVX__)
            // addition (AVX): vaddpd / _mm256_add_pd
            // subtraction (AVX): vsubpd / _mm256_sub_pd
            // multiplication (AVX): vmulpd / _mm256_mul_pd
            4u;
#elif defined(__SSE2__)
            // addition (SSE2): addpd / _mm_add_pd
            // subtraction (SSE2): subpd / _mm_sub_pd
            // multiplication (SSE2): mulpd / _mm_mul_pd
            2u;
#elif defined(__ARM_NEON__)
            // No support for double precision vectorization!
            1u;
#elif defined(__ALTIVEC__)
            2u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<float>
    {
        static constexpr std::size_t value =
#if defined(__AVX512F__) || defined(__MIC__)
            // addition (AVX512F,KNC): vaddps / _mm512_add_ps
            // subtraction (AVX512F,KNC): vsubps / _mm512_sub_ps
            // multiplication (AVX512F,KNC): vmulps / _mm512_mul_ps
            16u;
#elif defined(__AVX__)
            // addition (AVX): vaddps / _mm256_add_ps
            // subtraction (AVX): vsubps / _mm256_sub_ps
            // multiplication (AVX): vmulps / _mm256_mul_ps
            8u;
#elif defined(__SSE__)
            // addition (SSE): addps / _mm_add_ps
            // subtraction (SSE): subps / _mm_sub_ps
            // multiplication (SSE): mulps / _mm_mul_ps
            4u;
#elif defined(__ARM_NEON__)
            4u;
#elif defined(__ALTIVEC__)
            4u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<std::int8_t>
    {
        static constexpr std::size_t value =
#if defined(__AVX512BW__)
            // addition (AVX512BW): vpaddb / _mm512_mask_add_epi8
            // subtraction (AVX512BW): vpsubb / _mm512_sub_epi8
            // multiplication: -
            64u;
#elif defined(__AVX2__)
            // addition (AVX2): vpaddb / _mm256_add_epi8
            // subtraction (AVX2): vpsubb / _mm256_sub_epi8
            // multiplication: -
            32u;
#elif defined(__SSE2__)
            // addition (SSE2): paddb / _mm_add_epi8
            // subtraction (SSE2): psubb / _mm_sub_epi8
            // multiplication: -
            16u;
#elif defined(__ARM_NEON__)
            16u;
#elif defined(__ALTIVEC__)
            16u;
#elif defined(__CUDA_ARCH__)
            // addition: __vadd4
            // subtraction: __vsub4
            // multiplication: -
            4u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<std::uint8_t>
    {
        static constexpr std::size_t value =
#if defined(__AVX512BW__)
            // addition (AVX512BW): vpaddb / _mm512_mask_add_epi8
            // subtraction (AVX512BW): vpsubb / _mm512_sub_epi8
            // multiplication: -
            64u;
#elif defined(__AVX2__)
            // addition (AVX2): vpaddb / _mm256_add_epi8
            // subtraction (AVX2): vpsubb / _mm256_sub_epi8
            // multiplication: -
            32u;
#elif defined(__SSE2__)
            // addition (SSE2): paddb / _mm_add_epi8
            // subtraction (SSE2): psubb / _mm_sub_epi8
            // multiplication: -
            16u;
#elif defined(__ARM_NEON__)
            16u;
#elif defined(__ALTIVEC__)
            16u;
#elif defined(__CUDA_ARCH__)
            // addition: __vadd4
            // subtraction: __vsub4
            // multiplication: -
            4u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<std::int16_t>
    {
        static constexpr std::size_t value =
#if defined(__AVX512BW__)
            // addition (AVX512BW): vpaddw / _mm512_mask_add_epi16
            // subtraction (AVX512BW): vpsubw / _mm512_mask_sub_epi16
            // multiplication (AVX512BW): vpmullw / _mm512_mask_mullo_epi16
            32u;
#elif defined(__AVX2__)
            // addition (AVX2): vpaddw / _mm256_add_epi16
            // subtraction (AVX2): vpsubw / _mm256_sub_epi16
            // multiplication (AVX2): vpmullw / _mm256_mullo_epi16
            16u;
#elif defined(__SSE2__)
            // addition (SSE2): paddw / _mm_add_epi16
            // subtraction (SSE2): psubw / _mm_sub_epi16
            // multiplication (SSE2): pmullw / _mm_mullo_epi16
            8u;
#elif defined(__ARM_NEON__)
            8u;
#elif defined(__ALTIVEC__)
            8u;
#elif defined(__CUDA_ARCH__)
            // addition: __vadd2
            // subtraction: __vsub2
            // multiplication: -
            2u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<std::uint16_t>
    {
        static constexpr std::size_t value =
#if defined(__AVX512BW__)
            // addition (AVX512BW): vpaddusw / _mm512_mask_adds_epu16
            // subtraction (AVX512BW): vpsubw / _mm512_subs_epu16
            // multiplication: ?
            32u;
#elif defined(__AVX2__)
            // addition (AVX2): vpaddusw / _mm256_adds_epu16
            // subtraction (AVX2): vpsubusw / _mm256_subs_epu16
            // multiplication: ?
            16u;
#elif defined(__SSE2__)
            // addition (SSE2): paddusw / _mm_adds_epu16
            // subtraction (SSE2): psubusw / _mm_subs_epu16
            // multiplication: ?
            8u;
#elif defined(__ARM_NEON__)
            8u;
#elif defined(__ALTIVEC__)
            8u;
#elif defined(__CUDA_ARCH__)
            // addition: __vadd2
            // subtraction: __vsub2
            // multiplication: -
            2u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<std::int32_t>
    {
        static constexpr std::size_t value =
#if defined(__AVX512F__) || defined(__MIC__)
            // addition (AVX512F,KNC): vpaddd / _mm512_mask_add_epi32
            // subtraction (AVX512F,KNC): vpsubd / _mm512_mask_sub_epi32
            // multiplication (AVX512F,KNC): vpmulld / _mm512_mask_mullo_epi32
            16u;
#elif defined(__AVX2__)
            // addition (AVX2): vpaddd / _mm256_add_epi32
            // subtraction (AVX2): vpsubd / _mm256_sub_epi32
            // multiplication (AVX2): vpmulld / _mm256_mullo_epi32
            8u;
#elif defined(__SSE2__)
            // addition (SSE2): paddd / _mm_add_epi32
            // subtraction (SSE2): psubd / _mm_sub_epi32
            // multiplication (SSE4.1): pmulld / _mm_mullo_epi32
            4u;
#elif defined(__ARM_NEON__)
            4u;
#elif defined(__ALTIVEC__)
            4u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<std::uint32_t>
    {
        static constexpr std::size_t value =
#if defined(__AVX512F__) || defined(__MIC__)
            // addition (AVX512F,KNC): vpaddd / _mm512_mask_add_epi32
            // subtraction (AVX512F,KNC): vpsubd / _mm512_mask_sub_epi32
            // multiplication: ?
            16u;
#elif defined(__AVX2__)
            // addition (AVX2): vpaddd / _mm256_add_epi32
            // subtraction (AVX2): vpsubd / _mm256_sub_epi32
            // multiplication: ?
            8u;
#elif defined(__SSE2__)
            // addition (SSE2): paddd / _mm_add_epi32
            // subtraction (SSE2): psubd / _mm_sub_epi32
            // multiplication: ?
            4u;
#elif defined(__ARM_NEON__)
            4u;
#elif defined(__ALTIVEC__)
            4u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<std::int64_t>
    {
        static constexpr std::size_t value =
#if defined(__AVX512F__)
            // addition (AVX512F): vpaddq / _mm512_mask_add_epi64
            // subtraction (AVX512F): vpsubq / _mm512_mask_sub_epi64
            // multiplication (AVX512DQ): vpmullq / _mm512_mask_mullo_epi64
            8u;
#elif defined(__AVX2__)
            // addition (AVX2): vpaddq / _mm256_add_epi64
            // subtraction (AVX2): vpsubq / _mm256_sub_epi64
            // multiplication: -
            4u;
#elif defined(__SSE2__)
            // addition (SSE2): paddq / _mm_add_epi64
            // subtraction (SSE2): psubq / _mm_sub_epi64
            // multiplication: -
            2u;
#elif defined(__ARM_NEON__)
            2u;
#else
            1u;
#endif
    };

    // Number of elements of the given type that can be processed in parallel in a vector register.
    template<>
    struct GetVectorizationSizeElems<std::uint64_t>
    {
        static constexpr std::size_t value =
#if defined(__AVX512F__)
            // addition (AVX512F): vpaddq / _mm512_mask_add_epi64
            // subtraction (AVX512F): vpsubq / _mm512_mask_sub_epi64
            // multiplication: ?
            8u;
#elif defined(__AVX2__)
            // addition (AVX2): vpaddq / _mm256_add_epi64
            // subtraction (AVX2): vpsubq / _mm256_sub_epi64
            // multiplication: ?
            4u;
#elif defined(__SSE2__)
            // addition (SSE2): paddq / _mm_add_epi64
            // subtraction (SSE2): psubq / _mm_sub_epi64
            // multiplication: ?
            2u;
#elif defined(__ARM_NEON__)
            2u;
#else
            1u;
#endif
    };
} // namespace alpaka::core::vectorization
