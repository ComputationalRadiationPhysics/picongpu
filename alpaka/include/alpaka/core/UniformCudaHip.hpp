/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Andrea Bocci, Bernhard
 * Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::uniform_cuda_hip::detail
{
    //! CUDA/HIP runtime API error checking with log and exception, ignoring specific error values
    template<typename TApi, bool TThrow>
    ALPAKA_FN_HOST inline void rtCheck(
        typename TApi::Error_t const& error,
        char const* desc,
        char const* file,
        int const& line) noexcept(!TThrow)
    {
        if(error != TApi::success)
        {
            auto const sError = std::string{
                std::string(file) + "(" + std::to_string(line) + ") " + std::string(desc) + " : '"
                + TApi::getErrorName(error) + "': '" + std::string(TApi::getErrorString(error)) + "'!"};

            if constexpr(!TThrow || ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL)
                std::cerr << sError << std::endl;

            ALPAKA_DEBUG_BREAK;
            // reset the last error to allow user side error handling. Using std::ignore to discard unneeded
            // return values is suggested by the C++ core guidelines.
            std::ignore = TApi::getLastError();

            if constexpr(TThrow)
                throw std::runtime_error(sError);
        }
    }

    //! CUDA/HIP runtime API error checking with log and exception, ignoring specific error values
    template<typename TApi, bool TThrow>
    ALPAKA_FN_HOST inline void rtCheckIgnore(
        typename TApi::Error_t const& error,
        char const* cmd,
        char const* file,
        int const& line,
        std::initializer_list<typename TApi::Error_t> ignoredErrorCodes) noexcept(!TThrow)
    {
        if(error != TApi::success)
        {
            // If the error code is not one of the ignored ones.
            if(std::find(std::cbegin(ignoredErrorCodes), std::cend(ignoredErrorCodes), error)
               == std::cend(ignoredErrorCodes))
            {
                using namespace std::literals;
                rtCheck<TApi, TThrow>(error, ("'"s + std::string(cmd) + "' returned error "s).c_str(), file, line);
            }
            else
            {
                // reset the last error to avoid propagation to the next CUDA/HIP API call. Using std::ignore
                // to discard unneeded return values is recommended by the C++ core guidelines.
                std::ignore = TApi::getLastError();
            }
        }
    }

    //! CUDA/HIP runtime API last error checking with log and exception.
    template<typename TApi, bool TThrow>
    ALPAKA_FN_HOST inline void rtCheckLastError(char const* desc, char const* file, int const& line) noexcept(!TThrow)
    {
        typename TApi::Error_t const error(TApi::getLastError());
        rtCheck<TApi, TThrow>(error, desc, file, line);
    }
} // namespace alpaka::uniform_cuda_hip::detail

#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, throw, ...)                                                    \
        do                                                                                                            \
        {                                                                                                             \
            ::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, throw>(                                        \
                "'" #cmd "' A previous API call (not this one) set the error ",                                       \
                __FILE__,                                                                                             \
                __LINE__);                                                                                            \
            ::alpaka::uniform_cuda_hip::detail::rtCheckIgnore<TApi, throw>(                                           \
                cmd,                                                                                                  \
                #cmd,                                                                                                 \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                {__VA_ARGS__});                                                                                       \
        } while(0)

//! CUDA/HIP runtime error checking with log and exception, ignoring specific error values
#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                         \
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, true, __VA_ARGS__)

//! CUDA/HIP runtime error checking with log and exception.
#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cmd) ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, true, )

//! CUDA/HIP runtime error checking with log and exception, ignoring specific error values
#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                                \
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, false, __VA_ARGS__)

//! CUDA/HIP runtime error checking with log.
#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(cmd) ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IMPL(cmd, false, )
#endif
