/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>


#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <array>
#    include <stdexcept>
#    include <string>
#    include <type_traits>

namespace alpaka
{
    namespace uniform_cuda_hip
    {
        namespace detail
        {
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            using Error_t = cudaError;
#    else
            using Error_t = hipError_t;
#    endif
            //-----------------------------------------------------------------------------
            //! CUDA/HIP runtime API error checking with log and exception, ignoring specific error values
            ALPAKA_FN_HOST inline auto rtCheck(
                Error_t const& error,
                char const* desc,
                char const* file,
                int const& line) -> void
            {
                if(error != ALPAKA_API_PREFIX(Success))
                {
                    std::string const sError(
                        std::string(file) + "(" + std::to_string(line) + ") " + std::string(desc) + " : '"
                        + ALPAKA_API_PREFIX(GetErrorName)(error) + "': '"
                        + std::string(ALPAKA_API_PREFIX(GetErrorString)(error)) + "'!");
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    std::cerr << sError << std::endl;
#    endif
                    ALPAKA_DEBUG_BREAK;
                    // reset the last error to allow user side error handling
                    ALPAKA_API_PREFIX(GetLastError)();
                    throw std::runtime_error(sError);
                }
            }
            //-----------------------------------------------------------------------------
            //! CUDA/Hip runtime API error checking with log and exception, ignoring specific error values
            // NOTE: All ignored errors have to be convertible to Error_t.
            template<typename... TErrors>
            ALPAKA_FN_HOST auto rtCheckIgnore(
                Error_t const& error,
                char const* cmd,
                char const* file,
                int const& line,
                TErrors&&... ignoredErrorCodes) -> void
            {
                if(error != ALPAKA_API_PREFIX(Success))
                {
                    std::array<Error_t, sizeof...(ignoredErrorCodes)> const aIgnoredErrorCodes{ignoredErrorCodes...};

                    // If the error code is not one of the ignored ones.
                    if(std::find(aIgnoredErrorCodes.cbegin(), aIgnoredErrorCodes.cend(), error)
                       == aIgnoredErrorCodes.cend())
                    {
                        rtCheck(error, ("'" + std::string(cmd) + "' returned error ").c_str(), file, line);
                    }
                    else
                    {
                        // reset the last error to avoid propagation to the next CUDA/HIP API call
                        ALPAKA_API_PREFIX(GetLastError)();
                    }
                }
            }
            //-----------------------------------------------------------------------------
            //! CUDA runtime API last error checking with log and exception.
            ALPAKA_FN_HOST inline auto rtCheckLastError(char const* desc, char const* file, int const& line) -> void
            {
                Error_t const error(ALPAKA_API_PREFIX(GetLastError)());
                rtCheck(error, desc, file, line);
            }
        } // namespace detail
    } // namespace uniform_cuda_hip
} // namespace alpaka

#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
  //-----------------------------------------------------------------------------
//! CUDA runtime error checking with log and exception, ignoring specific error values
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                     \
            ::alpaka::uniform_cuda_hip::detail::rtCheckLastError(                                                     \
                "'" #cmd "' A previous API call (not this one) set the error ",                                       \
                __FILE__,                                                                                             \
                __LINE__);                                                                                            \
            ::alpaka::uniform_cuda_hip::detail::rtCheckIgnore(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#    else
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#        endif
  //-----------------------------------------------------------------------------
//! CUDA runtime error checking with log and exception, ignoring specific error values
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                     \
            ::alpaka::uniform_cuda_hip::detail::rtCheckLastError(                                                     \
                "'" #cmd "' A previous API call (not this one) set the error ",                                       \
                __FILE__,                                                                                             \
                __LINE__);                                                                                            \
            ::alpaka::uniform_cuda_hip::detail::rtCheckIgnore(cmd, #cmd, __FILE__, __LINE__, ##__VA_ARGS__)
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic pop
#        endif
#    endif

//-----------------------------------------------------------------------------
//! CUDA runtime error checking with log and exception.
#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cmd) ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd)

#endif
