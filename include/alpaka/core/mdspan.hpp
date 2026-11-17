/* Copyright 2026 Axel Hübl, Benjamin Worpitz, Matthias Werner, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber,
 *                Aurora Perego, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

// no pragma once, because the header defines alias namespaces

#ifdef ALPAKA_USE_MDSPAN
#    ifdef ALPAKA_HAS_STD_MDSPAN
//       mdspan from the standard library
#        include <mdspan>
#        include <version>

namespace alpaka::experimental
{
    // Import C++23 mdspan into alpaka::experimental namespace.
    // See https://wg21.link/P0009R18 .
    using ::std::default_accessor;
    using ::std::dextents;
    using ::std::extents;
    using ::std::layout_left;
    using ::std::layout_right;
    using ::std::layout_stride;
    using ::std::mdspan;
#        ifdef __cpp_lib_submdspan
    // Import C++26 submdspan into alpaka::experimental namespace.
    // See https://wg21.link/P2630R4 .
    using ::std::full_extent;
    using ::std::submdspan;
#        endif
} // namespace alpaka::experimental

#    else

#        ifdef ALPAKA_ACC_SYCL_ENABLED
//           do not expose the macro definition of printf
#            pragma push_macro("printf")
#            ifdef printf
#                undef printf
#            endif
#        endif // ALPAKA_ACC_SYCL_ENABLED

#        if defined(ALPAKA_ACC_SYCL_ENABLED) && (defined(ALPAKA_SYCL_ONEAPI_FPGA) || defined(ALPAKA_SYCL_TARGET_FPGA))
//           the fpga compiler does not handle well the [[no_unique_address]] attribute, resulting in:
//               GEP has !intel-tbaa annotation but its "shape" is unexpected!
//               /opt/intel/oneapi/compiler/2025.0/bin/compiler/llvm-link: error: linked module is broken!
//               icpx: error: sycl-link command failed with exit code 1 (use -v to see invocation)
#            include <experimental/__p0009_bits/config.hpp>
#            undef MDSPAN_IMPL_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS
#            undef MDSPAN_IMPL_NO_UNIQUE_ADDRESS
#            define MDSPAN_IMPL_NO_UNIQUE_ADDRESS
#        endif

//       mdspan from the Kokkos reference implementation
#        define MDSPAN_IMPL_STANDARD_NAMESPACE alpaka::experimental
#        include <experimental/mdspan>

#        ifdef ALPAKA_ACC_SYCL_ENABLED
//           restore the macro definition of printf
#            pragma pop_macro("printf")
#        endif // ALPAKA_ACC_SYCL_ENABLED
#    endif
#endif
