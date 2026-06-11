/*
 * SPDX-FileCopyrightText: Alexander Grund, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#ifndef ALPAKA_DISABLE_VENDOR_RNG
#    include "pmacc/algorithms/math.hpp"
#    include "pmacc/random/distributions/Normal.hpp"
#    include "pmacc/random/distributions/Uniform.hpp"
#    include "pmacc/random/distributions/misc/MullerBox.hpp"
#    include "pmacc/random/methods/MRG32k3aMin.hpp"
#    include "pmacc/random/methods/XorMin.hpp"
#    include "pmacc/types.hpp"

#    include <type_traits>

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
/* XorMin and MRG32k3aMin uses the alpaka RNG as fallback for CPU accelerators
 * therefore we are not allowed to add a specialization for those RNG methods
 */
#    if (ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)
                //! specialization for XorMin
                template<typename T_Acc>
                struct Normal<double, methods::XorMin<T_Acc>, void> : public MullerBox<double, methods::XorMin<T_Acc>>
                {
                };

                //! specialization for MRG32k3aMin
                template<typename T_Acc>
                struct Normal<double, methods::MRG32k3aMin<T_Acc>, void>
                    : public MullerBox<double, methods::MRG32k3aMin<T_Acc>>
                {
                };
#    endif
            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
#endif
