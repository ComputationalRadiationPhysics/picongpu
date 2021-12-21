/* Copyright 2019 Benjamin Worpitz
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

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <cuda_runtime.h>
#        if !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#        if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
#            include <cuda_runtime_api.h>
#        else
#            if BOOST_COMP_HIP
#                include <hip/math_functions.h>
#            else
#                include <math_functions.hpp>
#            endif
#        endif
#        if !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif
#    endif

#    include <alpaka/math/abs/AbsUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/acos/AcosUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/asin/AsinUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/atan/AtanUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/atan2/Atan2UniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/cbrt/CbrtUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/ceil/CeilUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/cos/CosUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/erf/ErfUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/exp/ExpUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/floor/FloorUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/fmod/FmodUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/log/LogUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/max/MaxUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/min/MinUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/pow/PowUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/remainder/RemainderUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/round/RoundUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/rsqrt/RsqrtUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/sin/SinUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/sincos/SinCosUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/sqrt/SqrtUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/tan/TanUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/trunc/TruncUniformCudaHipBuiltIn.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathUniformCudaHipBuiltIn
            : public AbsUniformCudaHipBuiltIn
            , public AcosUniformCudaHipBuiltIn
            , public AsinUniformCudaHipBuiltIn
            , public AtanUniformCudaHipBuiltIn
            , public Atan2UniformCudaHipBuiltIn
            , public CbrtUniformCudaHipBuiltIn
            , public CeilUniformCudaHipBuiltIn
            , public CosUniformCudaHipBuiltIn
            , public ErfUniformCudaHipBuiltIn
            , public ExpUniformCudaHipBuiltIn
            , public FloorUniformCudaHipBuiltIn
            , public FmodUniformCudaHipBuiltIn
            , public LogUniformCudaHipBuiltIn
            , public MaxUniformCudaHipBuiltIn
            , public MinUniformCudaHipBuiltIn
            , public PowUniformCudaHipBuiltIn
            , public RemainderUniformCudaHipBuiltIn
            , public RoundUniformCudaHipBuiltIn
            , public RsqrtUniformCudaHipBuiltIn
            , public SinUniformCudaHipBuiltIn
            , public SinCosUniformCudaHipBuiltIn
            , public SqrtUniformCudaHipBuiltIn
            , public TanUniformCudaHipBuiltIn
            , public TruncUniformCudaHipBuiltIn
        {
        };
    } // namespace math
} // namespace alpaka

#endif
