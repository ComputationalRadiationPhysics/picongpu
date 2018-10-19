/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/abs/AbsCudaBuiltIn.hpp>
#include <alpaka/math/acos/AcosCudaBuiltIn.hpp>
#include <alpaka/math/asin/AsinCudaBuiltIn.hpp>
#include <alpaka/math/atan/AtanCudaBuiltIn.hpp>
#include <alpaka/math/atan2/Atan2CudaBuiltIn.hpp>
#include <alpaka/math/cbrt/CbrtCudaBuiltIn.hpp>
#include <alpaka/math/ceil/CeilCudaBuiltIn.hpp>
#include <alpaka/math/cos/CosCudaBuiltIn.hpp>
#include <alpaka/math/erf/ErfCudaBuiltIn.hpp>
#include <alpaka/math/exp/ExpCudaBuiltIn.hpp>
#include <alpaka/math/floor/FloorCudaBuiltIn.hpp>
#include <alpaka/math/fmod/FmodCudaBuiltIn.hpp>
#include <alpaka/math/log/LogCudaBuiltIn.hpp>
#include <alpaka/math/max/MaxCudaBuiltIn.hpp>
#include <alpaka/math/min/MinCudaBuiltIn.hpp>
#include <alpaka/math/pow/PowCudaBuiltIn.hpp>
#include <alpaka/math/remainder/RemainderCudaBuiltIn.hpp>
#include <alpaka/math/round/RoundCudaBuiltIn.hpp>
#include <alpaka/math/rsqrt/RsqrtCudaBuiltIn.hpp>
#include <alpaka/math/sin/SinCudaBuiltIn.hpp>
#include <alpaka/math/sqrt/SqrtCudaBuiltIn.hpp>
#include <alpaka/math/tan/TanCudaBuiltIn.hpp>
#include <alpaka/math/trunc/TruncCudaBuiltIn.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathCudaBuiltIn :
            public AbsCudaBuiltIn,
            public AcosCudaBuiltIn,
            public AsinCudaBuiltIn,
            public AtanCudaBuiltIn,
            public Atan2CudaBuiltIn,
            public CbrtCudaBuiltIn,
            public CeilCudaBuiltIn,
            public CosCudaBuiltIn,
            public ErfCudaBuiltIn,
            public ExpCudaBuiltIn,
            public FloorCudaBuiltIn,
            public FmodCudaBuiltIn,
            public LogCudaBuiltIn,
            public MaxCudaBuiltIn,
            public MinCudaBuiltIn,
            public PowCudaBuiltIn,
            public RemainderCudaBuiltIn,
            public RoundCudaBuiltIn,
            public RsqrtCudaBuiltIn,
            public SinCudaBuiltIn,
            public SqrtCudaBuiltIn,
            public TanCudaBuiltIn,
            public TruncCudaBuiltIn
        {};
    }
}

#endif
