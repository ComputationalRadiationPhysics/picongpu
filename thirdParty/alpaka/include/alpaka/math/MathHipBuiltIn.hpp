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

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/math/abs/AbsHipBuiltIn.hpp>
#include <alpaka/math/acos/AcosHipBuiltIn.hpp>
#include <alpaka/math/asin/AsinHipBuiltIn.hpp>
#include <alpaka/math/atan/AtanHipBuiltIn.hpp>
#include <alpaka/math/atan2/Atan2HipBuiltIn.hpp>
#include <alpaka/math/cbrt/CbrtHipBuiltIn.hpp>
#include <alpaka/math/ceil/CeilHipBuiltIn.hpp>
#include <alpaka/math/cos/CosHipBuiltIn.hpp>
#include <alpaka/math/erf/ErfHipBuiltIn.hpp>
#include <alpaka/math/exp/ExpHipBuiltIn.hpp>
#include <alpaka/math/floor/FloorHipBuiltIn.hpp>
#include <alpaka/math/fmod/FmodHipBuiltIn.hpp>
#include <alpaka/math/log/LogHipBuiltIn.hpp>
#include <alpaka/math/max/MaxHipBuiltIn.hpp>
#include <alpaka/math/min/MinHipBuiltIn.hpp>
#include <alpaka/math/pow/PowHipBuiltIn.hpp>
#include <alpaka/math/remainder/RemainderHipBuiltIn.hpp>
#include <alpaka/math/round/RoundHipBuiltIn.hpp>
#include <alpaka/math/rsqrt/RsqrtHipBuiltIn.hpp>
#include <alpaka/math/sin/SinHipBuiltIn.hpp>
#include <alpaka/math/sqrt/SqrtHipBuiltIn.hpp>
#include <alpaka/math/tan/TanHipBuiltIn.hpp>
#include <alpaka/math/trunc/TruncHipBuiltIn.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathHipBuiltIn :
            public AbsHipBuiltIn,
            public AcosHipBuiltIn,
            public AsinHipBuiltIn,
            public AtanHipBuiltIn,
            public Atan2HipBuiltIn,
            public CbrtHipBuiltIn,
            public CeilHipBuiltIn,
            public CosHipBuiltIn,
            public ErfHipBuiltIn,
            public ExpHipBuiltIn,
            public FloorHipBuiltIn,
            public FmodHipBuiltIn,
            public LogHipBuiltIn,
            public MaxHipBuiltIn,
            public MinHipBuiltIn,
            public PowHipBuiltIn,
            public RemainderHipBuiltIn,
            public RoundHipBuiltIn,
            public RsqrtHipBuiltIn,
            public SinHipBuiltIn,
            public SqrtHipBuiltIn,
            public TanHipBuiltIn,
            public TruncHipBuiltIn
        {};
    }
}

#endif
