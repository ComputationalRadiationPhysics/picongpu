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

#include <alpaka/math/abs/AbsStl.hpp>
#include <alpaka/math/acos/AcosStl.hpp>
#include <alpaka/math/asin/AsinStl.hpp>
#include <alpaka/math/atan/AtanStl.hpp>
#include <alpaka/math/atan2/Atan2Stl.hpp>
#include <alpaka/math/cbrt/CbrtStl.hpp>
#include <alpaka/math/ceil/CeilStl.hpp>
#include <alpaka/math/cos/CosStl.hpp>
#include <alpaka/math/erf/ErfStl.hpp>
#include <alpaka/math/exp/ExpStl.hpp>
#include <alpaka/math/floor/FloorStl.hpp>
#include <alpaka/math/fmod/FmodStl.hpp>
#include <alpaka/math/log/LogStl.hpp>
#include <alpaka/math/max/MaxStl.hpp>
#include <alpaka/math/min/MinStl.hpp>
#include <alpaka/math/pow/PowStl.hpp>
#include <alpaka/math/remainder/RemainderStl.hpp>
#include <alpaka/math/round/RoundStl.hpp>
#include <alpaka/math/rsqrt/RsqrtStl.hpp>
#include <alpaka/math/sin/SinStl.hpp>
#include <alpaka/math/sqrt/SqrtStl.hpp>
#include <alpaka/math/tan/TanStl.hpp>
#include <alpaka/math/trunc/TruncStl.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathStl :
            public AbsStl,
            public AcosStl,
            public AsinStl,
            public AtanStl,
            public Atan2Stl,
            public CbrtStl,
            public CeilStl,
            public CosStl,
            public ErfStl,
            public ExpStl,
            public FloorStl,
            public FmodStl,
            public LogStl,
            public MaxStl,
            public MinStl,
            public PowStl,
            public RemainderStl,
            public RoundStl,
            public RsqrtStl,
            public SinStl,
            public SqrtStl,
            public TanStl,
            public TruncStl
        {};
    }
}
