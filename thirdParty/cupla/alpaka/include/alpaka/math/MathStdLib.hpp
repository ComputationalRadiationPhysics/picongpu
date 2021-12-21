/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/abs/AbsStdLib.hpp>
#include <alpaka/math/acos/AcosStdLib.hpp>
#include <alpaka/math/asin/AsinStdLib.hpp>
#include <alpaka/math/atan/AtanStdLib.hpp>
#include <alpaka/math/atan2/Atan2StdLib.hpp>
#include <alpaka/math/cbrt/CbrtStdLib.hpp>
#include <alpaka/math/ceil/CeilStdLib.hpp>
#include <alpaka/math/cos/CosStdLib.hpp>
#include <alpaka/math/erf/ErfStdLib.hpp>
#include <alpaka/math/exp/ExpStdLib.hpp>
#include <alpaka/math/floor/FloorStdLib.hpp>
#include <alpaka/math/fmod/FmodStdLib.hpp>
#include <alpaka/math/log/LogStdLib.hpp>
#include <alpaka/math/max/MaxStdLib.hpp>
#include <alpaka/math/min/MinStdLib.hpp>
#include <alpaka/math/pow/PowStdLib.hpp>
#include <alpaka/math/remainder/RemainderStdLib.hpp>
#include <alpaka/math/round/RoundStdLib.hpp>
#include <alpaka/math/rsqrt/RsqrtStdLib.hpp>
#include <alpaka/math/sin/SinStdLib.hpp>
#include <alpaka/math/sincos/SinCosStdLib.hpp>
#include <alpaka/math/sqrt/SqrtStdLib.hpp>
#include <alpaka/math/tan/TanStdLib.hpp>
#include <alpaka/math/trunc/TruncStdLib.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathStdLib
            : public AbsStdLib
            , public AcosStdLib
            , public AsinStdLib
            , public AtanStdLib
            , public Atan2StdLib
            , public CbrtStdLib
            , public CeilStdLib
            , public CosStdLib
            , public ErfStdLib
            , public ExpStdLib
            , public FloorStdLib
            , public FmodStdLib
            , public LogStdLib
            , public MaxStdLib
            , public MinStdLib
            , public PowStdLib
            , public RemainderStdLib
            , public RoundStdLib
            , public RsqrtStdLib
            , public SinStdLib
            , public SinCosStdLib
            , public SqrtStdLib
            , public TanStdLib
            , public TruncStdLib
        {
        };
    } // namespace math
} // namespace alpaka
