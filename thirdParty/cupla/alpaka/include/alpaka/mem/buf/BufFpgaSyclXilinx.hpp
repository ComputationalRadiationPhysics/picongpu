/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#    include <alpaka/mem/buf/BufGenericSycl.hpp>

namespace alpaka::experimental
{
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    using BufFpgaSyclXilinx = BufGenericSycl<TElem, TDim, TIdx, DevFpgaSyclXilinx>;
} // namespace alpaka::experimental

#endif
