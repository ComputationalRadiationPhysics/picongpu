/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/standalone/GenericSycl.hpp>

#ifndef ALPAKA_SYCL_BACKEND_ONEAPI
#    define ALPAKA_SYCL_BACKEND_ONEAPI
#endif

#ifndef ALPAKA_SYCL_ONEAPI_GPU
#    define ALPAKA_SYCL_ONEAPI_GPU
#endif

#ifndef ALPAKA_SYCL_TARGET_GPU
#    define ALPAKA_SYCL_TARGET_GPU
#endif
