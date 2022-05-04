/* Copyright 2022 Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <atomic>
#include <cstdlib>

#if !defined(__cpp_lib_atomic_ref)
#    error "std::atomic_ref<T> not supported!"
#endif

auto main() -> int
{
    return EXIT_SUCCESS;
}
