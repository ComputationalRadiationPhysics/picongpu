/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <version>

int main(int argc, char** argv)
{
#if defined(__cpp_lib_atomic_ref)
    return 0;
#else
    return 1;
#endif
}
