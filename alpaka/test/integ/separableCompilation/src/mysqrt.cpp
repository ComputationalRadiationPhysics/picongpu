/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "mysqrt.hpp"

// a square root calculation using simple operations
ALPAKA_FN_HOST_ACC auto mysqrt(double x) -> double
{
    if(x <= 0)
    {
        return 0.0;
    }

    double result = x;

    for(int i = 0; i < 100; ++i)
    {
        if(result <= 0)
        {
            result = 0.1;
        }
        double delta = x - (result * result);
        result = result + 0.5 * delta / result;
    }
    return result;
}
