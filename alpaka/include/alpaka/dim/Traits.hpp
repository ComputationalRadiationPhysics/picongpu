/* Copyright 2020 Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    //! The dimension trait.
    namespace trait
    {
        //! The dimension getter type trait.
        template<typename T, typename TSfinae = void>
        struct DimType;
    } // namespace trait

    //! The dimension type trait alias template to remove the ::type.
    template<typename T>
    using Dim = typename trait::DimType<T>::type;
} // namespace alpaka
