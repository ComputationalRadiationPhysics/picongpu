/* Copyright 2019 Benjamin Worpitz
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
    //-----------------------------------------------------------------------------
    //! The dimension traits.
    namespace traits
    {
        //#############################################################################
        //! The dimension getter type trait.
        template<typename T, typename TSfinae = void>
        struct DimType;
    } // namespace traits

    //#############################################################################
    //! The dimension type trait alias template to remove the ::type.
    template<typename T>
    using Dim = typename traits::DimType<T>::type;
} // namespace alpaka
