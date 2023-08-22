/* Copyright 2020 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
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
