/* Copyright 2022 Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

namespace alpaka
{
    //! The common trait.
    namespace trait
    {
        //! The native handle trait.
        template<typename TImpl, typename TSfinae = void>
        struct NativeHandle
        {
            static auto getNativeHandle(TImpl const&)
            {
                static_assert(!sizeof(TImpl), "This type does not have a native handle!");
                return 0;
            }
        };
    } // namespace trait

    //! Get the native handle of the alpaka object.
    //! It will return the alpaka object handle if there is any, otherwise it generates a compile time error.
    template<typename TImpl>
    ALPAKA_FN_HOST auto getNativeHandle(TImpl const& impl)
    {
        return trait::NativeHandle<TImpl>::getNativeHandle(impl);
    }

    //! Alias to the type of the native handle.
    template<typename TImpl>
    using NativeHandle = decltype(getNativeHandle(std::declval<TImpl>()));
} // namespace alpaka
