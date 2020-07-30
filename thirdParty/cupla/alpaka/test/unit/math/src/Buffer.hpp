/** Copyright 2019 Jakob Krude, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "Defines.hpp"

#include <alpaka/test/acc/TestAccs.hpp>

#include <ostream>

namespace alpaka {
namespace test {
namespace unit {
namespace math {

//! Provides alpaka-style buffer with arguments' data.
//! TData can be a plain value or a complex data-structure.
//! The operator() is overloaded and returns the value from the correct Buffer,
//! either from the host (index) or device buffer (index, acc).
//! Index out of range errors are not checked.
//! @brief Encapsulates buffer initialisation and communication with Device.
//! @tparam TAcc Used accelerator, not interchangeable
//! @tparam TData The Data-type, only restricted by the alpaka-interface.
//! @tparam Tcapacity The size of the buffer.
template<
    typename TAcc,
    typename TData,
    size_t Tcapacity
>
struct Buffer
{
    using value_type = TData;
    static constexpr size_t capacity = Tcapacity;
    using Dim = typename alpaka::dim::traits::DimType<TAcc>::type;
    using Idx = typename alpaka::idx::traits::IdxType<TAcc>::type;

    // Defines using's for alpaka-buffer.
    using DevAcc = alpaka::dev::Dev< TAcc >;
    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf< DevHost >;

    using BufHost = alpaka::mem::buf::Buf<
        DevHost,
        TData,
        Dim,
        Idx
    >;
    using BufAcc = alpaka::mem::buf::Buf<
        DevAcc,
        TData,
        Dim,
        Idx
    >;

    DevHost devHost;

    BufHost hostBuffer;
    BufAcc devBuffer;

    // Native pointer to access buffer.
    TData * const pHostBuffer;
    TData * const pDevBuffer;


    // This constructor cant be used,
    // because BufHost and BufAcc need to be initialised.
    Buffer( ) = delete;

    // Constructor needs to initialize all Buffer.
    Buffer(const DevAcc & devAcc)
      :
        devHost{ alpaka::pltf::getDevByIdx< PltfHost >( 0u ) },
        hostBuffer
        {
            alpaka::mem::buf::alloc<TData, Idx>(devHost, Tcapacity)
        },
        devBuffer
        {
            alpaka::mem::buf::alloc<TData, Idx>(devAcc, Tcapacity)
        },
        pHostBuffer{ alpaka::mem::view::getPtrNative( hostBuffer ) },
        pDevBuffer{ alpaka::mem::view::getPtrNative( devBuffer ) }
    {}

    // Copy Host -> Acc.
    template< typename Queue >
    auto copyToDevice( Queue queue ) -> void
    {
        alpaka::mem::view::copy(
            queue,
            devBuffer,
            hostBuffer,
            Tcapacity
        );
    }

    // Copy Acc -> Host.
    template< typename Queue >
    auto copyFromDevice( Queue queue ) -> void
    {
        alpaka::mem::view::copy(
            queue,
            hostBuffer,
            devBuffer,
            Tcapacity
        );
    }

    ALPAKA_FN_ACC
    auto operator()(
        size_t idx,
        TAcc const & acc ) const -> TData&
    {
        alpaka::ignore_unused(acc);
        return pDevBuffer[idx];
    }

    ALPAKA_FN_HOST
    auto operator()(
        size_t idx ) const -> TData&
    {
        return pHostBuffer[idx];
    }

    ALPAKA_FN_HOST
    friend std::ostream & operator<<(
        std::ostream & os,
        const Buffer & buffer
    )
    {
        os << "capacity: " << capacity
           << "\n";
        for( size_t i = 0; i < capacity; ++i )
        {
            os << i
               << ": " << buffer.pHostBuffer[i]
               << "\n";
        }
        return os;
    }
};

} // math
} // unit
} // test
} // alpaka
