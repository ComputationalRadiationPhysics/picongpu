/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Defines.hpp"

#include <alpaka/test/acc/TestAccs.hpp>

#include <ostream>

namespace mathtest
{
    //! Provides alpaka-style buffer with arguments' data.
    //! TData can be a plain value or a complex data-structure.
    //! The operator() is overloaded and returns the value from the correct Buffer,
    //! either from the host (index) or device buffer (index, acc).
    //! Index out of range errors are not checked.
    //! @brief Encapsulates buffer initialisation and communication with Device.
    //! @tparam TAcc Used accelerator, not interchangeable
    //! @tparam TData The Data-type, only restricted by the alpaka-interface.
    //! @tparam Tcapacity The size of the buffer.
    template<typename TAcc, typename TData, size_t Tcapacity>
    struct Buffer
    {
        using value_type = TData;
        static constexpr size_t capacity = Tcapacity;
        using Dim = typename alpaka::trait::DimType<TAcc>::type;
        using Idx = typename alpaka::trait::IdxType<TAcc>::type;

        // Defines using's for alpaka-buffer.
        using DevHost = alpaka::DevCpu;
        using PlatformHost = alpaka::Platform<DevHost>;
        using BufHost = alpaka::Buf<DevHost, TData, Dim, Idx>;

        using DevAcc = alpaka::Dev<TAcc>;
        using PlatformAcc = alpaka::Platform<DevAcc>;
        using BufAcc = alpaka::Buf<DevAcc, TData, Dim, Idx>;

        DevHost devHost;

        BufHost hostBuffer;
        BufAcc devBuffer;

        // Native pointer to access buffer.
        TData* const pHostBuffer;
        TData* const pDevBuffer;

        // This constructor cant be used,
        // because BufHost and BufAcc need to be initialised.
        Buffer() = delete;

        // Constructor needs to initialize all Buffer.
        Buffer(DevAcc const& devAcc, PlatformHost const& platformHost, PlatformAcc const& platformAcc)
            : devHost{alpaka::getDevByIdx(platformHost, 0)}
            , hostBuffer{alpaka::allocMappedBufIfSupported<TData, Idx>(devHost, platformAcc, Tcapacity)}
            , devBuffer{alpaka::allocBuf<TData, Idx>(devAcc, Tcapacity)}
            , pHostBuffer{std::data(hostBuffer)}
            , pDevBuffer{std::data(devBuffer)}
        {
        }

        // Copy Host -> Acc.
        template<typename Queue>
        auto copyToDevice(Queue queue) -> void
        {
            alpaka::memcpy(queue, devBuffer, hostBuffer);
        }

        // Copy Acc -> Host.
        template<typename Queue>
        auto copyFromDevice(Queue queue) -> void
        {
            alpaka::memcpy(queue, hostBuffer, devBuffer);
        }

        ALPAKA_FN_ACC auto operator()(size_t idx, TAcc const& /* acc */) const -> TData&
        {
            return pDevBuffer[idx];
        }

        ALPAKA_FN_HOST auto operator()(size_t idx) const -> TData&
        {
            return pHostBuffer[idx];
        }

        ALPAKA_FN_HOST friend auto operator<<(std::ostream& os, Buffer const& buffer) -> std::ostream&
        {
            os << "capacity: " << capacity << "\n";
            for(size_t i = 0; i < capacity; ++i)
            {
                os << i << ": " << buffer.pHostBuffer[i] << "\n";
            }
            return os;
        }
    };
} // namespace mathtest
