/* Copyright 2022 Bernhard Manfred Gruber

 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#pragma once

#include <alpaka/core/Utility.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/meta/DependentFalseType.hpp>
#include <alpaka/meta/TypeListOps.hpp>

#include <tuple>

namespace alpaka::experimental
{
    //! Access tag type indicating read-only access.
    struct ReadAccess
    {
    };

    //! Access tag type indicating write-only access.
    struct WriteAccess
    {
    };

    //! Access tag type indicating read-write access.
    struct ReadWriteAccess
    {
    };

    //! An accessor is an abstraction for accessing memory objects such as views and buffers.
    //! @tparam TMemoryHandle A handle to a memory object.
    //! @tparam TElem The type of the element stored by the memory object. Values and references to this type are
    //! returned on access.
    //! @tparam TBufferIdx The integral type used for indexing and index computations.
    //! @tparam TDim The dimensionality of the accessed data.
    //! @tparam TAccessModes Either a single access tag type or a `std::tuple` containing multiple access tag
    //! types.
    template<typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim, typename TAccessModes>
    struct Accessor;

    namespace trait
    {
        //! The customization point for how to build an accessor for a given memory object.
        template<typename TMemoryObject, typename SFINAE = void>
        struct BuildAccessor
        {
            template<typename... TAccessModes, typename TMemoryObjectForwardRef>
            ALPAKA_FN_HOST static auto buildAccessor(TMemoryObjectForwardRef&&)
            {
                static_assert(
                    meta::DependentFalseType<TMemoryObject>::value,
                    "BuildAccessor<TMemoryObject> is not specialized for your TMemoryObject.");
            }
        };
    } // namespace trait

    namespace internal
    {
        template<typename AccessorOrBuffer>
        struct MemoryHandle
        {
        };

        template<typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim, typename TAccessModes>
        struct MemoryHandle<Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, TAccessModes>>
        {
            using type = TMemoryHandle;
        };
    } // namespace internal

    /// Get the memory handle type of the given accessor or buffer type.
    template<typename Accessor>
    using MemoryHandle = typename internal::MemoryHandle<Accessor>::type;

    namespace internal
    {
        template<typename T>
        struct IsAccessor : std::false_type
        {
        };

        template<typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t Dim, typename TAccessModes>
        struct IsAccessor<Accessor<TMemoryHandle, TElem, TBufferIdx, Dim, TAccessModes>> : std::true_type
        {
        };
    } // namespace internal

    //! Creates an accessor for the given memory object using the specified access modes. Memory objects are e.g.
    //! alpaka views and buffers.
    template<
        typename... TAccessModes,
        typename TMemoryObject,
        typename = std::enable_if_t<!internal::IsAccessor<std::decay_t<TMemoryObject>>::value>>
    ALPAKA_FN_HOST auto accessWith(TMemoryObject&& memoryObject)
    {
        return trait::BuildAccessor<std::decay_t<TMemoryObject>>::template buildAccessor<TAccessModes...>(
            memoryObject);
    }

    //! Constrains an existing accessor with multiple access modes to the specified access modes.
    // TODO: currently only allows constraining down to 1 access mode
    template<
        typename TNewAccessMode,
        typename TMemoryHandle,
        typename TElem,
        typename TBufferIdx,
        std::size_t TDim,
        typename... TPrevAccessModes>
    ALPAKA_FN_HOST auto accessWith(
        Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, std::tuple<TPrevAccessModes...>> const& acc)
    {
        static_assert(
            meta::Contains<std::tuple<TPrevAccessModes...>, TNewAccessMode>::value,
            "The accessed accessor must already contain the requested access mode");
        return Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, TNewAccessMode>{acc};
    }

    //! Constrains an existing accessor to the specified access modes.
    // constraining accessor to the same access mode again just passes through
    template<typename TNewAccessMode, typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim>
    ALPAKA_FN_HOST auto accessWith(Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, TNewAccessMode> const& acc)
    {
        return acc;
    }

    //! Creates a read-write accessor for the given memory object (view, buffer, ...) or accessor.
    template<typename TMemoryObjectOrAccessor>
    ALPAKA_FN_HOST auto access(TMemoryObjectOrAccessor&& viewOrAccessor)
    {
        return accessWith<ReadWriteAccess>(std::forward<TMemoryObjectOrAccessor>(viewOrAccessor));
    }

    //! Creates a read-only accessor for the given memory object (view, buffer, ...) or accessor.
    template<typename TMemoryObjectOrAccessor>
    ALPAKA_FN_HOST auto readAccess(TMemoryObjectOrAccessor&& viewOrAccessor)
    {
        return accessWith<ReadAccess>(std::forward<TMemoryObjectOrAccessor>(viewOrAccessor));
    }

    //! Creates a write-only accessor for the given memory object (view, buffer, ...) or accessor.
    template<typename TMemoryObjectOrAccessor>
    ALPAKA_FN_HOST auto writeAccess(TMemoryObjectOrAccessor&& viewOrAccessor)
    {
        return accessWith<WriteAccess>(std::forward<TMemoryObjectOrAccessor>(viewOrAccessor));
    }

    //! An alias for an accessor accessing a buffer on the given accelerator.
    template<
        typename TAcc,
        typename TElem,
        std::size_t TDim,
        typename TAccessModes = ReadWriteAccess,
        typename TIdx = Idx<TAcc>>
    using BufferAccessor = Accessor<
        MemoryHandle<decltype(accessWith<TAccessModes>(core::declval<Buf<TAcc, TElem, DimInt<TDim>, TIdx>>()))>,
        TElem,
        TIdx,
        TDim,
        TAccessModes>;
} // namespace alpaka::experimental
