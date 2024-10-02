/* Copyright 2023 Jeffrey Kelling, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/shared/dyn/BlockSharedDynMemberAllocKiB.hpp"
#include "alpaka/block/shared/dyn/Traits.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Vectorize.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

namespace alpaka
{
    namespace detail
    {
        //! "namespace" for static constexpr members that should be in BlockSharedMemDynMember
        //! but cannot be because having a static const member breaks GCC 10
        //! OpenMP target: type not mappable.
        template<std::size_t TStaticAllocKiB>
        struct BlockSharedMemDynMemberStatic
        {
            //! Storage size in bytes
            static constexpr std::uint32_t staticAllocBytes = static_cast<std::uint32_t>(TStaticAllocKiB << 10u);
        };
    } // namespace detail

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4324) // warning C4324: structure was padded due to alignment specifier
#endif
    //! Dynamic block shared memory provider using fixed-size
    //! member array to allocate memory on the stack or in shared
    //! memory.
    template<std::size_t TStaticAllocKiB = BlockSharedDynMemberAllocKiB>
    class alignas(core::vectorization::defaultAlignment) BlockSharedMemDynMember
        : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynMember<TStaticAllocKiB>>
    {
    public:
        BlockSharedMemDynMember(std::size_t sizeBytes) : m_dynPitch(getPitch(sizeBytes))
        {
            ALPAKA_ASSERT_ACC(static_cast<std::uint32_t>(sizeBytes) <= staticAllocBytes());
        }

        auto dynMemBegin() const -> uint8_t*
        {
            return std::data(m_mem);
        }

        /*! \return the pointer to the begin of data after the portion allocated as dynamical shared memory.
         */
        auto staticMemBegin() const -> uint8_t*
        {
            return std::data(m_mem) + m_dynPitch;
        }

        /*! \return the remaining capacity for static block shared memory,
                    returns a 32-bit type for register efficiency on GPUs
            */
        auto staticMemCapacity() const -> std::uint32_t
        {
            return staticAllocBytes() - m_dynPitch;
        }

        //! \return size of statically allocated memory available for both
        //!         dynamic and static shared memory. Value is of a 32-bit type
        //!         for register efficiency on GPUs
        static constexpr auto staticAllocBytes() -> std::uint32_t
        {
            return detail::BlockSharedMemDynMemberStatic<TStaticAllocKiB>::staticAllocBytes;
        }

    private:
        static auto getPitch(std::size_t sizeBytes) -> std::uint32_t
        {
            constexpr auto alignment = core::vectorization::defaultAlignment;
            return static_cast<std::uint32_t>((sizeBytes / alignment + (sizeBytes % alignment > 0u)) * alignment);
        }

        mutable std::array<uint8_t, detail::BlockSharedMemDynMemberStatic<TStaticAllocKiB>::staticAllocBytes> m_mem;
        std::uint32_t m_dynPitch;
    };
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif

    namespace trait
    {
        template<typename T, std::size_t TStaticAllocKiB>
        struct GetDynSharedMem<T, BlockSharedMemDynMember<TStaticAllocKiB>>
        {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
            static auto getMem(BlockSharedMemDynMember<TStaticAllocKiB> const& mem) -> T*
            {
                static_assert(
                    core::vectorization::defaultAlignment >= alignof(T),
                    "Unable to get block shared dynamic memory for types with alignment higher than "
                    "defaultAlignment!");
                return reinterpret_cast<T*>(mem.dynMemBegin());
            }
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        };
    } // namespace trait
} // namespace alpaka
