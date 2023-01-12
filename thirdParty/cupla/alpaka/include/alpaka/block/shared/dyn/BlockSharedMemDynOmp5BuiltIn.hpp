/* Copyright 2022 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if(defined(ALPAKA_ACC_ANY_BT_OMP5_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED))

#    include <alpaka/block/shared/dyn/BlockSharedDynMemberAllocKiB.hpp>
#    include <alpaka/block/shared/dyn/Traits.hpp>
#    include <alpaka/core/Vectorize.hpp>

#    include <omp.h>

#    include <array>
#    include <cstddef>

namespace alpaka
{
    //! The OpenMP 5.0 block dynamic shared memory allocator.
    //!
    //! This class serves as a base for future strategies to obtain a dynamic
    //! shared memory allocation using OpenMP features. It does not assume to
    //! own the memory pointed to by `m_mem`, but derived classes may.
    class BlockSharedMemDynOmp5BuiltIn
        : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynOmp5BuiltIn>
    {
    public:
        BlockSharedMemDynOmp5BuiltIn(std::byte* mem) : m_mem(mem)
        {
        }

        BlockSharedMemDynOmp5BuiltIn(BlockSharedMemDynOmp5BuiltIn const&) = delete;
        auto operator=(BlockSharedMemDynOmp5BuiltIn const&) -> BlockSharedMemDynOmp5BuiltIn& = delete;
        BlockSharedMemDynOmp5BuiltIn(BlockSharedMemDynOmp5BuiltIn&&) = delete;
        auto operator=(BlockSharedMemDynOmp5BuiltIn&&) -> BlockSharedMemDynOmp5BuiltIn& = delete;

        [[nodiscard]] auto mem() const -> std::byte*
        {
            return m_mem;
        }

        //! \return size of statically allocated memory available for both
        //!         dynamic and static shared memory. This value is actually
        //!         unknown, will just return the statically defined limit for
        //!         BlockSharedMemDynMember.
        static constexpr auto staticAllocBytes() -> std::uint32_t
        {
            return BlockSharedDynMemberAllocKiB << 10u;
        }

    protected:
        std::byte* m_mem = nullptr;
    };

#    if _OPENMP >= 201811 // omp_alloc requires OpenMP 5.0
    //! The OpenMP 5.0 block dynamic shared memory allocator based on `omp_alloc()`
    class BlockSharedMemDynOmp5BuiltInOmpAlloc : public BlockSharedMemDynOmp5BuiltIn
    {
    public:
        BlockSharedMemDynOmp5BuiltInOmpAlloc([[maybe_unused]] std::size_t bytes)
            : BlockSharedMemDynOmp5BuiltIn(reinterpret_cast<std::byte*>(omp_alloc(bytes, omp_pteam_mem_alloc)))
        {
        }

        ~BlockSharedMemDynOmp5BuiltInOmpAlloc()
        {
            if(m_mem)
                omp_free(m_mem, omp_pteam_mem_alloc);
        }

    private:
    };
#    endif

#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        pragma warning(push)
#        pragma warning(disable : 4324) // warning C4324: structure was padded due to alignment specifier
#    endif
    //! The OpenMP 5.0 block dynamic shared memory allocator with fixed amount of smem
    class alignas(core::vectorization::defaultAlignment) BlockSharedMemDynOmp5BuiltInFixed
        : public BlockSharedMemDynOmp5BuiltIn
    {
        std::array<std::byte, (BlockSharedDynMemberAllocKiB << 10u)> m_fixed;

    public:
        BlockSharedMemDynOmp5BuiltInFixed(std::size_t /* bytes */) : BlockSharedMemDynOmp5BuiltIn(nullptr)
        {
            m_mem = m_fixed.data();
        }
    };
#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        pragma warning(pop)
#    endif

    namespace trait
    {
        template<typename T>
        struct GetDynSharedMem<T, BlockSharedMemDynOmp5BuiltIn>
        {
            static auto getMem(BlockSharedMemDynOmp5BuiltIn const& dyn)
            {
                static_assert(
                    core::vectorization::defaultAlignment >= alignof(T),
                    "Unable to get block shared dynamic memory for types with alignment higher than "
                    "defaultAlignment!");
                return reinterpret_cast<T*>(dyn.mem());
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
