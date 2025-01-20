#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/idx/Accessors.hpp"

#include <algorithm>
#include <ciso646> // workaround for MSVC in c++17 mode - TODO: remove once we move to c++20
#include <cstddef>
#include <type_traits>

namespace alpaka
{

    namespace detail
    {

        /* IndependentGroupsAlong
         *
         * `IndependentGroupsAlong<TAcc, Dim>(acc, groups)` returns a one-dimensional iteratable range than spans the
         * group indices from 0 to `groups`; the groups are assigned to the blocks along the `Dim` dimension. If
         * `groups` is not specified, it defaults to the number of blocks along the `Dim` dimension.
         *
         * `independentGroupsAlong<Dim>(acc, ...)` is a shorthand for `IndependentGroupsAlong<TAcc, Dim>(acc, ...)`
         * that can infer the accelerator type from the argument.
         *
         * In a 1-dimensional kernel, `independentGroups(acc, ...)` is a shorthand for `IndependentGroupsAlong<TAcc,
         * 0>(acc, ...)`.
         *
         * In an N-dimensional kernel, dimension 0 is the one that increases more slowly (e.g. the outer loop),
         * followed by dimension 1, up to dimension N-1 that increases fastest (e.g. the inner loop). For convenience
         * when converting CUDA or HIP code, `independentGroupsAlongX(acc, ...)`, `Y` and `Z` are shorthands for
         * `IndependentGroupsAlong<TAcc, N-1>(acc, ...)`, `<N-2>` and `<N-3>`.
         *
         * `independentGroupsAlong<Dim>(acc, ...)` should be called consistently by all the threads in a block. All
         * threads in a block see the same loop iterations, while threads in different blocks may see a different
         * number of iterations.
         * If the work division has more blocks than the required number of groups, the first blocks will perform one
         * iteration of the loop, while the other blocks will exit the loop immediately.
         * If the work division has less blocks than the required number of groups, some of the blocks will perform
         * more than one iteration, in order to cover then whole problem space.
         *
         * For example,
         *
         *   for (auto group: independentGroupsAlong<Dim>(acc, 7))
         *
         * will return the group range from 0 to 6, distributed across all blocks in the work division.
         * If the work division has more than 7 blocks, the first 7 will perform one iteration of the loop, while the
         * other blocks will exit the loop immediately. For example if the work division has 8 blocks, the blocks from
         * 0 to 6 will process one group while block 7 will no process any.
         * If the work division has less than 7 blocks, some of the blocks will perform more than one iteration of the
         * loop, in order to cover then whole problem space. For example if the work division has 4 blocks, block 0
         * will process the groups 0 and 4, block 1 will process groups 1 and 5, group 2 will process groups 2 and 6,
         * and block 3 will process group 3.
         */

        template<
            typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
        class IndependentGroupsAlong
        {
        public:
            using Idx = alpaka::Idx<TAcc>;

            ALPAKA_FN_ACC inline IndependentGroupsAlong(TAcc const& acc)
                : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[Dim]}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[Dim]}
                , extent_{stride_}
            {
            }

            ALPAKA_FN_ACC inline IndependentGroupsAlong(TAcc const& acc, Idx groups)
                : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[Dim]}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[Dim]}
                , extent_{groups}
            {
            }

            class const_iterator;
            using iterator = const_iterator;

            ALPAKA_FN_ACC inline const_iterator begin() const
            {
                return const_iterator(stride_, extent_, first_);
            }

            ALPAKA_FN_ACC inline const_iterator end() const
            {
                return const_iterator(stride_, extent_, extent_);
            }

            class const_iterator
            {
                friend class IndependentGroupsAlong;

                ALPAKA_FN_ACC inline const_iterator(Idx stride, Idx extent, Idx first)
                    : stride_{stride}
                    , extent_{extent}
                    , first_{std::min(first, extent)}
                {
                }

            public:
                ALPAKA_FN_ACC inline Idx operator*() const
                {
                    return first_;
                }

                // pre-increment the iterator
                ALPAKA_FN_ACC inline const_iterator& operator++()
                {
                    // increment the first-element-in-block index by the grid stride
                    first_ += stride_;
                    if(first_ < extent_)
                        return *this;

                    // the iterator has reached or passed the end of the extent, clamp it to the extent
                    first_ = extent_;
                    return *this;
                }

                // post-increment the iterator
                ALPAKA_FN_ACC inline const_iterator operator++(int)
                {
                    const_iterator old = *this;
                    ++(*this);
                    return old;
                }

                ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const
                {
                    return (first_ == other.first_);
                }

                ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const
                {
                    return not(*this == other);
                }

            private:
                // non-const to support iterator copy and assignment
                Idx stride_;
                Idx extent_;
                // modified by the pre/post-increment operator
                Idx first_;
            };

        private:
            Idx const first_;
            Idx const stride_;
            Idx const extent_;
        };

    } // namespace detail

    /* independentGroups
     *
     * `independentGroups(acc, groups)` returns a one-dimensional iteratable range than spans the group indices from 0
     * to `groups`. If `groups` is not specified, it defaults to the number of blocks.
     *
     * `independentGroups(acc, ...)` is a shorthand for `detail::IndependentGroupsAlong<TAcc, 0>(acc, ...)`.
     *
     * `independentGroups(acc, ...)` should be called consistently by all the threads in a block. All threads in a
     * block see the same loop iterations, while threads in different blocks may see a different number of iterations.
     * If the work division has more blocks than the required number of groups, the first blocks will perform one
     * iteration of the loop, while the other blocks will exit the loop immediately.
     * If the work division has less blocks than the required number of groups, some of the blocks will perform more
     * than one iteration, in order to cover then whole problem space.
     *
     * For example,
     *
     *   for (auto group: independentGroups(acc, 7))
     *
     * will return the group range from 0 to 6, distributed across all blocks in the work division.
     * If the work division has more than 7 blocks, the first 7 will perform one iteration of the loop, while the other
     * blocks will exit the loop immediately. For example if the work division has 8 blocks, the blocks from 0 to 6
     * will process one group while block 7 will no process any.
     * If the work division has less than 7 blocks, some of the blocks will perform more than one iteration of the
     * loop, in order to cover then whole problem space. For example if the work division has 4 blocks, block 0 will
     * process the groups 0 and 4, block 1 will process groups 1 and 5, group 2 will process groups 2 and 6, and block
     * 3 will process group 3.
     *
     * Note that `independentGroups(acc, ...)` is only suitable for one-dimensional kernels. For N-dimensional kernels,
     * use
     *   - `independentGroupsAlong<Dim>(acc, ...)` to perform the iteration explicitly along dimension `Dim`;
     *   - `independentGroupsAlongX(acc, ...)`, `independentGroupsAlongY(acc, ...)`, or `independentGroupsAlongZ(acc,
     *     ...)` to loop along the fastest, second-fastest, or third-fastest dimension.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
    ALPAKA_FN_ACC inline auto independentGroups(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupsAlong<TAcc, 0>(acc, static_cast<Idx>(args)...);
    }

    /* independentGroupsAlong<Dim>
     *
     * `independentGroupsAlong<Dim>(acc, ...)` is a shorthand for `detail::IndependentGroupsAlong<TAcc, Dim>(acc, ...)`
     * that can infer the accelerator type from the argument.
     */

    template<
        std::size_t Dim,
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
    ALPAKA_FN_ACC inline auto independentGroupsAlong(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupsAlong<TAcc, Dim>(acc, static_cast<Idx>(args)...);
    }

    /* independentGroupsAlongX, Y, Z
     *
     * Like `independentGroups` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest
     * dimensions.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
    ALPAKA_FN_ACC inline auto independentGroupsAlongX(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupsAlong<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
    ALPAKA_FN_ACC inline auto independentGroupsAlongY(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupsAlong<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
    ALPAKA_FN_ACC inline auto independentGroupsAlongZ(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupsAlong<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
    }

    namespace detail
    {

        /* IndependentGroupElementsAlong
         *
         * `independentGroupElementsAlong<Dim>(acc, ...)` is a shorthand for `IndependentGroupElementsAlong<TAcc,
         * Dim>(acc, ...)` that can infer the accelerator type from the argument.
         */

        template<
            typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
        class IndependentGroupElementsAlong
        {
        public:
            using Idx = alpaka::Idx<TAcc>;

            ALPAKA_FN_ACC inline IndependentGroupElementsAlong(TAcc const& acc)
                : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]}
                , thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_}
                , stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_}
                , extent_{stride_}
            {
            }

            ALPAKA_FN_ACC inline IndependentGroupElementsAlong(TAcc const& acc, Idx extent)
                : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]}
                , thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_}
                , stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_}
                , extent_{extent}
            {
            }

            ALPAKA_FN_ACC inline IndependentGroupElementsAlong(TAcc const& acc, Idx first, Idx extent)
                : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]}
                , thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_ + first}
                , stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_}
                , extent_{extent}
            {
            }

            class const_iterator;
            using iterator = const_iterator;

            ALPAKA_FN_ACC inline const_iterator begin() const
            {
                return const_iterator(elements_, stride_, extent_, thread_);
            }

            ALPAKA_FN_ACC inline const_iterator end() const
            {
                return const_iterator(elements_, stride_, extent_, extent_);
            }

            class const_iterator
            {
                friend class IndependentGroupElementsAlong;

                ALPAKA_FN_ACC inline const_iterator(Idx elements, Idx stride, Idx extent, Idx first)
                    : elements_{elements}
                    ,
                    // we need to reduce the stride by on element range because index_ is later increased with each
                    // increment
                    stride_{stride - elements}
                    , extent_{extent}
                    , index_{std::min(first, extent)}
                {
                }

            public:
                ALPAKA_FN_ACC inline Idx operator*() const
                {
                    return index_;
                }

                // pre-increment the iterator
                ALPAKA_FN_ACC inline const_iterator& operator++()
                {
                    ++indexElem_;
                    ++index_;
                    if(indexElem_ >= elements_)
                    {
                        indexElem_ = 0;
                        index_ += stride_;
                    }
                    if(index_ >= extent_)
                        index_ = extent_;

                    return *this;
                }

                // post-increment the iterator
                ALPAKA_FN_ACC inline const_iterator operator++(int)
                {
                    const_iterator old = *this;
                    ++(*this);
                    return old;
                }

                ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const
                {
                    return (*(*this) == *other);
                }

                ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const
                {
                    return not(*this == other);
                }

            private:
                // non-const to support iterator copy and assignment
                Idx elements_;
                Idx stride_;
                Idx extent_;
                // modified by the pre/post-increment operator
                Idx index_;
                Idx indexElem_ = 0;
            };

        private:
            Idx const elements_;
            Idx const thread_;
            Idx const stride_;
            Idx const extent_;
        };

    } // namespace detail

    /* independentGroupElements
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
    ALPAKA_FN_ACC inline auto independentGroupElements(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupElementsAlong<TAcc, 0>(acc, static_cast<Idx>(args)...);
    }

    /* independentGroupElementsAlong<Dim>
     *
     * `independentGroupElementsAlong<Dim>(acc, ...)` is a shorthand for `detail::IndependentGroupElementsAlong<TAcc,
     * Dim>(acc, ...)` that can infer the accelerator type from the argument.
     */

    template<
        std::size_t Dim,
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
    ALPAKA_FN_ACC inline auto independentGroupElementsAlong(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupElementsAlong<TAcc, Dim>(acc, static_cast<Idx>(args)...);
    }

    /* independentGroupElementsAlongX, Y, Z
     *
     * Like `independentGroupElements` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest
     * dimensions.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
    ALPAKA_FN_ACC inline auto independentGroupElementsAlongX(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 1>(
            acc,
            static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
    ALPAKA_FN_ACC inline auto independentGroupElementsAlongY(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 2>(
            acc,
            static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
    ALPAKA_FN_ACC inline auto independentGroupElementsAlongZ(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::IndependentGroupElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 3>(
            acc,
            static_cast<Idx>(args)...);
    }

} // namespace alpaka
