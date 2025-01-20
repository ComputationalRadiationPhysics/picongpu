#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/exec/ElementIndex.hpp"
#include "alpaka/idx/Accessors.hpp"

#include <algorithm>
#include <ciso646> // workaround for MSVC in c++17 mode - TODO: remove once we move to c++20
#include <cstddef>
#include <type_traits>

namespace alpaka
{

    namespace detail
    {

        /* UniformElementsAlong
         *
         * `UniformElementsAlong<TAcc, Dim>(acc [, first], extent)` returns a one-dimensional iteratable range that
         * spans the element indices from `first` (inclusive) to `extent` (exlusive) along the `Dim` dimension. If
         * `first` is not specified, it defaults to 0. If `extent` is not specified, it defaults to the kernel grid
         * size along the `Dim` dimension.
         *
         * `uniformElementsAlong<Dim>(acc, ...)` is a shorthand for `UniformElementsAlong<TAcc, Dim>(acc, ...)` that
         * can infer the accelerator type from the argument.
         *
         * In a 1-dimensional kernel, `uniformElements(acc, ...)` is a shorthand for `UniformElementsAlong<TAcc,
         * 0>(acc, ...)`.
         *
         * In an N-dimensional kernel, dimension 0 is the one that increases more slowly (e.g. the outer loop),
         * followed by dimension 1, up to dimension N-1 that increases fastest (e.g. the inner loop). For convenience
         * when converting CUDA or HIP code, `uniformElementsAlongX(acc, ...)`, `Y` and `Z` are shorthands for
         * `UniformElementsAlong<TAcc, N-1>(acc, ...)`, `<N-2>` and `<N-3>`.
         *
         * To cover the problem space, different threads may execute a different number of iterations. As a result, it
         * is not safe to call `alpaka::syncBlockThreads()` and other block-level synchronisations within this loop. If
         * a block synchronisation is needed, one should split the loop into an outer loop over the groups and an inner
         * loop over each group's elements, and synchronise only in the outer loop:
         *
         *  for (auto group : uniformGroupsAlong<Dim>(acc, extent)) {
         *    for (auto element : uniformGroupElementsAlong<Dim>(acc, group, extent)) {
         *       // first part of the computation
         *       // no synchronisations here
         *       ...
         *    }
         *    // wait for all threads to complete the first part
         *    alpaka::syncBlockThreads();
         *    for (auto element : uniformGroupElementsAlong<Dim>(acc, group, extent)) {
         *       // second part of the computation
         *       // no synchronisations here
         *       ...
         *    }
         *    // wait for all threads to complete the second part
         *    alpaka::syncBlockThreads();
         *    ...
         *  }
         *
         * Warp-level primitives require that all threads in the warp execute the same function. If `extent` is not a
         * multiple of the warp size, some of the warps may be incomplete, leading to undefined behaviour - for
         * example, the kernel may hang. To avoid this problem, round up `extent` to a multiple of the warp size, and
         * check the element index explicitly inside the loop:
         *
         *  for (auto element : uniformElementsAlong<N-1>(acc, round_up_by(extent, alpaka::warp::getSize(acc)))) {
         *    bool flag = false;
         *    if (element < extent) {
         *      // do some work and compute a result flag only for the valid elements
         *      flag = do_some_work();
         *    }
         *    // check if any valid element had a positive result
         *    if (alpaka::warp::any(acc, flag)) {
         *      // ...
         *    }
         *  }
         *
         * Note that the use of warp-level primitives is usually suitable only for the fastest-looping dimension,
         * `N-1`.
         */

        template<
            typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
        class UniformElementsAlong
        {
        public:
            using Idx = alpaka::Idx<TAcc>;

            ALPAKA_FN_ACC inline UniformElementsAlong(TAcc const& acc)
                : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]}
                , first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_}
                , extent_{stride_}
            {
            }

            ALPAKA_FN_ACC inline UniformElementsAlong(TAcc const& acc, Idx extent)
                : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]}
                , first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_}
                , extent_{extent}
            {
            }

            ALPAKA_FN_ACC inline UniformElementsAlong(TAcc const& acc, Idx first, Idx extent)
                : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]}
                , first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_ + first}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_}
                , extent_{extent}
            {
            }

            class const_iterator;
            using iterator = const_iterator;

            ALPAKA_FN_ACC inline const_iterator begin() const
            {
                return const_iterator(elements_, stride_, extent_, first_);
            }

            ALPAKA_FN_ACC inline const_iterator end() const
            {
                return const_iterator(elements_, stride_, extent_, extent_);
            }

            class const_iterator
            {
                friend class UniformElementsAlong;

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
                    // increment the index along the elements processed by the current thread
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
            Idx const first_;
            Idx const stride_;
            Idx const extent_;
        };

    } // namespace detail

    /* uniformElements
     *
     * `uniformElements(acc [, first], extent)` returns a one-dimensional iteratable range that spans the element
     * indices from `first` (inclusive) to `extent` (exlusive). If `first` is not specified, it defaults to 0. If
     * `extent` is not specified, it defaults to the kernel grid size.
     *
     * `uniformElements(acc, ...)` is a shorthand for `detail::UniformElementsAlong<TAcc, 0>(acc, ...)`.
     *
     * To cover the problem space, different threads may execute a different number of iterations. As a result, it is
     * not safe to call `alpaka::syncBlockThreads()` and other block-level synchronisations within this loop. If a
     * block synchronisation is needed, one should split the loop into an outer loop over the groups and an inner loop
     * over each group's elements, and synchronise only in the outer loop:
     *
     *  for (auto group : uniformGroups(acc, extent)) {
     *    for (auto element : uniformGroupElements(acc, group, extent)) {
     *       // first part of the computation
     *       // no synchronisations here
     *       ...
     *    }
     *    // wait for all threads to complete the first part
     *    alpaka::syncBlockThreads();
     *    for (auto element : uniformGroupElements(acc, group, extent)) {
     *       // second part of the computation
     *       // no synchronisations here
     *       ...
     *    }
     *    // wait for all threads to complete the second part
     *    alpaka::syncBlockThreads();
     *    ...
     *  }
     *
     * Warp-level primitives require that all threads in the warp execute the same function. If `extent` is not a
     * multiple of the warp size, some of the warps may be incomplete, leading to undefined behaviour - for example,
     * the kernel may hang. To avoid this problem, round up `extent` to a multiple of the warp size, and check the
     * element index explicitly inside the loop:
     *
     *  for (auto element : uniformElements(acc, round_up_by(extent, alpaka::warp::getSize(acc)))) {
     *    bool flag = false;
     *    if (element < extent) {
     *      // do some work and compute a result flag only for elements up to extent
     *      flag = do_some_work();
     *    }
     *    // check if any valid element had a positive result
     *    if (alpaka::warp::any(acc, flag)) {
     *      // ...
     *    }
     *  }
     *
     * Note that `uniformElements(acc, ...)` is only suitable for one-dimensional kernels. For N-dimensional kernels,
     * use
     *   - `uniformElementsND(acc, ...)` to cover an N-dimensional problem space with a single loop;
     *   - `uniformElementsAlong<Dim>(acc, ...)` to perform the iteration explicitly along dimension `Dim`;
     *   - `uniformElementsAlongX(acc, ...)`, `uniformElementsAlongY(acc, ...)`, or `uniformElementsAlongZ(acc, ...)`
     *     to loop along the fastest, second-fastest, or third-fastest dimension.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
    ALPAKA_FN_ACC inline auto uniformElements(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformElementsAlong<TAcc, 0>(acc, static_cast<Idx>(args)...);
    }

    /* uniformElementsAlong<Dim>
     *
     * `uniformElementsAlong<Dim>(acc, ...)` is a shorthand for `detail::UniformElementsAlong<TAcc, Dim>(acc, ...)`
     * that can infer the accelerator type from the argument.
     */

    template<
        std::size_t Dim,
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
    ALPAKA_FN_ACC inline auto uniformElementsAlong(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformElementsAlong<TAcc, Dim>(acc, static_cast<Idx>(args)...);
    }

    /* uniformElementsAlongX, Y, Z
     *
     * Like `uniformElements` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest
     * dimensions.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
    ALPAKA_FN_ACC inline auto uniformElementsAlongX(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
    ALPAKA_FN_ACC inline auto uniformElementsAlongY(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
    ALPAKA_FN_ACC inline auto uniformElementsAlongZ(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
    }

    namespace detail
    {

        /* UniformElementsND
         *
         * `UniformElementsND(acc, extent)` returns an N-dimensional iteratable range that spans the element indices
         * required to cover the given problem size, indicated by `extent`.
         *
         * `uniformElementsND(acc, ...)` is an alias for `UniformElementsND<TAcc>(acc, ...)`.
         *
         * To cover the problem space, different threads may execute a different number of iterations. As a result, it
         * is not safe to call `alpaka::syncBlockThreads()` and other block-level synchronisations within this loop. If
         * a block synchronisation is needed, one should split the loop into an outer loop over the groups and an inner
         * loop over each group's elements, and synchronise only in the outer loop:
         *
         *  for (auto group0 : uniformGroupsAlong<0>(acc, extent[0])) {
         *    for (auto group1 : uniformGroupsAlong<1>(acc, extent[1])) {
         *      for (auto element0 : uniformGroupElementsAlong<0>(acc, group0, extent[0])) {
         *        for (auto element1 : uniformGroupElementsAlong<1>(acc, group1, extent[1])) {
         *           // first part of the computation
         *           // no synchronisations here
         *           ...
         *        }
         *      }
         *      // wait for all threads to complete the first part
         *      alpaka::syncBlockThreads();
         *      for (auto element0 : uniformGroupElementsAlong<0>(acc, group0, extent[0])) {
         *        for (auto element1 : uniformGroupElementsAlong<1>(acc, group1, extent[1])) {
         *           // second part of the computation
         *           // no synchronisations here
         *           ...
         *        }
         *      }
         *      // wait for all threads to complete the second part
         *      alpaka::syncBlockThreads();
         *      ...
         *    }
         *  }
         *
         * For more details, see `UniformElementsAlong<TAcc, Dim>(acc, ...)`.
         */

        template<
            typename TAcc,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
        class UniformElementsND
        {
        public:
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            ALPAKA_FN_ACC inline UniformElementsND(TAcc const& acc)
                : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)}
                , thread_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_}
                , extent_{stride_}
            {
            }

            ALPAKA_FN_ACC inline UniformElementsND(TAcc const& acc, Vec extent)
                : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)}
                , thread_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_}
                , extent_{extent}
            {
            }

            // tag used to construct an end iterator
            struct at_end_t
            {
            };

            class const_iterator;
            using iterator = const_iterator;

            ALPAKA_FN_ACC inline const_iterator begin() const
            {
                // check that all dimensions of the current thread index are within the extent
                if((thread_ < extent_).all())
                {
                    // construct an iterator pointing to the first element to be processed by the current thread
                    return const_iterator{this, thread_};
                }
                else
                {
                    // construct an end iterator, pointing post the end of the extent
                    return const_iterator{this, at_end_t{}};
                }
            }

            ALPAKA_FN_ACC inline const_iterator end() const
            {
                // construct an end iterator, pointing post the end of the extent
                return const_iterator{this, at_end_t{}};
            }

            class const_iterator
            {
                friend class UniformElementsND;

            public:
                ALPAKA_FN_ACC inline Vec operator*() const
                {
                    return index_;
                }

                // pre-increment the iterator
                ALPAKA_FN_ACC inline constexpr const_iterator operator++()
                {
                    increment();
                    return *this;
                }

                // post-increment the iterator
                ALPAKA_FN_ACC inline constexpr const_iterator operator++(int)
                {
                    const_iterator old = *this;
                    increment();
                    return old;
                }

                ALPAKA_FN_ACC inline constexpr bool operator==(const_iterator const& other) const
                {
                    return (index_ == other.index_);
                }

                ALPAKA_FN_ACC inline constexpr bool operator!=(const_iterator const& other) const
                {
                    return not(*this == other);
                }

            private:
                // construct an iterator pointing to the first element to be processed by the current thread
                ALPAKA_FN_ACC inline const_iterator(UniformElementsND const* loop, Vec first)
                    : loop_{loop}
                    , first_{alpaka::elementwise_min(first, loop->extent_)}
                    , range_{alpaka::elementwise_min(first + loop->elements_, loop->extent_)}
                    , index_{first_}
                {
                }

                // construct an end iterator, pointing post the end of the extent
                ALPAKA_FN_ACC inline const_iterator(UniformElementsND const* loop, at_end_t const&)
                    : loop_{loop}
                    , first_{loop_->extent_}
                    , range_{loop_->extent_}
                    , index_{loop_->extent_}
                {
                }

                template<size_t I>
                ALPAKA_FN_ACC inline constexpr bool nth_elements_loop()
                {
                    bool overflow = false;
                    ++index_[I];
                    if(index_[I] >= range_[I])
                    {
                        index_[I] = first_[I];
                        overflow = true;
                    }
                    return overflow;
                }

                template<size_t N>
                ALPAKA_FN_ACC inline constexpr bool do_elements_loops()
                {
                    if constexpr(N == 0)
                    {
                        // overflow
                        return true;
                    }
                    else
                    {
                        if(not nth_elements_loop<N - 1>())
                        {
                            return false;
                        }
                        else
                        {
                            return do_elements_loops<N - 1>();
                        }
                    }
                    ALPAKA_UNREACHABLE(false);
                }

                template<size_t I>
                ALPAKA_FN_ACC inline constexpr bool nth_strided_loop()
                {
                    bool overflow = false;
                    first_[I] += loop_->stride_[I];
                    if(first_[I] >= loop_->extent_[I])
                    {
                        first_[I] = loop_->thread_[I];
                        overflow = true;
                    }
                    index_[I] = first_[I];
                    range_[I] = std::min(first_[I] + loop_->elements_[I], loop_->extent_[I]);
                    return overflow;
                }

                template<size_t N>
                ALPAKA_FN_ACC inline constexpr bool do_strided_loops()
                {
                    if constexpr(N == 0)
                    {
                        // overflow
                        return true;
                    }
                    else
                    {
                        if(not nth_strided_loop<N - 1>())
                        {
                            return false;
                        }
                        else
                        {
                            return do_strided_loops<N - 1>();
                        }
                    }
                    ALPAKA_UNREACHABLE(false);
                }

                // increment the iterator
                ALPAKA_FN_ACC inline constexpr void increment()
                {
                    // linear N-dimensional loops over the elements associated to the thread;
                    // do_elements_loops<>() returns true if any of those loops overflows
                    if(not do_elements_loops<Dim::value>())
                    {
                        // the elements loops did not overflow, return the next index
                        return;
                    }

                    // strided N-dimensional loop over the threads in the kernel launch grid;
                    // do_strided_loops<>() returns true if any of those loops overflows
                    if(not do_strided_loops<Dim::value>())
                    {
                        // the strided loops did not overflow, return the next index
                        return;
                    }

                    // the iterator has reached or passed the end of the extent, clamp it to the extent
                    first_ = loop_->extent_;
                    range_ = loop_->extent_;
                    index_ = loop_->extent_;
                }

                // const pointer to the UniformElementsND that the iterator refers to
                UniformElementsND const* loop_;

                // modified by the pre/post-increment operator
                Vec first_; // first element processed by this thread
                Vec range_; // last element processed by this thread
                Vec index_; // current element processed by this thread
            };

        private:
            Vec const elements_;
            Vec const thread_;
            Vec const stride_;
            Vec const extent_;
        };

    } // namespace detail

    /* uniformElementsND
     *
     * `uniformElementsND(acc, ...)` is a shorthand for `detail::UniformElementsND<TAcc>(acc, ...)`.
     */

    template<
        typename TAcc,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
    ALPAKA_FN_ACC inline auto uniformElementsND(TAcc const& acc)
    {
        return detail::UniformElementsND<TAcc>(acc);
    }

    template<
        typename TAcc,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
    ALPAKA_FN_ACC inline auto uniformElementsND(
        TAcc const& acc,
        alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> extent)
    {
        return detail::UniformElementsND<TAcc>(acc, extent);
    }

    namespace detail
    {

        /* UniformGroupsAlong
         *
         * `UniformGroupsAlong<Dim>(acc, elements)` returns a one-dimensional iteratable range than spans the group
         * indices required to cover the given problem size along the `Dim` dimension, in units of the block size.
         * `elements` indicates the total number of elements, across all groups; if not specified, it defaults to the
         * kernel grid size along the `Dim` dimension.
         *
         * `uniformGroupsAlong<Dim>(acc, ...)` is a shorthand for `UniformGroupsAlong<TAcc, Dim>(acc, ...)` that can
         * infer the accelerator type from the argument.
         *
         * In a 1-dimensional kernel, `uniformGroups(acc, ...)` is a shorthand for `UniformGroupsAlong<Tacc, 0>(acc,
         * ...)`.
         *
         * In an N-dimensional kernel, dimension 0 is the one that increases more slowly (e.g. the outer loop),
         * followed by dimension 1, up to dimension N-1 that increases fastest (e.g. the inner loop). For convenience
         * when converting CUDA or HIP code, `uniformGroupsAlongX(acc, ...)`, `Y` and `Z` are shorthands for
         * `UniformGroupsAlong<TAcc, N-1>(acc, ...)`, `<N-2>` and `<N-3>`.
         *
         * `uniformGroupsAlong<Dim>(acc, ...)` should be called consistently by all the threads in a block. All
         * threads in a block see the same loop iterations, while threads in different blocks may see a different
         * number of iterations. If the work division has more blocks than the required number of groups, the first
         * blocks will perform one iteration of the loop, while the other blocks will exit the loop immediately. If the
         * work division has less blocks than the required number of groups, some of the blocks will perform more than
         * one iteration, in order to cover then whole problem space.
         *
         * If the problem size is not a multiple of the block size, the last group will process a number of elements
         * smaller than the block size. However, also in this case all threads in the block will execute the same
         * number of iterations of this loop: this makes it safe to use block-level synchronisations in the loop body.
         * It is left to the inner loop (or the user) to ensure that only the correct number of threads process any
         * data; this logic is implemented by `uniformGroupElementsAlong<Dim>(acc, group, elements)`.
         *
         * For example, if the block size is 64 and there are 400 elements
         *
         *   for (auto group: uniformGroupsAlong<Dim>(acc, 400)
         *
         * will return the group range from 0 to 6, distributed across all blocks in the work division: group 0 should
         * cover the elements from 0 to 63, group 1 should cover the elements from 64 to 127, etc., until the last
         * group, group 6, should cover the elements from 384 to 399. All the threads of the block will process this
         * last group; it is up to the inner loop to not process the non-existing elements after 399.
         *
         * If the work division has more than 7 blocks, the first 7 will perform one iteration of the loop, while the
         * other blocks will exit the loop immediately. For example if the work division has 8 blocks, the blocks from
         * 0 to 6 will process one group while block 7 will no process any.
         *
         * If the work division has less than 7 blocks, some of the blocks will perform more than one iteration of the
         * loop, in order to cover then whole problem space. For example if the work division has 4 blocks, block 0
         * will process the groups 0 and 4, block 1 will process groups 1 and 5, group 2 will process groups 2 and 6,
         * and block 3 will process group 3.
         *
         * See `UniformElementsAlong<TAcc, Dim>(acc, ...)` for a concrete example using `uniformGroupsAlong<Dim>` and
         * `uniformGroupElementsAlong<Dim>`.
         */

        template<
            typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
        class UniformGroupsAlong
        {
        public:
            using Idx = alpaka::Idx<TAcc>;

            ALPAKA_FN_ACC inline UniformGroupsAlong(TAcc const& acc)
                : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[Dim]}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[Dim]}
                , extent_{stride_}
            {
            }

            // extent is the total number of elements (not blocks)
            ALPAKA_FN_ACC inline UniformGroupsAlong(TAcc const& acc, Idx extent)
                : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[Dim]}
                , stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[Dim]}
                , extent_{alpaka::core::divCeil(extent, alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[Dim])}
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
                friend class UniformGroupsAlong;

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

    /* uniformGroups
     *
     * `uniformGroups(acc, elements)` returns a one-dimensional iteratable range than spans the group indices required
     * to cover the given problem size, in units of the block size. `elements` indicates the total number of elements,
     * across all groups; if not specified, it defaults to the kernel grid size.
     *
     * `uniformGroups(acc, ...)` is a shorthand for `detail::UniformGroupsAlong<TAcc, 0>(acc, ...)`.
     *
     * `uniformGroups(acc, ...)` should be called consistently by all the threads in a block. All threads in a block
     * see the same loop iterations, while threads in different blocks may see a different number of iterations. If the
     * work division has more blocks than the required number of groups, the first blocks will perform one iteration of
     * the loop, while the other blocks will exit the loop immediately. If the work division has less blocks than the
     * required number of groups, some of the blocks will perform more than one iteration, in order to cover then whole
     * problem space.
     *
     * If the problem size is not a multiple of the block size, the last group will process a number of elements
     * smaller than the block size. However, also in this case all threads in the block will execute the same number of
     * iterations of this loop: this makes it safe to use block-level synchronisations in the loop body. It is left to
     * the inner loop (or the user) to ensure that only the correct number of threads process any data; this logic is
     * implemented by `uniformGroupElements(acc, group, elements)`.
     *
     * For example, if the block size is 64 and there are 400 elements
     *
     *   for (auto group: uniformGroups(acc, 400)
     *
     * will return the group range from 0 to 6, distributed across all blocks in the work division: group 0 should
     * cover the elements from 0 to 63, group 1 should cover the elements from 64 to 127, etc., until the last group,
     * group 6, should cover the elements from 384 to 399. All the threads of the block will process this last group;
     * it is up to the inner loop to not process the non-existing elements after 399.
     *
     * If the work division has more than 7 blocks, the first 7 will perform one iteration of the loop, while the other
     * blocks will exit the loop immediately. For example if the work division has 8 blocks, the blocks from 0 to 6
     * will process one group while block 7 will no process any.
     *
     * If the work division has less than 7 blocks, some of the blocks will perform more than one iteration of the
     * loop, in order to cover then whole problem space. For example if the work division has 4 blocks, block 0 will
     * process the groups 0 and 4, block 1 will process groups 1 and 5, group 2 will process groups 2 and 6, and block
     * 3 will process group 3.
     *
     * See `uniformElements(acc, ...)` for a concrete example using `uniformGroups` and `uniformGroupElements`.
     *
     * Note that `uniformGroups(acc, ...)` is only suitable for one-dimensional kernels. For N-dimensional kernels,
     * use
     *   - `uniformGroupsAlong<Dim>(acc, ...)` to perform the iteration explicitly along dimension `Dim`;
     *   - `uniformGroupsAlongX(acc, ...)`, `uniformGroupsAlongY(acc, ...)`, or `uniformGroupsAlongZ(acc, ...)` to loop
     *     along the fastest, second-fastest, or third-fastest dimension.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
    ALPAKA_FN_ACC inline auto uniformGroups(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupsAlong<TAcc, 0>(acc, static_cast<Idx>(args)...);
    }

    /* uniformGroupsAlong<Dim>
     *
     * `uniformGroupsAlong<Dim>(acc, ...)` is a shorthand for `detail::UniformGroupsAlong<TAcc, Dim>(acc, ...)` that
     * can infer the accelerator type from the argument.
     */

    template<
        std::size_t Dim,
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
    ALPAKA_FN_ACC inline auto uniformGroupsAlong(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupsAlong<TAcc, Dim>(acc, static_cast<Idx>(args)...);
    }

    /* uniformGroupsAlongX, Y, Z
     *
     * Like `uniformGroups` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest
     * dimensions.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
    ALPAKA_FN_ACC inline auto uniformGroupsAlongX(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupsAlong<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
    ALPAKA_FN_ACC inline auto uniformGroupsAlongY(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupsAlong<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
    ALPAKA_FN_ACC inline auto uniformGroupsAlongZ(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupsAlong<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
    }

    namespace detail
    {

        /* UniformGroupElementsAlong
         *
         * `UniformGroupElementsAlong<TAcc, Dim>(acc, group, elements)` returns a one-dimensional iteratable range that
         * spans all the elements within the given `group` along dimension `Dim`, as obtained from
         * `UniformGroupsAlong<Dim>`, up to `elements` (exclusive). `elements` indicates the total number of elements
         * across all groups; if not specified, it defaults to the kernel grid size.
         *
         * `uniformGroupElementsAlong<Dim>(acc, ...)` is a shorthand for `UniformGroupElementsAlong<TAcc, Dim>(acc,
         * ...)` that can infer the accelerator type from the argument.
         *
         * In a 1-dimensional kernel, `uniformGroupElements(acc, ...)` is a shorthand for
         * `UniformGroupElementsAlong<0>(acc, ...)`.
         *
         * In an N-dimensional kernel, dimension 0 is the one that increases more slowly (e.g. the outer loop),
         * followed by dimension 1, up to dimension N-1 that increases fastest (e.g. the inner loop). For convenience
         * when converting CUDA or HIP code, `uniformGroupElementsAlongX(acc, ...)`, `Y` and `Z` are shorthands for
         * `UniformGroupElementsAlong<TAcc, N-1>(acc, ...)`, `<N-2>` and `<N-3>`.
         *
         * Iterating over the range yields values of type `ElementIndex`, that provide the `.global` and `.local`
         * indices of the corresponding element. The global index spans a subset of the range from 0 to `elements`
         * (excluded), while the local index spans the range from 0 to the block size (excluded).
         *
         * The loop will perform a number of iterations up to the number of elements per thread, stopping earlier if
         * the global element index reaches `elements`.
         *
         * If the problem size is not a multiple of the block size, different threads may execute a different number of
         * iterations. As a result, it is not safe to call `alpaka::syncBlockThreads()` within this loop. If a block
         * synchronisation is needed, one should split the loop, and synchronise the threads between the loops.
         * See `UniformElementsAlong<Dim>(acc, ...)` for a concrete example using `uniformGroupsAlong<Dim>` and
         * `uniformGroupElementsAlong<Dim>`.
         *
         * Warp-level primitives require that all threads in the warp execute the same function. If `elements` is not a
         * multiple of the warp size, some of the warps may be incomplete, leading to undefined behaviour - for
         * example, the kernel may hang. To avoid this problem, round up `elements` to a multiple of the warp size, and
         * check the element index explicitly inside the loop:
         *
         *  for (auto element : uniformGroupElementsAlong<N-1>(acc, group, round_up_by(elements,
         * alpaka::warp::getSize(acc)))) { bool flag = false; if (element < elements) {
         *      // do some work and compute a result flag only for the valid elements
         *      flag = do_some_work();
         *    }
         *    // check if any valid element had a positive result
         *    if (alpaka::warp::any(acc, flag)) {
         *      // ...
         *    }
         *  }
         *
         * Note that the use of warp-level primitives is usually suitable only for the fastest-looping dimension,
         * `N-1`.
         */

        template<
            typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
        class UniformGroupElementsAlong
        {
        public:
            using Idx = alpaka::Idx<TAcc>;

            ALPAKA_FN_ACC inline UniformGroupElementsAlong(TAcc const& acc, Idx block)
                : first_{block * alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[Dim]}
                , local_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] * alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]}
                , range_{local_ + alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]}
            {
            }

            ALPAKA_FN_ACC inline UniformGroupElementsAlong(TAcc const& acc, Idx block, Idx extent)
                : first_{block * alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[Dim]}
                , local_{std::min(
                      extent - first_,
                      alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim]
                          * alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim])}
                , range_{
                      std::min(extent - first_, local_ + alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim])}
            {
            }

            class const_iterator;
            using iterator = const_iterator;

            ALPAKA_FN_ACC inline const_iterator begin() const
            {
                return const_iterator(local_, first_, range_);
            }

            ALPAKA_FN_ACC inline const_iterator end() const
            {
                return const_iterator(range_, first_, range_);
            }

            class const_iterator
            {
                friend class UniformGroupElementsAlong;

                ALPAKA_FN_ACC inline const_iterator(Idx local, Idx first, Idx range)
                    : index_{local}
                    , first_{first}
                    , range_{range}
                {
                }

            public:
                ALPAKA_FN_ACC inline ElementIndex<Idx> operator*() const
                {
                    return ElementIndex<Idx>{index_ + first_, index_};
                }

                // pre-increment the iterator
                ALPAKA_FN_ACC inline const_iterator& operator++()
                {
                    // increment the index along the elements processed by the current thread
                    ++index_;
                    if(index_ < range_)
                        return *this;

                    // the iterator has reached or passed the end of the extent, clamp it to the extent
                    index_ = range_;
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
                    return (index_ == other.index_);
                }

                ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const
                {
                    return not(*this == other);
                }

            private:
                // modified by the pre/post-increment operator
                Idx index_;
                // non-const to support iterator copy and assignment
                Idx first_;
                Idx range_;
            };

        private:
            Idx const first_;
            Idx const local_;
            Idx const range_;
        };

    } // namespace detail

    /* uniformGroupElements
     *
     * `uniformGroupElements(acc, group, elements)` returns a one-dimensional iteratable range that spans all the
     * elements within the given `group`, as obtained from `uniformGroups`, up to `elements` (exclusive). `elements`
     * indicates the total number of elements across all groups; if not specified, it defaults to the kernel grid size.
     *
     * `uniformGroupElements(acc, ...)` is a shorthand for `detail::UniformGroupElementsAlong<0>(acc, ...)`.
     *
     * Iterating over the range yields values of type `ElementIndex`, that provide the `.global` and `.local` indices
     * of the corresponding element. The global index spans a subset of the range from 0 to `elements` (excluded),
     * while the local index spans the range from 0 to the block size (excluded).
     *
     * The loop will perform a number of iterations up to the number of elements per thread, stopping earlier if the
     * global element index reaches `elements`.
     *
     * If the problem size is not a multiple of the block size, different threads may execute a different number of
     * iterations. As a result, it is not safe to call `alpaka::syncBlockThreads()` within this loop. If a block
     * synchronisation is needed, one should split the loop, and synchronise the threads between the loops.
     * See `uniformElements(acc, ...)` for a concrete example using `uniformGroups` and `uniformGroupElements`.
     *
     * Warp-level primitives require that all threads in the warp execute the same function. If `elements` is not a
     * multiple of the warp size, some of the warps may be incomplete, leading to undefined behaviour - for example,
     * the kernel may hang. To avoid this problem, round up `elements` to a multiple of the warp size, and check the
     * element index explicitly inside the loop:
     *
     *  for (auto element : uniformGroupElements(acc, group, round_up_by(elements, alpaka::warp::getSize(acc)))) {
     *    bool flag = false;
     *    if (element < elements) {
     *      // do some work and compute a result flag only for the valid elements
     *      flag = do_some_work();
     *    }
     *    // check if any valid element had a positive result
     *    if (alpaka::warp::any(acc, flag)) {
     *      // ...
     *    }
     *  }
     *
     * Note that `uniformGroupElements(acc, ...)` is only suitable for one-dimensional kernels. For N-dimensional
     * kernels, use
     *   - `detail::UniformGroupElementsAlong<Dim>(acc, ...)` to perform the iteration explicitly along dimension
     *     `Dim`;
     *   - `uniformGroupElementsAlongX(acc, ...)`, `uniformGroupElementsAlongY(acc, ...)`, or
     *     `uniformGroupElementsAlongZ(acc, ...)` to loop along the fastest, second-fastest, or third-fastest
     *     dimension.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
    ALPAKA_FN_ACC inline auto uniformGroupElements(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupElementsAlong<TAcc, 0>(acc, static_cast<Idx>(args)...);
    }

    /* uniformGroupElementsAlong<Dim>
     *
     * `uniformGroupElementsAlong<Dim>(acc, ...)` is a shorthand for `detail::UniformGroupElementsAlong<TAcc,
     * Dim>(acc, ...)` that can infer the accelerator type from the argument.
     */

    template<
        std::size_t Dim,
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
    ALPAKA_FN_ACC inline auto uniformGroupElementsAlong(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupElementsAlong<TAcc, Dim>(acc, static_cast<Idx>(args)...);
    }

    /* uniformGroupElementsAlongX, Y, Z
     *
     * Like `uniformGroupElements` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest
     * dimensions.
     */

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
    ALPAKA_FN_ACC inline auto uniformGroupElementsAlongX(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
    ALPAKA_FN_ACC inline auto uniformGroupElementsAlongY(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
    }

    template<
        typename TAcc,
        typename... TArgs,
        typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
    ALPAKA_FN_ACC inline auto uniformGroupElementsAlongZ(TAcc const& acc, TArgs... args)
    {
        using Idx = alpaka::Idx<TAcc>;
        return detail::UniformGroupElementsAlong<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
    }

} // namespace alpaka
