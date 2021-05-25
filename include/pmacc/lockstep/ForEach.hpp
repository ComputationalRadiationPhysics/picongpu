/* Copyright 2017-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/lockstep/Config.hpp"
#include "pmacc/lockstep/Idx.hpp"
#include "pmacc/lockstep/Variable.hpp"
#include "pmacc/lockstep/Worker.hpp"
#include "pmacc/types.hpp"

#include <cstdint>
#include <type_traits>

namespace pmacc
{
    namespace lockstep
    {
        /** Execute a functor for the given index domain.
         *
         * Algorithm to execute a subsequent lockstep for each index of the configured domain.
         * @attention There is no implicit synchronization between workers before or after the execution of is ForEach
         * performed.
         *
         * @tparam T_Config Configuration for the domain and execution strategy.
         *                  T_Config must provide: domainSize, numCollIter, numWorkers, and simdSize at compile time.
         */
        template<typename T_Config>
        class ForEach;

        template<uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize>
        class ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>
            : Config<T_domainSize, T_numWorkers, T_simdSize>
            , Worker<T_numWorkers>
        {
            /* Helper to check if a member exists
             *
             * Derived from C++17 std::void_t.
             * This implementation will be removed with Void provided by alpaka 0.6.0 release (not included in the
             * 0.6.0rc3 we currently using).
             */
            template<class...>
            using Void = void;

            /** Get the result of a functor invocation.
             *
             * @attention The behavior is undefined for ill-formed invocations.
             *
             * @{
             */
#if BOOST_COMP_CLANG_CUDA && __CUDACC_VER_MAJOR__ <= 10
            template<typename F, typename... T_Args>
            using InvokeResult_t = decltype(alpaka::core::declval<F>()(alpaka::core::declval<T_Args>()...));
#else
#    if __cplusplus >= 201703L
            template<typename F, typename... T_Args>
            using InvokeResult_t = typename std::invoke_result<F, T_Args...>::type;
#    else
            template<typename F, typename... T_Args>
            using InvokeResult_t = typename std::result_of<F(T_Args...)>::type;
#    endif
#endif
            /**@}*/

        public:
            using BaseConfig = Config<T_domainSize, T_numWorkers, T_simdSize>;

            using BaseConfig::domainSize;
            using BaseConfig::numCollIter;
            using BaseConfig::numWorkers;
            using BaseConfig::simdSize;

            /** constructor
             *
             * @param workerIdx index of the worker: range [0;workerSize)
             */
            HDINLINE
            ForEach(uint32_t const workerIdx) : Worker<numWorkers>(std::move(workerIdx))
            {
            }

            HDINLINE Worker<numWorkers> getWorkerCfg() const
            {
                return static_cast<Worker<numWorkers>>(*this);
            }

            /** execute a functor
             *
             * Distribute the indices of the domain even over all worker and execute a user defined functor.
             * There is no guarantee in which order the indices will be processed.
             *
             * @param functor is called for each index which is mapped to the worker
             * @{
             */

            /** The functor must fulfill the following interface:
             * @code
             * void operator()( lockstep::Idx< T_domainSize > const idx );
             * // or
             * void operator()( uint32_t const linearIdx );
             * @endcode
             */
            template<
                typename T_Functor,
                // check if functor is invocable
                typename = Void<decltype(alpaka::core::declval<T_Functor>()(alpaka::core::declval<Idx const>()))>,
                std::
                    enable_if_t<std::is_void<InvokeResult_t<T_Functor, Idx const>>::value && domainSize != 1, int> = 0>
            HDINLINE void operator()(T_Functor&& functor) const
            {
                for(uint32_t i = 0u; i < numCollIter; ++i)
                {
                    uint32_t const beginWorker = i * simdSize;
                    uint32_t const beginIdx = beginWorker * numWorkers + simdSize * this->getWorkerIdx();
                    if(outerLoopCondition || !innerLoopCondition || beginIdx < domainSize)
                    {
                        for(uint32_t j = 0u; j < simdSize; ++j)
                        {
                            uint32_t const localIdx = beginIdx + j;
                            if(innerLoopCondition || localIdx < domainSize)
                                functor(Idx(localIdx, beginWorker + j));
                        }
                    }
                }
            }

            /** The functor must fulfill the following interface:
             * @code
             * void operator()();
             * @endcode
             */
            template<
                typename T_Functor,
                // check if functor is invocable
                typename = Void<decltype(alpaka::core::declval<T_Functor>()())>,
                std::enable_if_t<std::is_void<InvokeResult_t<T_Functor>>::value && domainSize != 1, int> = 0>
            HDINLINE void operator()(T_Functor&& functor) const
            {
                this->operator()([&](Idx const&) { functor(); });
            }

            /** Execute the functor with the master worker only.
             * @code
             * void operator()(lockstep::Idx< T_domainSize > const idx);
             * //or
             * void operator()(uint32_t const linearIdx);
             * @endcode
             */
            template<
                typename T_Functor,
                // check if functor is invocable
                typename = Void<decltype(alpaka::core::declval<T_Functor>()(alpaka::core::declval<Idx const>()))>,
                std::
                    enable_if_t<std::is_void<InvokeResult_t<T_Functor, Idx const>>::value && domainSize == 1, int> = 0>
            HDINLINE void operator()(T_Functor&& functor) const
            {
                if(this->getWorkerIdx() == 0u)
                    functor(Idx(0u, 0u));
            }

            /** Execute the functor with the master worker only.
             * @code
             * void operator()();
             * @endcode
             */
            template<
                typename T_Functor,
                // check if functor is invocable
                typename = Void<decltype(alpaka::core::declval<T_Functor>()())>,
                std::enable_if_t<std::is_void<InvokeResult_t<T_Functor>>::value && domainSize == 1, int> = 0>
            HDINLINE void operator()(T_Functor&& functor) const
            {
                if(this->getWorkerIdx() == 0u)
                    functor();
            }


            /** Execute the functor and create and return a variable for each index of the domain.
             *
             * The type of the variable depends on the return type of the functor.
             *
             * @code
             * auto operator()(lockstep::Idx< T_domainSize > const idx);
             * //or
             * auto operator()(uint32_t const linearIdx);
             * @endcode
             *
             * @return Variable for each index of the domain.
             */
            template<
                typename T_Functor,
                // check if functor is invocable
                typename = Void<decltype(alpaka::core::declval<T_Functor>()(alpaka::core::declval<Idx const>()))>,
                std::enable_if_t<!std::is_void<InvokeResult_t<T_Functor, Idx const>>::value, int> = 0>
            HDINLINE auto operator()(T_Functor&& functor) const
            {
                auto tmp = makeVar<ALPAKA_DECAY_T(decltype(functor(alpaka::core::declval<Idx const>())))>(*this);
                this->operator()([&](Idx const& idx) { tmp[idx] = std::move(functor(idx)); });
                return tmp;
            }

            /** Execute the functor and create and return a variable for each index of the domain.
             *
             * The type of the variable depends on the return type of the functor.
             *
             * @code
             * auto operator()();
             * @endcode
             *
             * @return Variable for each index of the domain.
             */
            template<
                typename T_Functor,
                // check if functor is invocable
                typename = Void<decltype(alpaka::core::declval<T_Functor>()())>,
                std::enable_if_t<!std::is_void<InvokeResult_t<T_Functor>>::value, int> = 0>
            HDINLINE auto operator()(T_Functor&& functor) const
            {
                auto tmp = makeVar<ALPAKA_DECAY_T(decltype(functor()))>(*this);
                this->operator()([&](Idx const& idx) { tmp[idx] = std::move(functor()); });
                return tmp;
            }
            /** @} */

        private:
            static constexpr bool outerLoopCondition
                = (domainSize % (simdSize * numWorkers)) == 0u || (simdSize * numWorkers == 1u);

            static constexpr bool innerLoopCondition = (domainSize % simdSize) == 0u || (simdSize == 1u);
        };

        /** Execute a functor for a domain of one index. */
        using Master = ForEach<Config<1, 1, 1>>;

        /** Creates an executor to iterate over the given index domain.
         *
         * The executor is invoking a functor foreach index in the domain. @see ForEach
         * @see ForEach to got more information about synchronization behaviors.
         *
         * @tparam T_domainSize number of indices in the domain
         * @tparam T_numWorkers number of worker working on @p T_domainSize
         * @tparam T_simdSize SIMD width
         *
         * @param workerIdx worker index
         * @return ForEach executor which can be invoked with a functor as argument.
         */
        template<uint32_t T_domainSize, uint32_t T_numWorkers, uint32_t T_simdSize = 1u>
        HDINLINE auto makeForEach(uint32_t const workerIdx)
        {
            return ForEach<Config<T_domainSize, T_numWorkers, T_simdSize>>(workerIdx);
        }

        /** Creates an executor with a domain of one index.
         *
         * @attention It is not defined which worker is invoking the functor.
         *
         * If the executor is invoked with a functor multiple times it is guaranteed that the same worker is performing
         * the execution.
         *
         * @param workerIdx worker index
         * @return ForEach executor which can be invoked with a functor as argument.
         */
        HDINLINE auto makeMaster(uint32_t const workerIdx)
        {
            return Master(workerIdx);
        }

    } // namespace lockstep
} // namespace pmacc
