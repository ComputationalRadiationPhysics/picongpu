#pragma once

#include <pmacc/functor/Call.hpp>
#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace picongpu {
    namespace particles {
        namespace pypicongpu {
            /**
             * functor for initpipeline: does nothing
             *
             * Does *NOT* have an operator(), b/c should not be called anyways.
             * (Which is ensured via template specialization of the struct for handling init operations pmacc::functor::Call below.)
             *
             * Background: Code generation creates trailing commas, this functor "catches" the final trailing comma (i.e. prevents a syntax error).
             *
             * NOP: "no operation" (from assembly)
             */
            struct nop
            {
                // intentionally left blank
            };
        } // namespace pypicongpu
    } // namspace particles
} // namespace picongpu

namespace pmacc {
    namespace functor {
        /**
         * specialization for pypicongpu nop functor
         *
         * Ensure that pypicongpu::nop initpipeline functor is not even scheduled to GPU (as functors typically are)
         */
        template <>
        struct Call<picongpu::particles::pypicongpu::nop>
        {
            HINLINE void operator()(const uint32_t currentStep)
            {
                // do nothing!
                // (default for pmacc Call: schedule given functor to GPU)
            }
        };
    } // namespace functor
} // namespace pmacc
