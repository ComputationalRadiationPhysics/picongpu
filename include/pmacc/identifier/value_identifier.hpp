/* Copyright 2013-2023 Rene Widera, Pawel Ordyna
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/identifier/identifier.hpp"
#include "pmacc/particles/IdProvider.hpp"
#include "pmacc/ppFunctions.hpp"
#include "pmacc/traits/HasIdentifier.hpp"
#include "pmacc/types.hpp"

#include <string>


namespace pmacc::particles::identifier
{
    /** copy value from the source particle
     *
     * This class can be used as copyFunctor for value_identifier_func.
     */
    struct CallCopy
    {
        template<typename T_Identifier, typename T_ParticleType>
        constexpr auto operator()(T_Identifier const identifier, T_ParticleType const& srcParticle) const
        {
            return srcParticle[identifier];
        }
    };

    /** copy value or init from default value
     *
     * If the source particle contains the identifier the method copyValue() from the
     * identifier will be called else initValue() is called.
     *
     * This class can be used as deriveFunctor for value_identifier_func.
     */
    struct CallCopyOrInit
    {
        template<typename T_Worker, typename T_Identifier, typename T_ParticleType>
        constexpr auto operator()(
            T_Worker const& worker,
            IdGenerator& idGen,
            T_Identifier const identifier,
            T_ParticleType const& srcParticle) const
        {
            if constexpr(pmacc::traits::HasIdentifier<T_ParticleType, T_Identifier>::type::value)
                return identifier.copyValue(identifier, srcParticle);
            else
                return identifier.initValue(worker, idGen);
        }
    };

    /** calls the method initValue() from the identifier
     *
     * This class can be used as deriveFunctor for value_identifier_func.
     */
    struct CallInitValue
    {
        template<typename T_Worker, typename T_Identifier, typename T_ParticleType>
        constexpr auto operator()(
            T_Worker const& worker,
            IdGenerator& idGen,
            T_Identifier const identifier,
            T_ParticleType const&) const
        {
            return identifier.initValue(worker, idGen);
        }
    };
} // namespace pmacc::particles::identifier

/** define a unique identifier with name, type and a default value
 * @param in_type type of the value
 * @param name name of identifier
 *
 * The created identifier has the following options:
 *      getValue()        - return the user defined value of value_identifier
 *      initValue(worker,IdGenerator) - return the user defined value of value_identifier_func
 *      copyValue(identifier, SourceParticle) - return the identifier from the source particle
 *      deriveValue(worker,IdGenerator, identifier, SourceParticle)
 *                        - return the derived identifier value from the source particle
 *      getName()         - return the name of the identifier
 *      ::type            - get type of the value
 *
 * @code{.cpp}
 *      value_identifier(float,length,0.0f)
 *      typedef length::type value_type; // is float
 *      value_type x = length::getValue(); //set x to 0.f
 *      printf("Identifier name: %s",length::getName()); //print Identifier name: length
 * @endcode
 *
 * to create a instance of this value_identifier you can use:
 *      `length()` or `length_`
 * @{
 */

/**
 * @param initFunctor Should be a constexpr function/lambda with arguments worker, IdGenerator. Functor must be
 * surrounded by round brackets. The functor is called at the moment where a particle is the first time created.
 * @param copyFunctor Should be a constexpr function/lambda with arguments identifier, source particle. Functor must be
 * surrounded by round brackets. The functor is called if the same particle type is copied. The identifier can be used
 * without checking the source particle type.
 * @param deriveFunctor Should be a constexpr function/lambda with arguments worker, IdGenerator, identifier, source
 * particle. Functor must be surrounded by round brackets. The functor is called when a particle is derived from
 * another. It is not guaranteed that the source particle has each identifier given into the functor.
 *
 * @code{.cpp}
 * # possible functor for initFunctor
 * [] ALPAKA_FN_ACC(auto const& worker, IdGenerator& idGen, auto const& srcParticle) { return idGen.fetchInc(worker);
 * };
 *
 * # possible functor for copyFunctor
 * [] ALPAKA_FN_ACC(auto const identifier, auto const& srcParticle) { return srcParticle[identifier]; };
 *
 * # possible functor for deriveFunctor, see pmacc::particles::identifier::CopyOrInit
 * @endcode
 */
#define value_identifier_func(in_type, name, initFunctor, copyFunctor, deriveFunctor)                                 \
    identifier(                                                                                                       \
        name, using type = in_type;                                                                                   \
                                                                                                                      \
        template<typename T_Identifier, typename T_SrcParticleType>                                                   \
        constexpr type copyValue(T_Identifier const idName, T_SrcParticleType const& srcParticle) const               \
        {                                                                                                             \
            auto const func = PMACC_REMOVE_BRACKETS copyFunctor;                                                      \
            return func(idName, srcParticle);                                                                         \
        } template<typename T_Worker, typename T_Identifier, typename T_SrcParticleType>                              \
        constexpr type deriveValue(                                                                                   \
            T_Worker const& worker,                                                                                   \
            IdGenerator& idGen,                                                                                       \
            T_Identifier const idName,                                                                                \
            T_SrcParticleType const& srcParticle) const                                                               \
        {                                                                                                             \
            auto const func = PMACC_REMOVE_BRACKETS deriveFunctor;                                                    \
            return func(worker, idGen, idName, srcParticle);                                                          \
        } template<typename T_Worker>                                                                                 \
        constexpr type initValue(T_Worker const& worker, IdGenerator& idGen) const                                    \
        {                                                                                                             \
            auto const func = PMACC_REMOVE_BRACKETS initFunctor;                                                      \
            return func(worker, idGen);                                                                               \
        } static std::string getName() { return std::string(#name); })

/** getValue() is defined constexpr
 *  @param ... user defined value of in_type (can be a constructor of a class) e.g.
 *
 * @code{.cpp}
 * float3_X::create(0._X)
 * @endcode
 * @}
 */
#define value_identifier(in_type, name, ...)                                                                          \
    identifier(                                                                                                       \
        name, using type = in_type; template<typename T_Identifier, typename T_SrcParticleType>                       \
        constexpr type copyValue(T_Identifier const idName, T_SrcParticleType const& srcParticle) const               \
        { return srcParticle[idName]; } template<typename T_Worker>                                                   \
        constexpr type initValue(T_Worker const&, IdGenerator&) const                                                 \
        { return getValue(); } template<typename T_Worker, typename T_Identifier, typename T_SrcParticleType>         \
        constexpr type deriveValue(                                                                                   \
            T_Worker const&,                                                                                          \
            IdGenerator&,                                                                                             \
            T_Identifier const idName,                                                                                \
            T_SrcParticleType const& srcParticle) const                                                               \
        { return srcParticle[idName]; } static constexpr type getValue()                                              \
        { return __VA_ARGS__; } static std::string getName() { return std::string(#name); })
