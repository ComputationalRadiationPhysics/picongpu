.. _usage-workflows-particleFilters:

Particle Filters
----------------

.. sectionauthor:: Axel Huebl

A common task in both modeling, initializing and in situ processing (output) is the selection of particles of a particle species by attributes.
PIConGPU implements such selections as *particle filters*.

Particle filters are simple mappings assigning each particle of a species either ``true`` or ``false`` (ignore / filter out).
These filters can be defined in :ref:`particleFilters.param <usage-params-core>`.

Example
"""""""

Let us select particles with momentum vector within a cone with an opening angle of five degrees (pinhole):

.. code:: cpp

   namespace picongpu
   {
   namespace particles
   {
   namespace filter
   {
       struct FunctorParticlesForwardPinhole
       {
           static constexpr char const * name = "forwardPinhole";

           template< typename T_Particle >
           HDINLINE bool operator()(
               T_Particle const & particle
           )
           {
               bool result = false;
               float3_X const mom = particle[ momentum_ ];
               float_X const absMom = math::abs( mom );

               if( absMom > float_X( 0. ) )
               {
                   /* place detector in y direction, "infinite distance" to target,
                    * and five degree opening angle
                    */
                   constexpr float_X openingAngle = 5.0 * PI / 180.;
                   float_X const dotP = mom.y() / absMom;
                   float_X const degForw = math::acos( dotP );

                   if( math::abs( degForw ) <= openingAngle * float_X( 0.5 ) )
                       result = true;
               }
               return result;
           }
       };
       using ParticlesForwardPinhole = generic::Free<
          FunctorParticlesForwardPinhole
       >;

and add ``ParticlesForwardPinhole`` to the ``AllParticleFilters`` list:

.. code:: cpp

   using AllParticleFilters = MakeSeq_t<
       All,
       ParticlesForwardPinhole
   >;

   } // namespace filter
   } // namespace particles
   } // namespace picongpu

Limiting Filters to Eligible Species
""""""""""""""""""""""""""""""""""""

Besides :ref:`the list of pre-defined filters <usage-params-core-particles-filters>` with parametrization, users can also define generic, "free" implementations as shown above.
All filters are added to ``AllParticleFilters`` and then *combined with all available species* from ``VectorAllSpecies`` (see :ref:`speciesDefinition.param <usage-params-core>`).

In the case of user-defined free filters we can now check if each species in ``VectorAllSpecies`` fulfills the requirements of the filter.
That means: if one accesses specific *attributes* or *flags* of a species in a filter, they must exist or will lead to a compile error.

As an example, :ref:`probe particles <usage-workflows-probeParticles>` usually do not need a ``momentum`` attribute which would be used for an energy filter.
So they should be ignored from compilation when combining filters with particle species.

In order to exclude all species that have no ``momentum`` attribute from the ``ParticlesForwardPinhole`` filter, specialize the C++ trait ``SpeciesEligibleForSolver``.
This trait is implemented to be checked during compile time when combining filters with species:

.. code:: cpp

   // ...

   } // namespace filter

   namespace traits
   {
       template<
           typename T_Species
       >
       struct SpeciesEligibleForSolver<
           T_Species,
           filter::ParticlesForwardPinhole
       >
       {
           using type = typename pmacc::traits::HasIdentifiers<
               typename T_Species::FrameType,
               MakeSeq_t< momentum >
           >::type;
       };
   } // namespace traits
   } // namespace particles
   } // namespace picongpu
