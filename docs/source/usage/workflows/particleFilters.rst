.. _usage-workflows-particleFilters:

Particle Filters
----------------

.. sectionauthor:: Axel Huebl, Sergei Bastrakov

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
               float_X const normMom = pmacc::math::l2norm( mom );

               if( normMom > float_X( 0. ) )
               {
                   /* place detector in y direction, "infinite distance" to target,
                    * and five degree opening angle
                    */
                   constexpr float_X openingAngle = 5.0 * PI / 180.;
                   float_X const dotP = mom.y() / normMom;
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

.. note::

    User defined filter functors must be wrapped to fit the general filter interface. This can be done using wrappers like ``generic::Free<T_UserFunctor>`` and ``generic::FreeTotalCellOffset<T_UserFunctor>``

and add ``ParticlesForwardPinhole`` to the ``AllParticleFilters`` list:

.. code:: cpp

       using AllParticleFilters = MakeSeq_t<
           All,
           ParticlesForwardPinhole
       >;

   } // namespace filter
   } // namespace particles
   } // namespace picongpu

Filtering Based on Global Position
""""""""""""""""""""""""""""""""""

A particle only stores its relative position inside the cell.
Thus, with a simple filter like one shown above, it is not possible to get `global position <https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions>`_ of a particle.
However, there are helper wrapper filters that provide such information in addition to the particle data.

For a special case of slicing along one axis this is a simple existing filter that only needs to be parametrized:

.. code:: cpp

   namespace picongpu
   {
   namespace particles
   {
   namespace filter
   {
       namespace detail
       {
           //! Parameters to be used with RelativeGlobalDomainPosition, change the values inside
           struct SliceParam
           {
               // Lower bound in relative coordinates: global domain is [0.0, 1.0]
               static constexpr float_X lowerBound = 0.55_X;

               // Upper bound in relative coordinates
               static constexpr float_X upperBound = 0.6_X;

               // Axis: x = 0; y= 1; z = 2
               static constexpr uint32_t dimension = 0;

               // Text name of the filter, will be used in .cfg file
               static constexpr char const* name = "slice";
           };

           //! Use the existing RelativeGlobalDomainPosition filter with our parameters
           using Slice = RelativeGlobalDomainPosition<SliceParam>;
       }

and add ``detail::Slice`` to the ``AllParticleFilters`` list:

.. code:: cpp

       using AllParticleFilters = MakeSeq_t<
           All,
           detail::Slice
       >;

   } // namespace filter
   } // namespace particles
   } // namespace picongpu

For a more general case of filtering based on cell index (possibly combined with other particle properties) use the following pattern:

.. code:: cpp

   namespace picongpu
   {
   namespace particles
   {
   namespace filter
   {
       namespace detail
       {
           struct AreaFilter
           {
               static constexpr char const* name = "areaFilter";

               template<typename T_Particle>
               HDINLINE bool operator()(
                   DataSpace<simDim> const totalCellOffset,
                   T_Particle const & particle
               )
               {
                   /* Here totalCellOffset is the cell index of the particle in the total coordinate system.
                    * So we can define conditions based on both cell index and other particle data.
                    */
                   return (totalCellOffset.x() >= 10) && (particle[momentum_].x() < 0.0_X);
                }
            };

            //! Wrap AreaFilter so that it fits the general filter interface
            using Area = generic::FreeTotalCellOffset<AreaFilter>;
       }

and add ``detail::Area`` to the ``AllParticleFilters`` list:

.. code:: cpp

       using AllParticleFilters = MakeSeq_t<
           All,
           detail::Area
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
