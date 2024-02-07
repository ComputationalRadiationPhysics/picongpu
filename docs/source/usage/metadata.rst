.. _usage-metadata:

Dumping Metadata
================

Starting your simulation with

``<executable> [...] --dump-metadata [<filename>]``

will dump a `json`_ respresentation of some metadata to `<filename>`. If no `<filename>` is given, the default value
`???` is used. This feature might in a future revision default to being active.

The dumping happens immediately before the simulation starts. This implies that

 * No dynamic information about the simulation can be included (e.g. information about the state at time step 10).
 * The dumped information will represent the actual parameters used which might be different from the parameters given
   in the input files, e.g., due to :ref:`automatic adjustment<???>`.

 .. note::

  The scope of this feature is to provide a human- and machine-readable **summary of the physical content of the
  simulated conditions**. The envisioned usage is a human researcher quickly getting an overview over their simulation 
  data, an experimentalist comparing with simulation data or a database using such information for tagging, filtering 
  and searching. 

  The following related aspects are out of scope:
    
    * Reproducibility: The only faithful, feature-complete representation of the input necessary to reproduce a 
      PIConGPU simulation is the complete input directory. If a more standardised and human-readable repesentation is 
      desired, :ref:`PICMI<???>` provides access to a small subset of features.
    * Completeness: This feature is intended to be fed with well-structured information considered important by the 
      researchers. It is :ref:`customisable<???>` but the design does not allow to ensure any form of completeness with 
      appropriate maintenance effort. We therefore do not aim to describe simulations exhaustively.
    * (De-)Serialisation: We do not provide infrastructure to fully or partially reconstruct C++ objects from the 
      dumped information.


The Format
----------

The created file is a human-readable text file containing valid `json` the content of which is partially
:ref:`customisable<???>`. We do not enforce a particular format but suggest that you stick as closely as possible to the
naming conventions from :ref:`PyPIConGPU<???>` and :ref:`PICMI<???>`. By default, the output has the following
high-level structure which might be supplemented with further details as appropriate for the described elements of the 
simulation:

``???``

Customisation
-------------

Content Creation
^^^^^^^^^^^^^^^^

The main customisation point for adding and adjusting the output related to some `typename TObject`, say a Laser or the
`Simulation` object itself, is providing a specialisation for

.. code::

  template <typename TObject>
  struct picongpu::traits::GetMetadata {

    // omitted for purely compile-time types
    TObject const& obj;

    // must be static for purely compile-time types
    json description() const;

  };

for example

.. code::

   template<>
   struct picongpu::traits::GetMetadata<MyClass> {
    
     MyClass const& obj;

     json description() const {
       json result = json::object(); // always use objects and not arrays as root
       result["my"]["cool"]["runtimeValue"] = obj.runtimeValue;
       result["my"]["cool"]["compiletimeValue"] = MyClass::MyCompileTime::value;
       result["somethingElseThatSeemedImportant"] = "not necessarily derived from obj or MyClass";
       return result;
     }
   };

put anywhere where `MyClass` is known, e.g., in a pertinent `.param` file or directly below the declaration of `MyClass`
itself.

The `json` object returned from `description()` is related to the final output via a `merge_patch`_ operation but we do
not guarantee any particular order in which these are merged. So it is effectively the responsibility of the programmer
to make sure that no metadata entries overwrite each other.

These external classes might run into access restrictions when attempting to dump `private` or `protected` members.
These can be circumvented in three ways: 

1. If `MyClass` already implements a `.metadata()` method, it might already provide the necessary information through
   that interface, e.g.

   .. code::
      
      
      template<>
      struct picongpu::traits::GetMetadata<MyClass> {
       
        MyClass const& obj;

        json description() const {
          json result = obj.metadata();
          result["adjust"]["to"]["your"]["liking"] = obj.moreToDump;
          return result;
        }
      };

  This is the preferred way of handling this situation (if applicable). The default implementation of 
  `picongpu::traits::GetMetadata` forwards to such `.metadata()` methods anyway.

2. Declare `picongpu::traits::GetMetadata<MyClass` a friend of `MyClass`,
   i.e.

   .. code::
   
      class MyClass {
        friend picongpu::traits::GetMetadata<MyClass>;
        // ...
      }

   This way is minimally invasive and preferred if your change is only applicable to your personal situation and is 
   not intended to land into mainline.

3. Implement/adjust the `.metadata()` member function of `MyClass`

   .. code::
      
      class MyClass {
        // ...
        
        json metadata() const {
          // here you have all access you could possibly have
        }

        // ..
      }

   This method is preferred if your change is general enough to make it into the mainline. If so, you are invited to
   :ref:`open a pull request<???>`. It is also the approach used to provide you with default implementations to build
   upon.

Content Registration
^^^^^^^^^^^^^^^^^^^^

If you are not only adjusting existing output but instead you are adding metadata to a class that did not report any in
the past, this class must register itself **before the simulation starts**. Anything that experiences some form of
initialisation at runtime, e.g., :ref:`plugins <???>` should register themselves after their initialisation. To stick
with the example, a plugin could implement

.. code::
   void pluginLoad() {
     // ...

     registerMetadata(\*this);
   }

Classes that only affect compile-time aspects of the program need to be registered in
`include/picongpu/param/metadata.param` by extending the compile-time list `MetadataRegisteredAtCT`. Remember: Their
specialisation of `picongpu::traits::GetMetadata` does not hold a reference and must have a static `description` method.

Classes that get instantiated within a running simulation (and not in the initialisation phase) cannot be included
(because they are dynamic information, see above) unless their exact state could be forseen at compile time in which
case they can be handled exactly as compile-time-only classes.

.. _json: https://www.json.org
.. _merge_patch: https://datatracker.ietf.org/doc/html/rfc7396
