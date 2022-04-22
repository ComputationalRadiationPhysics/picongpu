Intro
=====

This chapter describes the internals of the PICMI translation (and subsequently the internal representation PyPIConGPU).

.. warning::

   This section is aimed at developers.
   It describes how the `PICMI user interface <TODO>`__ is implemented.

   For the PICMI in PIConGPU user documentation, see `here <TODO>`__.
   For the PICMI standard (upstream) definition, see `here <https://picmi-standard.github.io/>`__.

If you read this documentation like a book carry on in the order of the table of contents.
However, the recommended way is to also review the implementation side-by-side to this documentation:

To get started familiarize yourself with the :doc:`translation process <translation>` and skim how the :doc:`testing approach<testing>` works.
Read the :doc:`FAQ <faq>` for some more fundamental questions.
After looking at some examples of implementations yourself,
continue with :doc:`running <running>`,
the :doc:`general notes on the implementation <misc>`,
and read the notes on :doc:`how to write schemas<howto/schema>`.
If you have to work with species also read :doc:`their section <species>`.
(Note: Species are by far the most complex issue in PyPIConGPU,
make sure you understand the fundamentals before working on them.)
