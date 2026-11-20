.. _usage-index:

Introduction
============

.. sectionauthor:: Julian J. Lenz

The legacy C++ interface is the original interface of PIConGPU.
It consists of writing C++ header files, so-called ``.param`` files, and
runtime configuration files, so-called ``.cfg`` files, directly.
From the perspective of [the modern Python interface](Using PIConGPU (PICMI)),
it is an intermediate representation of the input
and should be considered an implementation detail.
Normal users should find no reason to interact with this interface!
This chapter is aimed at developers, experts and long-time users
having yet to transition to the new interface
and is pertained mostly as historical documents.
