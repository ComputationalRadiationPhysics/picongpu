.. _usage-index:

Introduction
============

.. sectionauthor:: Julian Lenz

The low-level C++ interface is the original interface of PIConGPU.
It consists of writing C++ header files, so-called ``.param`` files, and
runtime configuration files, so-called ``.cfg`` files, directly.
New users are strongly advised to use the
:ref:`the modern Python interface <python_package/foundations/index:Foundations>`.
From its perspective,
the C++ interface is an intermediate representation of the input
and should be considered an implementation detail.
This chapter is aimed at developers, experts and long-time users
having yet to transition to the new interface.

This chapter
------------

The chapter is organized into the following sections:

* :ref:`Core Concepts & First Run <usage/first_run:Core Concepts & First Run>`
* :ref:`Generating Input <usage/input:Generating Input>`
* :ref:`Recipes & Ready-Made Sets <usage/recipes:Recipes & Ready-Made Sets>`
* :ref:`Expert & Site-Specific <usage/expert:Expert & Site-Specific>`
