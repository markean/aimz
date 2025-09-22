aimz: Scalable probabilistic impact modeling
********************************************

**Version**: |version|

**Useful links**:
`Home <https://aimz.readthedocs.io/>`__ |
`Code Repository <https://github.com/markean/aimz>`__ |
`Issues <https://github.com/markean/aimz/issues>`__ |
`Releases <https://github.com/markean/aimz/releases>`__

**aimz** is a Python library for flexible and scalable probabilistic impact modeling to assess the effects of interventions on outcomes of interest.
Designed to work with user-defined models with probabilistic primitives, the library builds on `NumPyro <https://num.pyro.ai/en/stable/>`_, `JAX <https://jax.readthedocs.io/en/latest/>`_, `Xarray <https://xarray.dev/>`_, and `Zarr <https://zarr.readthedocs.io/en/stable/>`_ to enable efficient inference workflows.

Features
--------

- An intuitive API that combines ease of use from ML frameworks with the flexibility of probabilistic modeling.
- Scalable computation via parallelism and distributed data processing—no manual orchestration required.
- Variational inference as the primary inference engine, supporting custom optimization strategies and results.
- Support for interventional causal inference for modeling counterfactuals and causal relations.
- MLflow integration for experiment tracking and model management.

Workflow
--------

1. Outline the model, considering the data generating process, latent variables, and causal relationships, if any.
2. Translate the model into a **kernel** (i.e., a function) using NumPyro and JAX.
3. Integrate the kernel into the provided API to train the model and perform inference.

Navigation
----------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :class-card: intro-card
      :link: getting_started/index
      :link-type: doc

      Begin with installation and a brief overview.

   .. grid-item-card:: User Guide
      :class-card: intro-card
      :link: user_guide/index
      :link-type: doc

      Explore tutorials and practical examples.

   .. grid-item-card:: Frequently Asked Questions
      :class-card: intro-card
      :link: faq
      :link-type: doc

      Browse common questions and troubleshooting tips.

   .. grid-item-card:: API Reference
      :class-card: intro-card
      :link: api
      :link-type: doc

      Browse the documentation for all public functions and classes.

   .. grid-item-card:: Development
      :class-card: intro-card
      :link: development/index
      :link-type: doc

      Access the development documentation and guidelines.

   .. grid-item-card:: Changelog
      :class-card: intro-card
      :link: changelog
      :link-type: doc

      View the changelog for release notes and version history.


.. toctree::
   :hidden:
   :maxdepth: 1

   Getting Started <getting_started/index>
   User Guide <user_guide/index>
   FAQs <faq>
   API Reference <api>
   Development <development/index>
   Changelog <changelog>
