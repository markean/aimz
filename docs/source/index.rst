aimz: Scalable probabilistic impact modeling
********************************************

**Version**: |version|

**Useful links**:
`Home <https://aimz.readthedocs.io/>`__ |
`Code Repository <https://github.com/markean/aimz>`__ |
`Issues <https://github.com/markean/aimz/issues>`__ |
`Releases <https://github.com/markean/aimz/releases>`__

aimz is a Python library for scalable probabilistic impact modeling—estimating how interventions affect outcomes while quantifying uncertainty. It provides an intuitive interface for fitting Bayesian models, drawing posterior samples, generating large-scale posterior predictive simulations, and estimating intervention effects with minimal boilerplate.

* **Flexible model specification:** Built on `NumPyro <https://num.pyro.ai/en/stable/>`_ and `JAX <https://jax.readthedocs.io/en/latest/>`_, bring a NumPyro model as a "kernel"—aimz does not enforce a fixed architecture.
* **Stochastic variational inference and MCMC:** Supports both SVI (including minibatch) and MCMC sampling through NumPyro's inference algorithms.
* **Scalable predictive sampling:** JIT-compiled, sharded sampling streams results to chunked `Zarr <https://zarr.readthedocs.io/en/stable/>`_ stores, enabling large-scale posterior predictive simulations.
* **Structured outputs:** Predictions, samples, and effect estimates are materialized as `Xarray <https://xarray.dev/>`_ objects backed by Zarr, integrating cleanly with the scientific Python ecosystem.
* **Intervention handling:** Specify interventions declaratively and estimate effects from posterior predictive distributions.
* **Experiment tracking:** `MLflow <https://mlflow.org/>`_ integration for logging runs, parameters, metrics, and model artifacts with full lineage.


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
