.. currentmodule:: aimz

ImpactModel
===========

.. autosummary::
   :toctree: generated/

   ImpactModel


Attributes
----------

.. autosummary::
   :toctree: generated/

   ImpactModel.inference
   ImpactModel.kernel
   ImpactModel.kernel_spec
   ImpactModel.param_input
   ImpactModel.param_output
   ImpactModel.posterior
   ImpactModel.rng_key
   ImpactModel.temp_dir
   ImpactModel.vi_result


Inference
---------

.. autosummary::
   :toctree: generated/

   ImpactModel.train_on_batch
   ImpactModel.fit_on_batch
   ImpactModel.fit
   ImpactModel.set_posterior_sample
   ImpactModel.is_fitted
   ImpactModel.predict_on_batch
   ImpactModel.predict
   ImpactModel.log_likelihood


Explicit Sampling
-----------------

.. autosummary::
   :toctree: generated/

   ImpactModel.sample_prior_predictive_on_batch
   ImpactModel.sample_prior_predictive
   ImpactModel.sample
   ImpactModel.sample_posterior_predictive_on_batch
   ImpactModel.sample_posterior_predictive


Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   ImpactModel.estimate_effect
   ImpactModel.cleanup
   ImpactModel.cleanup_models
