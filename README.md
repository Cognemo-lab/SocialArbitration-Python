# SocialArbitration-Python
This is the Python version of the winning computational model published in the paper "Altered Perception of Environmental Volatility During Social Learning in Emerging Psychosis" by Hauke et al., published in the Computational Psychiatry Journal (DOI: 10.5334/cpsy.95).

Members of the project
----------------------
Daniel J. Hauke, Andreea O. Diaconescu

Summary
----------
- 1) Implement [`tapas_fitModel()`](matlab/code_inversion/tapas_fitModel.m) function in python. Done: [`tapas_fitModel()`](python/HGF/code_inversion/tapas_fitModel.py).
- 2) Implement [`tapas_simModel()`](matlab/code_inversion/tapas_simModel.m) function in python
- 3) Implement plotting functions (e.g. [`tapas_fit_plotCorr`](matlab/code_inversion/tapas_fit_plotCorr.m), [`plot_hgf_binary_traj`](matlab/code_model1/prc_model/plot_hgf_binary_traj.m), [`tapas_hgf_binary_plotTraj`](matlab/code_inversion/tapas_hgf_binary_plotTraj.m), [`tapas_fit_plotResidualDiagnostics`](matlab/code_inversion/tapas_fit_plotResidualDiagnostics.m))


Other important information
---------------------------
- [`test_model_inversion`](matlab/test_model_inversion.m) will allow you to reproduce the results of the test cases in Matlab, the python implementation should give the same or at least very similar results up to numerical errors
- [`tapas_hgf_binary_plotTraj`](matlab/code_inversion/tapas_hgf_binary_plotTraj.m) and [`tapas_fit_plotResidualDiagnostics`](matlab/code_inversion/tapas_fit_plotResidualDiagnostics.m) will not work with `model1`, to translate and test these functions I will provide another model later on
- Important: The modular structure (especially of the model code) needs to be preserved in the translation to facilitate implementation of new models/optimization algorithms later on.

