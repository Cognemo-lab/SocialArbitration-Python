
%% Test model inversion
load('test_cases/test_case_1_data.mat');
est = tapas_fitModel(data.y, data.input_u, ...
    'MS9_dmpad_hgf_ar1_lvl3_config', ...
    'MS9_dmpad_constant_voltemp_exp_config');


%% Test plotting
tapas_fit_plotCorr(est);
plot_hgf_binary_traj(est);


%% Test simulation
seed = 1;
sim = tapas_simModel(data.input_u,...
    'MS9_dmpad_hgf_ar1_lvl3', est.p_prc.p,...
    'MS9_dmpad_constant_voltemp_exp', est.p_obs.p, seed);
plot_hgf_binary_traj(sim);

seed = 2;
sim = tapas_simModel(data.input_u,...
    'MS9_dmpad_hgf_ar1_lvl3', est.p_prc.p,...
    'MS9_dmpad_constant_voltemp_exp', est.p_obs.p, seed);
plot_hgf_binary_traj(sim);


