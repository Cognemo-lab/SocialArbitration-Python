import scipy.io
from HGF.code_inversion.tapas_fitModel import tapas_fitModel
from HGF.code_inversion.tapas_quasinewton_optim_config import tapas_quasinewton_optim_config
from HGF.code_model1.prc_model.MS9_dmpad_hgf_ar1_lvl3_config import MS9_dmpad_hgf_ar1_lvl3_config
from HGF.code_model1.obs_model.MS9_dmpad_constant_voltemp_exp_config import MS9_dmpad_constant_voltemp_exp_config


# Load data
mat = scipy.io.loadmat('../test_cases/test_case_1_data.mat')
data = mat['data']

# Test model inversion
est = tapas_fitModel(data['y'][0][0], data['input_u'][0][0],
		c_prc = MS9_dmpad_hgf_ar1_lvl3_config,
		c_obs = MS9_dmpad_constant_voltemp_exp_config,
		c_opt = tapas_quasinewton_optim_config)
