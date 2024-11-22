import numpy as np


def iscolumn(x) :
	if type(x) is not np.ndarray :
		return False
	if x.shape[0] > 0 and x.shape[1] == 1 :
		return True
	else :
		return False
