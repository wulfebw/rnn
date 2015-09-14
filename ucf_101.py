"""
Pylearn2 compliant UCF-101 dataset.
"""

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class UCF_101(DenseDesignMatrix):
	"""
	:description: A pylearn2 compliant dataset for UCF_101. This is dataset contains videos,
		and as such, the topological view of this dataset is (examples, frames, rows, cols, channels).
	"""
	pass