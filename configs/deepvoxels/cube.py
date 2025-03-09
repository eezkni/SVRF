_base_ = '../default.py'

expname = 'dvgo_cube'
basedir = './logs/deepvoxels'

data = dict(
    datadir='/root/data1/ypq/data/DeepVoxels/',
    dataset_type='deepvoxels',
    scene='cube',
    white_bkgd=True,
)

