_base_ = '../default.py'

expname = 'dvgo_armchair'
basedir = './logs/deepvoxels'

data = dict(
    datadir='/root/data1/ypq/data/DeepVoxels/',
    dataset_type='deepvoxels',
    scene='armchair',
    white_bkgd=True,
)

