_base_ = '../default.py'

expname = 'dvgo_vase'
basedir = './logs/deepvoxels'

data = dict(
    datadir='/root/data1/ypq/data/DeepVoxels/',
    dataset_type='deepvoxels',
    scene='vase',
    white_bkgd=True,
)

