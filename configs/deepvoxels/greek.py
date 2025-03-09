_base_ = '../default.py'

expname = 'dvgo_greek'
basedir = './logs/deepvoxels'

data = dict(
    datadir='/root/data1/ypq/data/DeepVoxels/',
    dataset_type='deepvoxels',
    scene='greek',
    white_bkgd=True,
)

