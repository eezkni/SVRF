_base_ = '../default.py'

expname = 'dvgo_Lifestyle'
basedir = './logs/nsvf_synthetic'

data = dict(
    datadir='./data/Lifestyle',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

