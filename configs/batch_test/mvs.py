_base_ = '../default.py'

expname = 'mvs_Character'
basedir = './logs/blended_mvs'

data = dict(
    # datadir='/bfs/HoloResearch/NeRFData/Synthetic_NSVF/Bike',
    dataset_type='blendedmvs',
    inverse_y=True,
    white_bkgd=True,
)

