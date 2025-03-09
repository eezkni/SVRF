### nerf
#### Train
python autotask_final.py -g "0 1 2 3"  --configname syn
#### Train and Eval
python autotask_final.py -g "0 1 2 3"  --configname syn --eval
#### Eval only
python autotask_eval_only.py -g "0 1 2 3"  --configname syn

### nsvf
#### Train
python autotask_final.py -g "0 1 2 3"  --configname nsvf --dataset nsvf
#### Train and Eval
python autotask_final.py -g "0 1 2 3"  --configname nsvf --dataset nsvf --eval
#### Eval only
python autotask_eval_only.py -g "0 1 2 3"  --configname nsvf --dataset nsvf

## BlendedMVS
### Train
python autotask_final.py -g "0 1 2 3"  --configname mvs --dataset mvs
### Train and Eval
python autotask_final.py -g "0 1 2 3"  --configname mvs --dataset mvs --eval
#### Eval only
python autotask_eval_only.py -g "0 1 2 3"  --configname mvs --dataset mvs

### T&T
#### Train
python autotask_final.py -g "0 1 2 3"  --configname tnt --dataset tnt
#### ....