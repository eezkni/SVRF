### Example: NeRF_Synthetic
#### Train
python run.py --config configs/nerf/lego.py --render_test --render_fine --importance_prune 0.999 

#### Eval
python run.py --config configs/nerf/lego.py --render_only --render_test \
                                            --render_fine --eval_ssim --eval_lpips_vgg

#### Render video
python run.py --config configs/nerf/lego.py --render_only --render_video --render_video_factor 4

### Example: NSVF_Synthetic
#### Train
python run.py --config configs/nsvf/Bike.py --render_test --render_fine --importance_prune 0.999 

#### Eval
python run.py --config configs/nsvf/Bike.py --render_only --render_test \
                                            --render_fine --eval_ssim --eval_lpips_vgg

#### Render video
python run.py --config configs/nsvf/Bike.py --render_only --render_video --render_video_factor 4

#### ....