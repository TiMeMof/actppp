export MUJOCO_GL=egl
python3 imitate_episodes.py --eval --task_name sim_transfer_cube_ee --ckpt_dir ckpt/act2 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 1000 --lr 1e-5 --seed 0 --onscreen_render --use_ee 
