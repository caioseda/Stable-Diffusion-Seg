nohup python -u main.py --base configs/SDSeg/vfss-inca.yaml -t --gpus 0, --name experiment_old_vfss > nohup/experiment-old-vfss.log 2>&1 &
nohup python -u main.py --base configs/SDSeg/vfss-new-inca.yaml -t --gpus 0, --name experiment_new_vfss_inca > nohup/experiment-new-vfss.log 2>&1 &

tensorboard --logdir logs --port 6006 --host 0.0.0.0