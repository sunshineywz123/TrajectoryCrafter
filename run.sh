# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'gradual' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \



python -m ptvsd --host 127.0.0.1 --port 5691 inference.py \
    --video_path './test/videos/p7.mp4' \
    --stride 2 \
    --out_dir experiments \
    --radius_scale 1 \
    --camera 'target' \
    --mode 'bullet' \
    --mask \
    --target_pose 0 -30 0.3 0 0 \
    --traj_txt 'test/trajs/loop2.txt' \
