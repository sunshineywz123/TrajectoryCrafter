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



# python -m ptvsd --host 127.0.0.1 --port 5691 inference.py \
#     --video_path '/nas/datasets/DeepStab/unstable/1.avi' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'gradual' \
#     --mask \
#     --target_pose 0 0 0 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
#     --video_length 49

# python -m ptvsd --host 127.0.0.1 --port 5691 inference.py \
#     --video_path '/nas/datasets/DeepStab/unstable/1.avi' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'gradual' \
#     --mask \
#     --target_pose 0 0 0 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
#     --video_length 49

# python -m ptvsd --host 127.0.0.1 --port 5691 inference.py \
#     --video_path '/nas/datasets/DAVIS/JPEGImages/1080p/breakdance.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'bullet' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
#     --video_length 49
export CUDA_VISIBLE_DEVICES=0
debug=false
in_server=true
if [ $debug = true ]; then
    python -m ptvsd --host 127.0.0.1 --port 5692 inference.py \
        --video_path '/nas/datasets/DAVIS/JPEGImages/1080p/rollerblade.mp4' \
        --stride 2 \
        --out_dir experiments \
        --radius_scale 1 \
        --camera 'target' \
        --mode 'gradual' \
        --mask \
        --target_pose 0 -30 0.3 0 0 \
        --traj_txt 'test/trajs/loop2.txt' \
        --video_length 49 \
        --sample_size 512 512\
        --debug \
        --start_frame 49 \
        --in_server $in_server
else    
    for i in {0..150..49}
    do
        python inference.py \
            --video_path '/nas/datasets/DAVIS/JPEGImages/1080p/rollerblade.mp4' \
            --stride 2 \
            --out_dir experiments \
            --radius_scale 1 \
            --camera 'target' \
            --mode 'gradual' \
            --mask \
            --target_pose 0 -30 0.3 0 0 \
            --traj_txt 'test/trajs/loop2.txt' \
            --video_length 49 \
            --sample_size 512 512 \
            --start_frame $i \
            --in_server $in_server
    done
fi
# python inference.py \
#     --video_path '/nas/datasets/DAVIS/JPEGImages/1080p/rollerblade.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'gradual' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
#     --video_length 35 \
#     --sample_size 512 384

# python inference.py \
#     --video_path '/nas/datasets/DAVIS/JPEGImages/1080p/rollerblade.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'gradual' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
#     --video_length 35
