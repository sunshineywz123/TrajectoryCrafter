# export CUDA_VISIBLE_DEVICES=0
# 指定要遍历的目录
directory="./test/videos/"
# 检查目录是否存在
if [ ! -d "$directory" ]; then
  echo "错误：目录 '$directory' 不存在。"
  exit 1
fi

file="./test/videos/UST-fn-RvhJwMR5S.mp4"
echo "正在处理文件：$file"
time python inference.py \
    --video_path $file \
    --stride 2 \
    --out_dir experiments \
    --radius_scale 1 \
    --camera 'target' \
    --mode 'gradual' \
    --mask \
    --target_pose 0 -30 0.3 0 0 \
    --traj_txt 'test/trajs/loop2.txt' 


# 遍历目录下的所有 .mp4 文件
# for file in $directory/*.mp4; do
#     echo "正在处理文件：$file"
#     time python inference.py \
#         --video_path $file \
#         --stride 2 \
#         --out_dir experiments \
#         --radius_scale 1 \
#         --camera 'target' \
#         --mode 'gradual' \
#         --mask \
#         --target_pose 0 -30 0.3 0 0 \
#         --traj_txt 'test/trajs/loop2.txt' 
# done

echo "处理完成。"


# python -m ptvsd --host 127.0.0.1 --port 5691 inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'gradual' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
