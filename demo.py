import gc
import os

import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transformers import (AutoProcessor, Blip2ForConditionalGeneration,
                          T5EncoderModel)

from adjust_pose import orient_camera_to_centers
from calculate_centers import calculate_image_center, calculate_mask_center
from camera_rotation import calculate_camera_transformation
from models.autoencoder_magvit import AutoencoderKLCogVideoX
from models.crosstransformer3d import CrossTransformer3DModel
from models.infer import DepthCrafterDemo
from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
from models.utils import *
from pointcloud_processing import process_point_cloud
import cv2

from rotation_interpolation import slerp_rotation_matrix
# 四元数格式 [x,y,z,w]
# quat = [0.707, 0, 0.707, 0]
def quat_to_matrix(quat):
    rotation_matrix = R.from_quat(quat).as_matrix()  # 生成3x3旋转矩阵
    return rotation_matrix

class TrajCrafter:
    def __init__(self, opts, gradio=False):
        self.funwarp = Warper(device=opts.device)
        # self.depth_estimater = VDADemo(pre_train_path=opts.pre_train_path_vda,device=opts.device)
        self.depth_estimater = DepthCrafterDemo(
            unet_path=opts.unet_path,
            pre_train_path=opts.pre_train_path,
            cpu_offload=opts.cpu_offload,
            device=opts.device,
        )
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(
            opts.blip_path, torch_dtype=torch.float16
        ).to(opts.device)
        self.setup_diffusion(opts)
        if gradio:
            self.opts = opts

    def get_monst3r_dataset(self, opts, path):
        """Load MONST3R dataset from given path.
        
        Args:
            opts: Options object containing configuration
            path: Path to dataset directory
            
        Returns:
            tuple: (frames, depths, enlarged_masks, K, R_matrix, t)
        """
        frames = []
        depths = []
        enlarged_masks = []
        for i in range(opts.video_length):
            frames.append(Image.open(path + 'frame_{:04d}.png'.format(i)))
            depths.append(np.load(path + 'frame_{:04d}.npy'.format(i)))
            enlarged_masks.append(Image.open(path + 'enlarged_dynamic_mask_{:d}.png'.format(i)))
        
        original_frames = np.array(frames)
        frames = original_frames.transpose(0, 3, 1, 2)
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames)
        frames = frames.to(opts.device) * 2.0 - 1.0
        
        depths = torch.from_numpy(np.array(depths).reshape(opts.video_length, 1, depths[0].shape[0], depths[0].shape[1]))
        depths = depths.to(opts.device)
        
        enlarged_masks = torch.from_numpy(np.array(enlarged_masks))
        enlarged_masks = enlarged_masks.to(opts.device).unsqueeze(1)

        if frames.shape[0] != opts.video_length:
            opts.video_length = frames.shape[0]
        assert frames.shape[0] == opts.video_length

        num_frames = opts.video_length
        intrinsic = np.loadtxt(path + 'pred_intrinsics.txt')
        poses = np.loadtxt(path + 'pred_traj.txt')
        K = torch.from_numpy(intrinsic[:num_frames, :]).reshape(num_frames, 3, 3)
        R_matrix = quat_to_matrix(np.concatenate([poses[:num_frames, 5:], poses[:num_frames, 4:5]], -1))
        t = poses[:num_frames, 1:4]
        
        return original_frames,frames, depths, K, R_matrix, t,enlarged_masks
    def get_vggt_dataset(self, opts, path,start_frame):
        """Load MONST3R dataset from given path.
        
        Args:
            opts: Options object containing configuration
            path: Path to dataset directory
            start_frame: Start frame index
        Returns:
            tuple: (frames, depths, enlarged_masks, K, R_matrix, t)
        """
        frames = []
        depths = []
        enlarged_masks = []
        num_frames = opts.video_length
        intrinsic = np.loadtxt(path + 'pred_intrinsics.txt')[start_frame:start_frame+num_frames, :]
        poses = np.loadtxt(path + 'pred_traj.txt')[start_frame:start_frame+num_frames, :]
        for i in range(num_frames):
            frames.append(Image.open(path + 'crop_images/frame_{:06d}.jpg'.format(i+start_frame)))
            enlarged_masks.append(Image.open(path + 'images_sky_masks/frame_{:06d}.jpg'.format(i+start_frame)))
        
        original_frames = np.array(frames)
        frames = original_frames.transpose(0, 3, 1, 2)
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames)
        frames = frames.to(opts.device) * 2.0 - 1.0
        

        
        enlarged_masks = torch.from_numpy(np.array(enlarged_masks))
        enlarged_masks = enlarged_masks.to(opts.device).unsqueeze(1)

        if frames.shape[0] != opts.video_length:
            opts.video_length = frames.shape[0]
        assert frames.shape[0] == opts.video_length


        K = torch.from_numpy(intrinsic[:num_frames, :]).reshape(num_frames, 3, 3)
        # R_matrix = quat_to_matrix(np.concatenate([poses[:num_frames, 5:], poses[:num_frames, 4:5]], -1))
        # t = poses[:num_frames, 1:4]
        extrinsics_cam = poses[:num_frames,:].reshape(-1,3,4)
        extrinsics_cam_homo = np.concatenate([extrinsics_cam,np.zeros((extrinsics_cam.shape[0],1,4))],axis=1)
        extrinsics_cam_homo[:,-1,3] = 1.0
        c2w_matrix = np.linalg.inv(extrinsics_cam_homo)
        R_matrix = c2w_matrix[:,:3,:3]
        t = c2w_matrix[:,:3,3]
        return original_frames,frames, depths, K, R_matrix, t,enlarged_masks
    def infer_gradual(self, opts):
        if 1:
            path = '/nas/users/yuanweizhong/TrajectoryCrafter/gugong/'
            # original_frames,frames, depths, K, R_matrix, t,enlarged_masks = self.get_monst3r_dataset(opts, path)
            start_frame = 0
            original_frames,frames, depths, K, R_matrix, t,enlarged_masks = self.get_vggt_dataset(opts, path,start_frame)
            scale = 1
            num_frames=opts.video_length
            pose_s_in = torch.from_numpy(np.eye(4).astype(np.float32)).repeat(num_frames, 1, 1)

            R_start=torch.from_numpy(R_matrix).float()[0]
            R_end=torch.from_numpy(R_matrix).float()[-1]

            
            R_interp=slerp_rotation_matrix(R_start, R_end, num_frames-1)
            pose_s_in[:, :3, :3] = scale*torch.from_numpy(np.array(R_interp)).float()
            # Linear interpolation for translation between first and last frame
            t_start = torch.from_numpy(t).float()[0]
            t_end = torch.from_numpy(t).float()[-1]
            t_interp = torch.stack([
                torch.linspace(t_start[i], t_end[i], num_frames) 
                for i in range(3)
            ], dim=1)
            pose_s_in[:, :3, 3] = scale * t_interp
            #pose_s_in 是 c2w
            ai_follow_flag=False
            if ai_follow_flag:
                pose_s, pose_t = ai_follow(enlarged_masks, original_frames, K, pose_s_in, opts, num_frames)
            else:
                # 设置源相机姿态
                # pose_s = pose_s_in[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
                # pose_t =pose_s_in[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
                pose_s = pose_s_in
                pose_t = torch.eye(4).to(pose_s_in.device).to(pose_s_in.dtype).repeat(num_frames,1,1)
                
            if 0:
                # 创建4x4的单位矩阵并复制num_frames份
                transfor_matrix=torch.eye(4).repeat(num_frames,1,1)
                # 将旋转矩阵填充到transform_matrix的左上3x3部分
                transfor_matrix[:,:3,:3]=torch.from_numpy(rotation_matrix).float()
                # 获取锚点帧的相机姿态并复制num_frames份
                pose_t =pose_s[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
                # 将变换矩阵与锚点帧的相机姿态相乘得到变换后的相机姿态
                pose_s_trans =transfor_matrix@ (pose_s[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1))
                # pose_t[:,:,:] =pose_s_trans[:,:,:]

                pose_t=pose_s_trans


            #pose_t = torch.linalg.inv(transfor_matrix)@ (pose_s[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1))

            #这边的输入需要的是c2w
            # pose_s=pose_s_in
            # pose_t = pose_s[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)

            # 初始化用于存储扭曲图像和掩码的列表
            warped_images = []
            masks = []

            # 遍历每一帧，进行扭曲处理
            for i in tqdm(range(opts.video_length)):
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[i : i + 1],
                    None,
                    depths[i : i + 1],
                    pose_s[i : i + 1],
                    pose_t[i : i + 1],
                    K[i : i + 1],
                    None,
                    opts.mask,
                    twice=False,
                    load_pcd=True,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)

            # 将扭曲图像和掩码转换为适合保存的格式
            cond_video = (torch.cat(warped_images) + 1.0) / 2.0
            cond_masks = torch.cat(masks)

            # 调整帧、扭曲视频和掩码的大小
            interpolated_frames = F.interpolate(
                frames, size=opts.sample_size, mode='bilinear', align_corners=False
            )
            cond_video = F.interpolate(
                cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
            )
            cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')

            # 保存原始帧、扭曲视频和掩码为视频文件
            save_video(
                (interpolated_frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
                os.path.join(opts.save_dir, 'input.mp4'),
                fps=opts.fps,
            )
            save_video(
                cond_video.permute(0, 2, 3, 1),
                os.path.join(opts.save_dir, 'render.mp4'),
                fps=opts.fps,
            )
            save_video(
                cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
                os.path.join(opts.save_dir, 'mask.mp4'),
                fps=opts.fps,
            )

        # assert(0)
        # import ipdb;ipdb.set_trace()
        
        # assert(0)
    
        # save_path = opts.save_dir
        save_path = 'experiments/p7/'
        #从input.mp4 render.mp4 mask.mp4 还原 frames cond_video cond_masks
        #frames shape:(49, 3, 384, 672) type:(float32 of torch.Tensor) max: 1.0, min: -0.87829, mean: -0.10329
        vid = VideoReader(os.path.join(save_path, 'input.mp4'), ctx=cpu(0))
        frames_idx = list(range(0, len(vid), 1))
        original_frames=vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
        #original_frames shape:(49, 384, 672, 3) type:(float32 of numpy.ndarray) max: 1.0, min: 0.054902, mean: 0.44257
        frames=torch.from_numpy(original_frames).permute(0,3,1,2).to(opts.device)*2.0-1.0

        cond_vid = VideoReader(os.path.join(save_path, 'render.mp4'), ctx=cpu(0))
        cond_frames_idx = list(range(0, len(cond_vid), 1))
        original_cond_frames=cond_vid.get_batch(cond_frames_idx).asnumpy().astype("float32") / 255.0
        cond_video = torch.from_numpy(original_cond_frames).permute(0,3,1,2).to(opts.device)

        cond_masks_vid = VideoReader(os.path.join(save_path, 'mask.mp4'), ctx=cpu(0))
        cond_masks_frames_idx = list(range(0, len(cond_masks_vid), 1))
        original_cond_masks_frames=cond_masks_vid.get_batch(cond_masks_frames_idx).asnumpy().astype("float32") / 255.0
        cond_masks = torch.from_numpy(original_cond_masks_frames).permute(0,3,1,2).to(opts.device)[:,:1,:,:]

        mid_indx = min(opts.video_length // 2,24)

        prompt_frame = (frames.permute(0,2,3,1)[mid_indx].cpu().numpy()+1)/2.0
        prompt = self.get_caption(opts, prompt_frame)
        # 调试断点
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0

        # 创建随机数生成器
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        # 释放不再需要的资源
        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()

        # 在不计算梯度的情况下进行生成
        with torch.no_grad():
            # 使用管道进行生成
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks[:,0,:,:,:],
                reference=frames_ref,
            ).videos

        # 保存生成的视频
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        # 可视化选项
        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_right = sample[0].to(opts.device)
            interval = torch.ones(3, 49, 384, 30).to(opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def infer_direct(self, opts):
        opts.cut = 20
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        mid_indx = min(opts.video_length // 2,24)
        prompt = self.get_caption(opts, frames[mid_indx])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        if frames.shape[0] != opts.video_length:
            opts.video_length = frames.shape[0] 
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.cut)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            if i < opts.cut:
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[0:1],
                    None,
                    depths[0:1],
                    pose_s[0:1],
                    pose_t[i : i + 1],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
            else:
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[i - opts.cut : i - opts.cut + 1],
                    None,
                    depths[i - opts.cut : i - opts.cut + 1],
                    pose_s[0:1],
                    pose_t[-1:],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames[: opts.video_length - opts.cut].permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video[opts.cut :].permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks[opts.cut :].repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        assert(0)
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0)[opts.cut :],
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0][:, : opts.video_length - opts.cut, ...].to(
                opts.device
            )
            tensor_right = sample[0][:, opts.cut :, ...].to(opts.device)
            interval = torch.ones(3, opts.video_length - opts.cut, 384, 30).to(
                opts.device
            )
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def infer_bullet(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        mid_indx = min(opts.video_length // 2,24)
        prompt = self.get_caption(opts, frames[mid_indx])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)

        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        if frames.shape[0] != opts.video_length:
                opts.video_length = frames.shape[0] 
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[-1:],
                None,
                depths[-1:],
                pose_s[0:1],
                pose_t[i : i + 1],
                K[0:1],
                None,
                opts.mask,
                twice=False,
                load_pcd=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        # assert(0)
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, -10:, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = False
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_left_full = torch.cat(
                [tensor_left, tensor_left[:, -1:, :, :].repeat(1, 48, 1, 1)], dim=1
            )
            tensor_right = sample[0].to(opts.device)
            tensor_right_full = torch.cat(
                [tensor_left, tensor_right[:, 1:, :, :]], dim=1
            )
            interval = torch.ones(3, 49 * 2 - 1, 384, 30).to(opts.device)
            result = torch.cat((tensor_left_full, interval, tensor_right_full), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 4,
            )

    def get_caption(self, opts, image):
        image_array = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
            opts.device, torch.float16
        )
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text + opts.refine_prompt

    def get_poses(self, opts, depths, num_frames):
        """
        生成相机姿态和相机内参矩阵。

        Args:
            opts (Namespace): 包含相机轨迹生成相关参数的命名空间对象。
            depths (torch.Tensor): 深度图张量，形状为[num_frames, 1, height, width]。
            num_frames (int): 帧数。

        Returns:
            tuple: 包含锚点相机姿态、目标相机姿态和相机内参矩阵的元组。
                - pose_s (torch.Tensor): 锚点相机姿态张量，形状为[num_frames, 4, 4]。
                - pose_t (torch.Tensor): 目标相机姿态张量，形状为[num_frames, 4, 4]。
                - K (torch.Tensor): 相机内参矩阵张量，形状为[num_frames, 3, 3]。

        """
        # 计算半径
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)

        # 设置相机中心点坐标
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.

        # 计算相机内参矩阵K
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )

        # 初始化相机到世界坐标系的变换矩阵c2w_init
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )

        # 根据opts.camera的值选择生成相机轨迹的方式
        if opts.camera == 'target':
            # 从opts.target_pose中解包相机轨迹参数:旋转角度dtheta、倾斜角度dphi、半径缩放dr、x方向平移dx和y方向平移dy
            dtheta, dphi, dr, dx, dy = opts.target_pose
            # 调用generate_traj_specified函数生成指定参数的相机轨迹
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)

        # 将相机轨迹沿z轴平移radius距离
        poses[:, 2, 3] = poses[:, 2, 3] + radius

        # 生成锚点相机姿态和目标相机姿态
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses

        return pose_s, pose_t, K

    def setup_diffusion(self, opts):
        # transformer = CrossTransformer3DModel.from_pretrained_cus(opts.transformer_path).to(opts.weight_dtype)
        transformer = CrossTransformer3DModel.from_pretrained(opts.transformer_path).to(
            opts.weight_dtype
        )
        transformer.is_train_cross = False
        # transformer = transformer.to(opts.weight_dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(
            opts.model_name, subfolder="vae"
        ).to(opts.weight_dtype)
        text_encoder = T5EncoderModel.from_pretrained(
            opts.model_name, subfolder="text_encoder", torch_dtype=opts.weight_dtype
        )
        # Get Scheduler
        Choosen_Scheduler = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM_Cog": CogVideoXDDIMScheduler,
            "DDIM_Origin": DDIMScheduler,
        }[opts.sampler_name]
        scheduler = Choosen_Scheduler.from_pretrained(
            opts.model_name, subfolder="scheduler"
        )

        self.pipeline = TrajCrafter_Pipeline.from_pretrained(
            opts.model_name,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=opts.weight_dtype,
        )

        if opts.low_gpu_memory_mode:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()

    def run_gradio(self, input_video, stride, radius_scale, pose, steps, seed):
        frames = read_video_frames(
            input_video, self.opts.video_length, stride, self.opts.max_res
        )
        prompt = self.get_caption(self.opts, frames[self.opts.video_length // 2])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            self.opts.near,
            self.opts.far,
            self.opts.depth_inference_steps,
            self.opts.depth_guidance_scale,
            window_size=self.opts.window_size,
            overlap=self.opts.overlap,
        ).to(self.opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.opts.device) * 2.0
            - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        num_frames = frames.shape[0]
        assert num_frames == self.opts.video_length
        radius_scale = float(radius_scale)
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(self.opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(self.opts.device)
            .unsqueeze(0)
        )

        # import pdb
        # pdb.set_trace()
        theta, phi, r, x, y = [float(i) for i in pose.split(';')]
        # theta,phi,r,x,y = [float(i) for i in theta.split()],[float(i)
        # for i in phi.split()],[float(i)
        # for i in r.split()],[float(i) for i in x.split()],[float(i) for i in y.split()]
        # target mode
        poses = generate_traj_specified(
            c2w_init, theta, phi, r * radius, x, y, num_frames, self.opts.device
        )
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[self.opts.anchor_idx : self.opts.anchor_idx + 1].repeat(
            num_frames, 1, 1
        )
        pose_t = poses

        warped_images = []
        masks = []
        for i in tqdm(range(self.opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[i : i + 1],
                None,
                self.opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames = F.interpolate(
            frames, size=self.opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=self.opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(
            cond_masks, size=self.opts.sample_size, mode='nearest'
        )
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(self.opts.save_dir, 'input.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'render.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'mask.mp4'),
            fps=self.opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=self.opts.device).manual_seed(seed)

        # del self.depth_estimater
        # del self.caption_processor
        # del self.captioner
        # gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=self.opts.video_length,
                negative_prompt=self.opts.negative_prompt,
                height=self.opts.sample_size[0],
                width=self.opts.sample_size[1],
                generator=generator,
                guidance_scale=self.opts.diffusion_guidance_scale,
                num_inference_steps=steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(self.opts.save_dir, 'gen.mp4'),
            fps=self.opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(self.opts.device)
            tensor_right = sample[0].to(self.opts.device)
            interval = torch.ones(3, 49, 384, 30).to(self.opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(self.opts.save_dir, 'viz.mp4'),
                fps=self.opts.fps * 2,
            )
        return os.path.join(self.opts.save_dir, 'viz.mp4')

def ai_follow(enlarged_masks, original_frames, K, pose_s_in, opts, num_frames):
    """
    根据mask中心点计算相机跟随变换。
    
    Args:
        enlarged_masks (torch.Tensor): 扩大后的mask, shape (N, 1, H, W)
        original_frames (np.ndarray): 原始帧图像, shape (N, H, W, 3)
        K (torch.Tensor): 相机内参矩阵, shape (N, 3, 3)
        pose_s_in (torch.Tensor): 输入的相机姿态, shape (N, 4, 4)
        opts: 配置选项
        num_frames (int): 帧数
        
    Returns:
        tuple: (pose_s, pose_t) 变换后的源相机姿态和目标相机姿态
    """
    # 计算 mask 的中心点
    enlarged_masks_center = calculate_mask_center(enlarged_masks)
    
    # 将列表转换为 numpy 数组
    enlarged_masks_center_array = np.array(enlarged_masks_center)
    
    # 在原始帧上可视化中心点
    for i in range(len(enlarged_masks_center)):
        cv2.circle(
            original_frames[i],
            (enlarged_masks_center[i][0], enlarged_masks_center[i][1]),
            5,
            (0, 0, 255),  # BGR 格式的红色
            -1
        )
        cv2.imwrite('debug/enlarged_masks_{:d}.png'.format(i), original_frames[i])
    
    enlarged_masks_center = enlarged_masks_center_array
    
    # 计算图像中心点
    image_center = calculate_image_center(enlarged_masks)
    image_center = np.array(image_center)
    
    # 计算相机变换矩阵
    rotation_matrix, A_transformed, A_transformed_camera = calculate_camera_transformation(
        enlarged_masks_center, image_center, K
    )
    
    # 设置源相机姿态
    pose_s = pose_s_in
    
    # 计算目标相机姿态
    pose_s_trans = pose_s[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1).clone()
    pose_t = pose_s_trans.clone()
    pose_t[:,:3,:3] = torch.from_numpy(np.linalg.inv(rotation_matrix)).float()
    
    return pose_s, pose_t
