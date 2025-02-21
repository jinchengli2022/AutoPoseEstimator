# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from nerf_runner import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from datareader import *
from bundlesdf.tool import *
import yaml,argparse


def run_neural_object_field(cfg, K, rgbs, depths, masks, cam_in_obs, debug=0, save_dir='/home/bowen/debug/foundationpose_bundlesdf'):
  rgbs = np.asarray(rgbs)
  depths = np.asarray(depths)
  masks = np.asarray(masks)
  cam_in_obs = np.asarray(cam_in_obs)
  glcam_in_obs = cam_in_obs@glcam_in_cvcam

  cfg['save_dir'] = save_dir
  os.makedirs(save_dir, exist_ok=True)

  for i,rgb in enumerate(rgbs):
    imageio.imwrite(f'{save_dir}/rgb_{i:07d}.png', rgb)

  sc_factor,translation,pcd_real_scale, pcd_normalized = compute_scene_bounds(None,glcam_in_obs, K, use_mask=True,base_dir=save_dir,rgbs=rgbs,depths=depths,masks=masks, eps=cfg['dbscan_eps'], min_samples=cfg['dbscan_eps_min_samples'])
  cfg['sc_factor'] = sc_factor
  cfg['translation'] = translation

  o3d.io.write_point_cloud(f'{save_dir}/pcd_normalized.ply', pcd_normalized)

  rgbs_, depths_, masks_, normal_maps,poses = preprocess_data(rgbs, depths, masks,normal_maps=None,poses=glcam_in_obs,sc_factor=cfg['sc_factor'],translation=cfg['translation'])

  nerf = NerfRunner(cfg, rgbs_, depths_, masks_, normal_maps=None, poses=poses, K=K, occ_masks=None, build_octree_pcd=pcd_normalized)
  nerf.train()    # 在train中会保存检查点，但不会进行纹理优化

  mesh = nerf.extract_mesh(isolevel=0,voxel_size=cfg['mesh_resolution'])    # 从训练好的网络中提取
  mesh = nerf.mesh_texture_from_train_images(mesh, rgbs_raw=rgbs, tex_res=1028) # 通过RGB图像添加纹理
  optimized_cvcam_in_obs,offset = get_optimized_poses_in_real_world(poses,nerf.models['pose_array'],cfg['sc_factor'],cfg['translation'])  # 获取优化后的相机位姿
  mesh = mesh_to_real_world(mesh, pose_offset=offset, translation=nerf.cfg['translation'], sc_factor=nerf.cfg['sc_factor'])   # 调整mesh符合现实中的位置及尺度
  return mesh


def run_one_ob(base_dir, cfg, use_refined_mask=False):
  save_dir = f'{base_dir}/nerf'
  os.system(f'rm -rf {save_dir} && mkdir -p {save_dir}')
  # with open(f'{base_dir}/select_frames.yml','r') as ff:
  #   info = yaml.safe_load(ff)
  rgbs = []
  depths = []
  masks = []
  cam_in_obs = []
  color_files = sorted(glob.glob(f'{base_dir}/rgb/*.png'))
  K = np.loadtxt(f'{base_dir}/K.txt')
  for i,color_file in enumerate(color_files):
    rgb = imageio.imread(color_file)
    depth = cv2.imread(color_file.replace('rgb','depth_enhanced'), -1)/1e3
    if use_refined_mask:
      mask = cv2.imread(color_file.replace('rgb','mask_refined'), -1)
    else:
      mask = cv2.imread(color_file.replace('rgb','mask'), -1)
    cam_in_ob = np.loadtxt(color_file.replace('rgb','cam_in_ob').replace('.png','.txt')).reshape(4,4)
    rgbs.append(rgb)
    depths.append(depth)
    masks.append(mask)
    cam_in_obs.append(cam_in_ob)

  mesh = run_neural_object_field(cfg, K, rgbs, depths, masks, cam_in_obs, save_dir=save_dir, debug=0)
  return mesh


def run_ycbv():
  ob_ids = np.arange(1,22)
  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open(f'{code_dir}/config_ycbv.yml','r') as ff:
    cfg = yaml.safe_load(ff)

  for ob_id in ob_ids:
    base_dir = f'{args.ref_view_dir}/ob_{ob_id:07d}'
    mesh = run_one_ob(base_dir=base_dir, cfg=cfg)
    out_file = f'{base_dir}/model/model.obj'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mesh.export(out_file)


def run_linemod():
  # 动态生成存在的 ob_ids 列表
  base_path = args.ref_view_dir
  code_dir = os.path.dirname(os.path.realpath(__file__))

  # 检索文件夹中所有以 "ob_" 开头的目录
  ob_folders = sorted(
    [f for f in os.listdir(base_path) if f.startswith("ob_") and os.path.isdir(os.path.join(base_path, f))]
  )

  # 提取编号列表
  ob_ids = [int(folder.split("_")[-1]) for folder in ob_folders]
  print(f"Detected object folders: {ob_ids}")

  # 读取配置文件
  with open(f'{code_dir}/config_linemod.yml', 'r') as ff:
    cfg = yaml.safe_load(ff)

  # 遍历每个对象文件夹
  for ob_id in ob_ids:
    base_dir = f'{args.ref_view_dir}/ob_{ob_id:07d}'

    # 检查文件夹是否存在
    if not os.path.exists(base_dir):
      print(f"Skipping {base_dir}: Folder does not exist.")
      continue

    try:
      # 调用单对象处理逻辑
      mesh = run_one_ob(base_dir=base_dir, cfg=cfg, use_refined_mask=False)
      out_file = f'{base_dir}/model/model.obj'
      os.makedirs(os.path.dirname(out_file), exist_ok=True)
      mesh.export(out_file)
      print(f"Saved model to {out_file}")
    except Exception as e:
      print(f"Error processing {base_dir}: {e}")


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--ref_view_dir', type=str, default=f'/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16')
  parser.add_argument('--dataset', type=str, default=f'ycbv', help='one of [ycbv/linemod]')
  args = parser.parse_args()

  if args.dataset=='ycbv':
    run_ycbv()
  else:
    run_linemod()
