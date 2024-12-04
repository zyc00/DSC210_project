import torch
import os
from yacs.config import CfgNode as CN

_C = CN()

_C.dbg = False
_C.evaltime = False
_C.deterministic = False
_C.backup_src = True
_C.do_eval_after_train = False

_C.sim = CN()
_C.sim.cam_pose = []
_C.sim.fx = 0.0
_C.sim.fy = 0.0
_C.sim.cx = 0.0
_C.sim.cy = 0.0
_C.sim.qpos = []
_C.sim.outdir = ""
_C.sim.random_qpos = False
_C.sim.random_qpos_number = 10
_C.sim.camera_ring = False
_C.sim.envmaps = ["SaintPetersSquare2"]
_C.sim.random_light_each_step = False
_C.sim.n_point_light = 10
# _C.sim.eye_on_hand = False
_C.sim.eye_on_hand_pose = []
_C.sim.urdf_path = "data/xarm7.urdf"
_C.sim.srdf_path = "data/xarm7.urdf"
_C.sim.move_group = ""
_C.sim.add_ground = CN()
_C.sim.add_ground.enable = False
_C.sim.add_desk_cube = CN()
_C.sim.add_desk_cube.enable = False
_C.sim.add_desk_cube.half_size = [0.5, 0.5, 0.5]
_C.sim.add_desk_cube.pose = [0.0, 0.0, 0.0]
_C.sim.add_desk_cube.color = [0.5, 0.5, 0.5]
_C.sim.eih_nqpos = 100
_C.sim.rt = CN()
_C.sim.rt.enable = False

_C.sim_hec = CN()
_C.sim_hec.Tc_c2b = []
_C.sim_hec.Tc_c2b_npy_path = ""
_C.sim_hec.Tc_c2b_index = 0
_C.sim_hec.Tc_c2e = []  # for eye in hand
_C.sim_hec.Tc_c2e_npy_path = ""
_C.sim_hec.Tc_c2e_index = 0
_C.sim_hec.height = 720
_C.sim_hec.width = 1280
_C.sim_hec.fx = 9.068051757812500000e+02
_C.sim_hec.fy = 9.066802978515625000e+02
_C.sim_hec.cx = 6.501978759765625000e+02
_C.sim_hec.cy = 3.677142944335937500e+02
_C.sim_hec.n_point_light = 10
_C.sim_hec.rt = CN()
_C.sim_hec.rt.enable = False
_C.sim_hec.qpos = []
_C.sim_hec.outdir = ""
_C.sim_hec.random_qpos = False
_C.sim_hec.random_qpos_number = 10
_C.sim_hec.camera_ring = False
_C.sim_hec.eye_on_hand_pose = []
_C.sim_hec.urdf_path = "data/xarm7_textured.urdf"
_C.sim_hec.srdf_path = ""  # requiref for random sample qpos
_C.sim_hec.step = False
_C.sim_hec.add_ground = False
_C.sim_hec.add_desk_cube = CN()
_C.sim_hec.add_desk_cube.enable = False
_C.sim_hec.add_desk_cube.half_size = [0.5, 0.5, 0.5]
_C.sim_hec.add_desk_cube.color = [0, 0, 0]
_C.sim_hec.add_desk_cube.pose = [-0.3, 0, -0.5]
_C.sim_hec.add_desk_cube.half_sizes = []  # for eye in hand
_C.sim_hec.add_desk_cube.colors = []  # for eye in hand
_C.sim_hec.add_desk_cube.poses = []  # for eye in hand

_C.sim_hec_with_marker = CN()
_C.sim_hec_with_marker.Tc_c2b = []
_C.sim_hec_with_marker.fx = 9.068051757812500000e+02
_C.sim_hec_with_marker.fy = 9.066802978515625000e+02
_C.sim_hec_with_marker.cx = 6.501978759765625000e+02
_C.sim_hec_with_marker.cy = 3.677142944335937500e+02
_C.sim_hec_with_marker.n_point_light = 10
_C.sim_hec_with_marker.rt = CN()
_C.sim_hec_with_marker.rt.enable = False
_C.sim_hec_with_marker.qpos = []
_C.sim_hec_with_marker.outdir = ""
_C.sim_hec_with_marker.random_qpos = False
_C.sim_hec_with_marker.random_qpos_number = 10
_C.sim_hec_with_marker.camera_ring = CN()
_C.sim_hec_with_marker.camera_ring.enable = False
_C.sim_hec_with_marker.camera_ring.min_dist = 0.7
_C.sim_hec_with_marker.camera_ring.max_dist = 1.5
_C.sim_hec_with_marker.camera_ring.min_elev = -80
_C.sim_hec_with_marker.camera_ring.max_elev = 80
_C.sim_hec_with_marker.camera_ring.ndist = 5
_C.sim_hec_with_marker.camera_ring.nelev = 18
_C.sim_hec_with_marker.camera_ring.nazim = 12
_C.sim_hec_with_marker.camera_ring.trans_noise = 0.0
_C.sim_hec_with_marker.camera_ring.endpoint = True
_C.sim_hec_with_marker.camera_ring.start_azim = 0.0
_C.sim_hec_with_marker.camera_ring.end_azim = 6.283185307179586
_C.sim_hec_with_marker.eye_on_hand_pose = []
_C.sim_hec_with_marker.urdf_path = "data/xarm7_textured.urdf"
_C.sim_hec_with_marker.srdf_path = "data/xarm7_textured.srdf"
_C.sim_hec_with_marker.move_group = "link_eef"
# _C.sim_hec_with_marker.step = False
_C.sim_hec_with_marker.add_chessboard = True
_C.sim_hec_with_marker.chessboard_path = "data/chessboard2.glb"
_C.sim_hec_with_marker.add_ground = False
_C.sim_hec_with_marker.add_desk_cube = CN()
_C.sim_hec_with_marker.add_desk_cube.enable = False
_C.sim_hec_with_marker.add_desk_cube.half_size = [0.5, 0.5, 0.5]
_C.sim_hec_with_marker.add_desk_cube.color = [0, 0, 0]
_C.sim_hec_with_marker.add_desk_cube.pose = [-0.3, 0, -0.5]

_C.sim_hec_eye_in_hand = CN()
_C.sim_hec_eye_in_hand.fx = 9.068051757812500000e+02
_C.sim_hec_eye_in_hand.fy = 9.066802978515625000e+02
_C.sim_hec_eye_in_hand.cx = 6.501978759765625000e+02
_C.sim_hec_eye_in_hand.cy = 3.677142944335937500e+02
_C.sim_hec_eye_in_hand.n_point_light = 10
_C.sim_hec_eye_in_hand.rt = CN()
_C.sim_hec_eye_in_hand.rt.enable = False
_C.sim_hec_eye_in_hand.outdir = ""
_C.sim_hec_eye_in_hand.qpos = []
_C.sim_hec_eye_in_hand.random_qpos = False
_C.sim_hec_eye_in_hand.random_qpos_number = 10
_C.sim_hec_eye_in_hand.Tc_c2e = []
_C.sim_hec_eye_in_hand.urdf_path = "data/xarm7_textured.urdf"
_C.sim_hec_eye_in_hand.srdf_path = "data/xarm7_textured.srdf"
_C.sim_hec_eye_in_hand.move_group = "link_eef"
_C.sim_hec_eye_in_hand.add_chessboard = True
_C.sim_hec_eye_in_hand.chessboard_path = "data/chessboard2.glb"
_C.sim_hec_eye_in_hand.add_ground = False
_C.sim_hec_eye_in_hand.add_desk_cube = CN()
_C.sim_hec_eye_in_hand.add_desk_cube.enable = False
_C.sim_hec_eye_in_hand.add_desk_cube.half_sizes = [[0.5, 0.5, 0.5]]
_C.sim_hec_eye_in_hand.add_desk_cube.colors = [[0, 0, 0]]
_C.sim_hec_eye_in_hand.add_desk_cube.poses = [[-0.3, 0, -0.5]]

_C.sim_pvnet = CN()
_C.sim_pvnet.K = [[9.068051757812500000e+02, 0.000000000000000000e+00, 6.501978759765625000e+02],
                  [0.000000000000000000e+00, 9.066802978515625000e+02, 3.677142944335937500e+02],
                  [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]]
_C.sim_pvnet.Tc_c2b = []
_C.sim_pvnet.min_dist = 0.5
_C.sim_pvnet.max_dist = 2.0
_C.sim_pvnet.n_dist = 10
_C.sim_pvnet.min_elev = 0
_C.sim_pvnet.max_elev = 80
_C.sim_pvnet.n_elev = 10
_C.sim_pvnet.nazim = 30
_C.sim_pvnet.trans_noise = [0.03] * 3
_C.sim_pvnet.n_point_light = 10
_C.sim_pvnet.random_light_each_step = False
_C.sim_pvnet.envmaps = ["SaintPetersSquare2"]
_C.sim_pvnet.rt = CN()
_C.sim_pvnet.rt.enable = False
_C.sim_pvnet.outdir = ""
_C.sim_pvnet.urdf_path = "data/xarm7_textured.urdf"
_C.sim_pvnet.add_ground = False
_C.sim_pvnet.add_desk_cube = CN()
_C.sim_pvnet.add_desk_cube.enable = False
_C.sim_pvnet.add_desk_cube.half_size = [0.5, 0.5, 0.5]
_C.sim_pvnet.add_desk_cube.color = [0, 0, 0]
_C.sim_pvnet.add_desk_cube.pose = [-0.3, 0, -0.5]

_C.rb_solve = CN()
_C.rb_solve.nimgs = -1
_C.rb_solve.use_nvdiffrast = False
_C.rb_solve.use_gt_mask = False
_C.rb_solve.data_dir = ""
_C.rb_solve.init_Tbc = []  # eye off hand version
_C.rb_solve.init_Tc_c2e = []  # eye on hand version
_C.rb_solve.steps = []
_C.rb_solve.validate_dir = ""
_C.rb_solve.dbg = True
_C.rb_solve.render_sizes = []
_C.rb_solve.lrs = []
_C.rb_solve.links = [0, 1, 2, 3, 4, 5, 6, 7]
_C.rb_solve.optimizer = 'adam'  # adam or sgd
_C.rb_solve.neg_iou_loss = False
_C.rb_solve.log_interval = 20
_C.rb_solve.eye_on_hand = False
_C.rb_solve.space_explore = CN()
_C.rb_solve.space_explore.start = 200
_C.rb_solve.space_explore.sample = 10
_C.rb_solve.space_explore.n_sample_qpos_each_joint = 2  # for grid
_C.rb_solve.space_explore.n_sample_qposes = 1000  # for random
_C.rb_solve.space_explore.qpos_sample_method = 'grid'  # grid or random_qpos or random_eef

_C.meshloc = CN()
_C.meshloc.ref_dir = "data/sim_hec_for_meshloc_ref_v1"
_C.meshloc.query_dir = "data/sim_hec_with_marker/version2_no_marker"
# _C.meshloc.mesh_dir = "/home/linghao/clhroot/mnt/data/project_data/instant-nsr-pl/exp/neus-oneposeppv2_reducelr_r0.3/0700-toyrobot-others/@20230623-161001/save/it20000-mc512.obj"
_C.meshloc.object_ids = ["000000"]
_C.meshloc.topk = 10
_C.meshloc.matching_method = "dkm"
_C.meshloc.pnp_method = "ransac"
_C.meshloc.pnp_ransac_reproj_error = 8.0
_C.meshloc.pnp_ransac_iterations_count = 10000
_C.meshloc.loftr_thresh = 0.2
# _C.meshloc.matchings_path = "data/onepose_lowtexture/2dmatching_loftr_512.pkl"

_C.output_dir = ""

_C.pnp_hole_pose = CN()
_C.pnp_hole_pose.whl = [0.0635, 0.04445, 0.127]
_C.pnp_hole_pose.pts3d = [
    [0.03175, -0.022225, -0.0635],
    [0.03175, 0.022225, -0.0635],
    [-0.03175, 0.022225, -0.0635],
    [-0.03175, -0.022225, -0.0635],
    [-0.03175, -0.022225, 0.0635],
    [0.03175, -0.022225, 0.0635],
    [0.03175, 0.022225, 0.0635]
]
_C.pnp_hole_pose.pts2d = []
_C.pnp_hole_pose.data_dir = ""

_C.pnp_with_chessboard = CN()
_C.pnp_with_chessboard.wlh = [0.0635, 0.127, 0.04445]
_C.pnp_with_chessboard.data_dir = ""
_C.pnp_with_chessboard.patterns = []
_C.pnp_with_chessboard.pattern_planes = []
_C.pnp_with_chessboard.grid_sizes = []

_C.annotate_video = CN()
_C.annotate_video.data_dir = ""
_C.annotate_video.whl = [0.0635, 0.04445, 0.127]
_C.annotate_video.pts_3d = [
    [0.03175, -0.022225, -0.0635],
    [0.03175, 0.022225, -0.0635],
    [-0.03175, 0.022225, -0.0635],
    [-0.03175, -0.022225, -0.0635],
    [-0.03175, -0.022225, 0.0635],
    [0.03175, -0.022225, 0.0635],
    [0.03175, 0.022225, 0.0635]
]
_C.annotate_video.use_aruco = False
_C.annotate_video.n = -1
_C.annotate_video.chessboard = CN()
_C.annotate_video.chessboard.rows = 6
_C.annotate_video.chessboard.cols = 9
_C.annotate_video.chessboard.grid_size = 0.028
_C.annotate_video.aruco = CN()
_C.annotate_video.aruco.rows = 4
_C.annotate_video.aruco.cols = 7
_C.annotate_video.aruco.marker_size = 0.015
_C.annotate_video.aruco.marker_sep = 0.005
_C.annotate_video.aruco.dictionary = "DICT_ARUCO_ORIGINAL"

_C.bproc_cube = CN()
_C.bproc_cube.mesh_path = "data/untitled.obj"
_C.bproc_cube.K = [[9.068051757812500000e+02, 0.000000000000000000e+00, 6.501978759765625000e+02],
                   [0.000000000000000000e+00, 9.066802978515625000e+02, 3.677142944335937500e+02],
                   [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]]
_C.bproc_cube.H = 720
_C.bproc_cube.W = 1280
_C.bproc_cube.resize_factor = 1
_C.bproc_cube.dists = [0.4, 0.9]
_C.bproc_cube.noise = 0.0
_C.bproc_cube.max_elev = 60.0
_C.bproc_cube.n_elev = 10
_C.bproc_cube.nazim = 12
_C.bproc_cube.output_dir = "data/blenderproc"
_C.bproc_cube.do_render = False
_C.bproc_cube.light_plane_z = 2
_C.bproc_cube.colors = CN()
_C.bproc_cube.colors.top = [1.0, 1.0, 1.0, 1]
_C.bproc_cube.colors.left = [1.0, 0.0, 0.0, 1]
_C.bproc_cube.colors.front = [0.0, 1.0, 0.0, 1]
_C.bproc_cube.colors.right = [0.54, 0.27, 0.074, 1]
_C.bproc_cube.colors.back = [0.0, 0.0, 1.0, 1]
_C.bproc_cube.colors.bottom = [1.0, 1.0, 0.0, 1]

_C.sapien_cube = CN()
_C.sapien_cube.robot_pose = [-0.6, 0, 0, 1, 0, 0, 0]
# _C.sapien_cube.robot_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0]
_C.sapien_cube.robot_qpos_path = "tmp/robot_qpos.npy"
_C.sapien_cube.Tw_w2o_path = "tmp/cubaA_poses.npy"
_C.sapien_cube.mesh_path = "data/untitled.obj"
_C.sapien_cube.visual_path = "data/untitled.dae"
_C.sapien_cube.K = [[9.068051757812500000e+02, 0.000000000000000000e+00, 6.501978759765625000e+02],
                    [0.000000000000000000e+00, 9.066802978515625000e+02, 3.677142944335937500e+02],
                    [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]]
_C.sapien_cube.H = 720
_C.sapien_cube.W = 1280
_C.sapien_cube.TB_B2w = [[1, 0, 0, 0.6],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]
_C.sapien_cube.rotz = 0
_C.sapien_cube.gripper_qpos = [0, 0, 0, 0, 0, 0]
_C.sapien_cube.urdf_path = "data/xarm7_with_gripper.urdf"
_C.sapien_cube.rt = CN()
_C.sapien_cube.rt.enable = False
_C.sapien_cube.camera_pose = []
_C.sapien_cube.outdir = ""
_C.sapien_cube.fx = 0.0
_C.sapien_cube.fy = 0.0
_C.sapien_cube.cx = 0.0
_C.sapien_cube.cy = 0.0

# _C.sapien_cube.dists = [0.4, 0.9]
# _C.sapien_cube.noise = 0.0
# _C.sapien_cube.max_elev = 60.0
# _C.sapien_cube.n_elev = 10
# _C.sapien_cube.nazim = 12
_C.sapien_cube.output_dir = "data/sapien_cube"
# _C.sapien_cube.do_render = False
# _C.sapien_cube.light_plane_z = 2
# _C.sapien_cube.colors = CN()
# _C.sapien_cube.colors.top = [1.0, 1.0, 1.0, 1]
# _C.sapien_cube.colors.left = [1.0, 0.0, 0.0, 1]
# _C.sapien_cube.colors.front = [0.0, 1.0, 0.0, 1]
# _C.sapien_cube.colors.right = [0.54, 0.27, 0.074, 1]
# _C.sapien_cube.colors.back = [0.0, 0.0, 1.0, 1]
# _C.sapien_cube.colors.bottom = [1.0, 1.0, 0.0, 1]


_C.maskrcnn = CN()
_C.maskrcnn.data_dir = ""
_C.maskrcnn.model_cfg_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
_C.maskrcnn.base_lr = 0.00025
_C.maskrcnn.max_iter = 8000
_C.maskrcnn.batch_size = 2
_C.maskrcnn.num_workers = 4
_C.maskrcnn.score_thresh_test = 0.7
_C.maskrcnn.batch_size_per_image = 128
_C.maskrcnn.checkpoint_period = 5000

_C.dataset = CN()

_C.dataset.xarm = CN()
_C.dataset.xarm.load_mask_gt = False
_C.dataset.xarm.load_mask_pred = False
_C.dataset.xarm.load_mask_anno = False
_C.dataset.xarm.load_mask_gripper = False
_C.dataset.xarm.eye_in_hand = False
_C.dataset.xarm.urdf_path = "data/xarm7_with_gripper_reduced_dof.urdf"
_C.dataset.xarm.use_links = [2, 3, 4, 5, 6, 7, 8]
_C.dataset.xarm.shift_gripper = False
_C.dataset.xarm.selected_indices = []

_C.dataset.realman = CN()
_C.dataset.realman.load_mask_gt = False
_C.dataset.realman.load_mask_pred = False
_C.dataset.realman.load_mask_anno = True
# _C.dataset.realman.load_mask_gripper = False
_C.dataset.realman.eye_in_hand = False
_C.dataset.realman.urdf_path = "data/hillbot_pgi_description_linghao/hillbot_pgi_description_linghao.urdf"
_C.dataset.realman.use_links = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# _C.dataset.realman.shift_gripper = False
_C.dataset.realman.selected_indices = []
_C.dataset.realman.camera_id = 'head'
_C.dataset.realman.relative_to = 'base_link'
_C.dataset.realman.resize = [-1, -1]  # h,w

_C.dataset.realman_hec_marker = CN()
_C.dataset.realman_hec_marker.camera_id = 'right'
_C.dataset.realman_hec_marker.urdf_path = "data/hillbot_pgi_description_linghao/hillbot_pgi_description_linghao.urdf"

_C.dataset.mobilerobot = CN()
_C.dataset.mobilerobot.load_mask_gt = False
_C.dataset.mobilerobot.load_mask_pred = False
_C.dataset.mobilerobot.load_mask_anno = False
_C.dataset.mobilerobot.load_mask_gripper = False
_C.dataset.mobilerobot.eye_in_hand = False
_C.dataset.mobilerobot.urdf_path = "data/mobilerobot/mycobot_pi_v2/mycobot_urdf_stl.urdf"
_C.dataset.mobilerobot.use_links = [0, 1, 2, 3, 4, 5, 6]
_C.dataset.mobilerobot.shift_gripper = False
_C.dataset.mobilerobot.selected_indices = []

_C.dataset.xarm_hec_sim_marker = CN()
_C.dataset.xarm_hec_sim_marker.urdf_path = "data/xarm7_with_gripper_reduced_dof.urdf"

_C.dataset.dream_real_panda_3cam_realsense = CN()
_C.dataset.dream_real_panda_3cam_realsense.load_mask_pred = False
_C.dataset.dream_real_panda_3cam_realsense.load_mask_anno = False
_C.dataset.dream_real_panda_3cam_realsense.urdf_path = "./ManiSkill2/mani_skill2/assets/descriptions/panda_v2.urdf"
_C.dataset.dream_real_panda_3cam_realsense.use_links = [0, 2, 3, 4, 6, 7, 9]

_C.dataset.dream_synthetic_panda_train_dr = CN()
_C.dataset.dream_synthetic_panda_train_dr.load_mask_pred = False
_C.dataset.dream_synthetic_panda_train_dr.load_mask_anno = False
_C.dataset.dream_synthetic_panda_train_dr.urdf_path = "./ManiSkill2/mani_skill2/assets/descriptions/panda_v2.urdf"
_C.dataset.dream_synthetic_panda_train_dr.use_links = [0, 2, 3, 4, 6, 7, 9]

_C.dataset.dream_synthetic_panda_test_dr = CN()
_C.dataset.dream_synthetic_panda_test_dr.load_mask_pred = False
_C.dataset.dream_synthetic_panda_test_dr.load_mask_anno = False
_C.dataset.dream_synthetic_panda_test_dr.urdf_path = "data/panda_v2.urdf"
_C.dataset.dream_synthetic_panda_test_dr.use_links = [0, 2, 3, 4, 6, 7, 9]

_C.dataset.dream_synthetic_panda_test_photo = CN()
_C.dataset.dream_synthetic_panda_test_photo.load_mask_pred = False
_C.dataset.dream_synthetic_panda_test_photo.load_mask_anno = False
_C.dataset.dream_synthetic_panda_test_photo.urdf_path = "data/panda_v2.urdf"
_C.dataset.dream_synthetic_panda_test_photo.use_links = [0, 2, 3, 4, 6, 7, 9]

_C.dataset.baxter_real = CN()
_C.dataset.baxter_real.load_mask_pred = False
_C.dataset.baxter_real.load_mask_anno = False
_C.dataset.baxter_real.load_mask_sam = False
_C.dataset.baxter_real.load_mask_gsam = False
_C.dataset.baxter_real.urdf_path = "./baxter_common/baxter.urdf"
_C.dataset.baxter_real.use_links = [32, 33, 34, 35, 36, 38]
_C.dataset.baxter_real.selected_indices = []

_C.model = CN()
_C.model.meta_architecture = ""
_C.model.device = "cuda"

_C.model.hand_eye_solver = CN()
_C.model.hand_eye_solver.icp = False
_C.model.hand_eye_solver.cpa = False
_C.model.hand_eye_solver.subpixel = True
_C.model.hand_eye_solver.use_gt_marker_pose = False
_C.model.hand_eye_solver.manual_set_zero = []
_C.model.hand_eye_solver.chessboard = True

_C.model.hand_eye_solver_gn = CN()
_C.model.hand_eye_solver_gn.icp = False
_C.model.hand_eye_solver_gn.cpa = False
_C.model.hand_eye_solver_gn.subpixel = True
_C.model.hand_eye_solver_gn.use_gt_marker_pose = False
_C.model.hand_eye_solver_gn.manual_set_zero = []

_C.model.rbsolver = CN()
_C.model.rbsolver.use_mask = "gt"  # gt or pred or anno
_C.model.rbsolver.weighted_loss = False
_C.model.rbsolver.loss_type = 'mse'
_C.model.rbsolver.render_scale = 1.0
_C.model.rbsolver.optim_trans_only = False
_C.model.rbsolver.init_Tc_c2b = []  # eye off hand version
_C.model.rbsolver.init_Tc_c2e = []  # eye on hand version
_C.model.rbsolver.lrs = []
# _C.model.rbsolver.mesh_dir = "data/xarm_description/meshes/xarm7/visual"
# _C.model.rbsolver.links = [0, 1, 2, 3, 4, 5, 6, 7]
_C.model.rbsolver.mesh_paths = ["data/xarm_description/meshes/xarm7/visual/link0.STL",
                                "data/xarm_description/meshes/xarm7/visual/link1.STL",
                                "data/xarm_description/meshes/xarm7/visual/link2.STL",
                                "data/xarm_description/meshes/xarm7/visual/link3.STL",
                                "data/xarm_description/meshes/xarm7/visual/link4.STL",
                                "data/xarm_description/meshes/xarm7/visual/link5.STL",
                                "data/xarm_description/meshes/xarm7/visual/link6.STL",
                                "data/xarm_description/meshes/xarm7/visual/link7.STL", ]
_C.model.rbsolver.eye_in_hand = False
_C.model.rbsolver.optim_mesh_scale = False
_C.model.rbsolver.optim_eep = False
_C.model.rbsolver.ignore_gripper_mask = False
_C.model.rbsolver.eep_init = []
_C.model.rbsolver.optim_eep_xy = True
_C.model.rbsolver.optim_eep_rot = True
_C.model.rbsolver.optim_eep_mask = []
_C.model.rbsolver.eep_rotz = 0.0
_C.model.rbsolver.eep_tz = 0.0
_C.model.rbsolver.reg_loss_weight = 0.0
_C.model.rbsolver.H = 720
_C.model.rbsolver.W = 1280
_C.model.rbsolver.use_last_as_result = True
_C.model.rbsolver.optim_inverse = False
_C.model.rbsolver.last2link_loss_weights = 1
# _C.model.rbsolver.relative_to_link_index = 0

_C.model.space_explorer = CN()
_C.model.space_explorer.parallel_rendering = False
_C.model.space_explorer.variance_based_sampling = True  # for ablation study: random sample next qpos
_C.model.space_explorer.qpos_choices = ""  # if "", will use the sampled qpos, else, it should be a txt file with len(qpos)xnjoints
_C.model.space_explorer.qpos_choices_pad_left = 0
_C.model.space_explorer.qpos_choices_pad_right = 0
_C.model.space_explorer.sample_camera_poses_method = "random"
_C.model.space_explorer.render_resize_factor = 1

_C.model.space_explorer.var_in_valid_mask = False

_C.model.space_explorer.mesh_dir = "data/xarm_description/meshes/xarm7/visual"
_C.model.space_explorer.ckpt_path = ""
_C.model.space_explorer.start = 200
_C.model.space_explorer.resample_camera_poses = CN()
_C.model.space_explorer.resample_camera_poses.enable = False
_C.model.space_explorer.resample_camera_poses.maxn = 0
# _C.model.space_explorer.resample_camera_poses.rot_noise = 0.0
# _C.model.space_explorer.resample_camera_poses.trans_noise = 0.0

_C.model.space_explorer.sample = 10  # num camera sampled
_C.model.space_explorer.qpos_sample_method = 'random'  # grid or random
_C.model.space_explorer.qpos_sample_method_switch = False  # randomly switch between random and random_eef
_C.model.space_explorer.n_sample_qpos_each_joint = 2  # for grid
_C.model.space_explorer.n_sample_qposes = 1000  # for random
_C.model.space_explorer.sample_dof = 6
_C.model.space_explorer.sample_limit = False  # should be True!
_C.model.space_explorer.qpos_sample_range = 0.0
_C.model.space_explorer.camera_intrinsic = [9.068051757812500000e+02, 9.066802978515625000e+02,
                                            6.501978759765625000e+02, 3.677142944335937500e+02]
_C.model.space_explorer.urdf_path = ""
_C.model.space_explorer.srdf_path = ""
_C.model.space_explorer.move_group = ""

_C.model.space_explorer.self_collision_check = CN()
_C.model.space_explorer.self_collision_check.enable = True

_C.model.space_explorer.max_dist_constraint = CN()
_C.model.space_explorer.max_dist_constraint.enable = False
_C.model.space_explorer.max_dist_constraint.max_dist = 0.5
# _C.model.space_explorer.max_dist_constraint.max_dist_constraint = 0.5
_C.model.space_explorer.max_dist_constraint.max_dist_center_compute_n = 100000
_C.model.space_explorer.max_dist_constraint.max_dist_center = 0.0

_C.model.space_explorer.collision_check = CN()
_C.model.space_explorer.collision_check.enable = False
_C.model.space_explorer.collision_check.use_pointcloud = True
_C.model.space_explorer.collision_check.pc_sampled = -1
_C.model.space_explorer.collision_check.ignore_posz_pts = False
_C.model.space_explorer.collision_check.dilate = 10
_C.model.space_explorer.collision_check.by_eef_pose = True
_C.model.space_explorer.collision_check.timestep = 0.01
_C.model.space_explorer.collision_check.planning_time = 1.0

_C.model.rbsolver_iter = CN()
_C.model.rbsolver_iter.data_dir = ""
_C.model.rbsolver_iter.start_qpos = [0, 0, 0, 0, 0, 0, 0]
_C.model.rbsolver_iter.start_index = 0  # for data pool
_C.model.rbsolver_iter.segmentor = "pointrend"
_C.model.rbsolver_iter.autosam_box_scale = 1.0
_C.model.rbsolver_iter.autosam_links = [1, 2]  # 1,2,-1
_C.model.rbsolver_iter.autosam_point_center = False
_C.model.rbsolver_iter.autosam_random_points = False
_C.model.rbsolver_iter.init_method = "manual"
_C.model.rbsolver_iter.meshloc_database_dir = ""
_C.model.rbsolver_iter.meshloc_use_all_imgs = False
_C.model.rbsolver_iter.meshloc_init_Tc_c2b = []
_C.model.rbsolver_iter.explore_iters = 1
_C.model.rbsolver_iter.qpos_noise = 0.0

_C.model.rbsolver_iter.qpos_pool = []

# if empty, will use sapien to render;
# if not empty, the first one is the initial state, and the rest are the qpos choices
# in each sample in the data pool, the data format is [qpos/000000.txt, anno_mask/000000.png, color/00000.png]

_C.model.rbsolver_iter.data_pool = []

_C.model.rbsolver_iter.use_realarm = CN()
_C.model.rbsolver_iter.use_realarm.enable = False
_C.model.rbsolver_iter.use_realarm.ip = "192.168.1.209"
_C.model.rbsolver_iter.use_realarm.speed = 0.1
_C.model.rbsolver_iter.use_realarm.wait_time = 1
_C.model.rbsolver_iter.use_realarm.speed_control = False
_C.model.rbsolver_iter.use_realarm.timestep = 0.1
_C.model.rbsolver_iter.use_realarm.safety_factor = 3
_C.model.rbsolver_iter.use_realarm.record_video = CN()  # for mask data collection
_C.model.rbsolver_iter.use_realarm.record_video.enable = False
_C.model.rbsolver_iter.use_realarm.record_video.outdir = "data/realarm_video"

_C.model.rbsolver_iter.use_realman = CN()
_C.model.rbsolver_iter.use_realman.enable = False
_C.model.rbsolver_iter.use_realman.arm = ''  # left or right
_C.model.rbsolver_iter.use_realman.ip = "192.168.31.175"
_C.model.rbsolver_iter.use_realman.speed = 1
_C.model.rbsolver_iter.use_realman.wait_time = 1
# _C.model.rbsolver_iter.use_realman.speed_control = False
# _C.model.rbsolver_iter.use_realman.timestep = 0.1
# _C.model.rbsolver_iter.use_realman.safety_factor = 3
# _C.model.rbsolver_iter.use_realman.record_video = CN()  # for mask data collection
# _C.model.rbsolver_iter.use_realman.record_video.enable = False
# _C.model.rbsolver_iter.use_realman.record_video.outdir = "data/realarm_video"

_C.model.rbsolver_iter.pointrend_cfg_file = "configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_xarm.yaml"
_C.model.rbsolver_iter.pointrend_model_weight = "output/model_0099999.pth"

_C.model.megapose6d = CN()
_C.model.megapose6d.generate_data = CN()
_C.model.megapose6d.generate_data.urdf_path = "data/xarm7_textured.urdf"
_C.model.megapose6d.generate_data.H = 720
_C.model.megapose6d.generate_data.W = 1280
_C.model.megapose6d.generate_data.K = [[9.068051757812500000e+02, 0.000000000000000000e+00, 6.501978759765625000e+02],
                                       [0.000000000000000000e+00, 9.066802978515625000e+02, 3.677142944335937500e+02],
                                       [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]]
_C.model.megapose6d.generate_data.min_dist = 0.5
_C.model.megapose6d.generate_data.max_dist = 2.0
_C.model.megapose6d.generate_data.n_dist = 10
_C.model.megapose6d.generate_data.min_elev = 0
_C.model.megapose6d.generate_data.max_elev = 80
_C.model.megapose6d.generate_data.n_elev = 10
_C.model.megapose6d.generate_data.nazim = 30
_C.model.megapose6d.generate_data.trans_noise = [0.03] * 3

_C.model.megapose6d.generate_data.output_dir = ""

_C.model.dream = CN()
_C.model.dream.resnet = CN()
_C.model.dream.resnet.n_keypoints = 7
_C.model.dream.resnet.pretrained = True
_C.model.dream.resnet.full = False
_C.model.dream.resnet.old_process = True

_C.solver = CN()
_C.solver.explore_iters = 10  # explore iterations for rendering-based solver
_C.solver.num_epochs = 1
_C.solver.max_lr = 0.001
_C.solver.end_lr = 0.0001
_C.solver.end_pose_lr = 0.00001
_C.solver.bias_lr_factor = 1
_C.solver.momentum = 0.9
_C.solver.weight_decay = 0.0005
_C.solver.weight_decay_bias = 0.0
_C.solver.gamma = 0.1
_C.solver.gamma_pose = 0.1
_C.solver.lrate_decay = 250
_C.solver.lrate_decay_pose = 250
_C.solver.steps = (30000,)
_C.solver.warmup_factor = 1.0 / 3
_C.solver.warmup_iters = 500
_C.solver.warmup_method = "linear"
_C.solver.num_iters = 10000  # volsdf
_C.solver.min_factor = 0.1  # volsdf
_C.solver.pose_lr = 0.00005  # volsdf
_C.solver.mlp_lr = 0.001
_C.solver.log_interval = 1
_C.solver.image_grid_on_tb_writer = True

_C.solver.optimizer = 'Adam'
_C.solver.scheduler = 'OneCycleScheduler'
_C.solver.scheduler_decay_thresh = 0.00005
_C.solver.do_grad_clip = False
_C.solver.grad_clip_type = 'norm'  # norm or value
_C.solver.grad_clip = 1.0
_C.solver.ds_len = -1
_C.solver.batch_size = 1
_C.solver.loss_function = ''
####save ckpt configs#####
_C.solver.save_min_loss = 20.0
_C.solver.save_every = False
_C.solver.save_freq = 1
_C.solver.save_mode = 'epoch'  # epoch or iteration
_C.solver.val_freq = 1
_C.solver.save_last_only = False
_C.solver.empty_cache = True
_C.solver.metric_functions = ()
_C.solver.trainer = "base"
_C.solver.load_model = ""
_C.solver.load_model_extras = []
_C.solver.load = ""
_C.solver.print_it = False
_C.solver.detect_anomaly = False
_C.solver.convert_sync_batchnorm = False
_C.solver.ddp_version = 'torch'  # or dtr
_C.solver.broadcast_buffers = False
_C.solver.find_unused_parameters = False
_C.solver.resume = False
_C.solver.dist = CN()
_C.solver.dist.sampler = CN()
_C.solver.dist.sampler.type = 'pytorch'
_C.solver.dist.sampler.shuffle = False
_C.solver.pop_verts_faces = False
_C.solver.compress_history_ops = False
_C.solver.save_optimizer = True
_C.solver.save_scheduler = True
_C.solver.grad_noise = CN()
_C.solver.grad_noise.enable = False
_C.solver.grad_noise.noise = 0.0

_C.dataloader = CN()
_C.dataloader.num_workers = 0
_C.dataloader.collator = 'DefaultBatchCollator'
_C.dataloader.pin_memory = False

_C.datasets = CN()
_C.datasets.train = ()
_C.datasets.test = ""

_C.input = CN()
_C.input.transforms = []
_C.input.shuffle = True

_C.paths_catalog = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

_C.test = CN()
_C.test.batch_size = 1
_C.test.evaluators = []
_C.test.visualizer = 'default'
_C.test.force_recompute = True
_C.test.do_evaluation = True
_C.test.do_visualization = False
_C.test.eval_all = False
_C.test.eval_all_min = 0
_C.test.save_predictions = False
_C.test.ckpt_dir = ''

_C.autoinit = CN()
_C.autoinit.radius = 1.0
_C.autoinit.elev_min = 0  # degree, from horizontal plane
_C.autoinit.elev_max = 70  # degree
_C.autoinit.elev_interval = 5  # degree
_C.autoinit.azim_interval = 5  # degree
_C.autoinit.urdf_path = "data/sapien_packages/xarm7/xarm_urdf/xarm7_gripper.urdf"
_C.autoinit.data_dir = ""
_C.autoinit.use_mask = ""  # gt, pred, ...
_C.autoinit.n = -1
_C.autoinit.nqpos = 10

_C.video_segm = CN()
_C.video_segm.data_dir = ""
