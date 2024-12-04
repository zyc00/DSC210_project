import os.path as osp
import os


class DatasetCatalog(object):
    default_data_dir = os.path.expanduser('.')
    DATA_DIR = os.environ.get('DATASET_HOME', default_data_dir)

    @staticmethod
    def get(name: str):
        if name.startswith("xarm_hec_sim_marker_eih"):
            return get_xarm_hec_sim_marker_eih(name)
        if name.startswith("xarm_hec_real_marker_eih"):
            return get_xarm_hec_real_marker_eih(name)
        if name.startswith("xarm_hec_sim_marker"):
            return get_xarm_hec_sim_marker(name)
        if name.startswith("xarm"):
            return get_xarm(name)
        # if name.startswith('xarm_hec_sim_eih'):
        #     raise DeprecationWarning()
        #     return get_xarm_hec_sim_eih(name)
        # if name.startswith('xarm_hec_sim'):
        #     raise DeprecationWarning()
        #     return get_xarm_hec_sim(name)
        # if name.startswith("xarm_real"):
        #     raise DeprecationWarning()
        #     return get_xarm_real(name)
        if name.startswith("dream_xarm"):
            return get_dream_xarm(name)
        if name.startswith("dream_kuka"):
            return get_dream_kuka(name)
        if name.startswith("dream_real_panda3cam_realsense"):
            return get_dream_real_panda3cam_realsense(name)
        if name.startswith("dream_synthetic_panda_train_dr"):
            return get_dream_synthetic_panda_train_dr(name)
        if name.startswith("dream_synthetic_panda_test_dr"):
            return get_dream_synthetic_panda_test_dr(name)
        if name.startswith("dream_synthetic_panda_test_photo"):
            return get_dream_synthetic_panda_test_photo(name)
        if name.startswith("baxter_real_se"):
            return get_baxter_real_se(name)
        if name.startswith("baxter_real"):
            return get_baxter_real(name)
        if name.startswith("mobilerobot"):
            return get_mobilerobot(name)
        if name.startswith("realman_hec_marker_eih"):
            return get_realman_hec_marker_eih(name)
        if name.startswith("realman"):
            return get_realman(name)
        raise RuntimeError("Dataset not available: {}".format(name))


def get_xarm(name):
    #     xarm_sim_for_hec_eih/example/000000
    items = name.split("_")
    scene = items[1:]
    scene = "_".join(scene)
    return dict(
        factory='XarmDataset',
        args={'data_dir': osp.join('data', scene),
              'ds_len': -1,
              }
    )


def get_mobilerobot(name):
    items = name.split("_")
    scene = items[1:]
    scene = "_".join(scene)
    return dict(
        factory='MobileRobotDataset',
        args={'data_dir': osp.join('data/mobilerobot', scene),
              'ds_len': -1,
              }
    )


def get_xarm_hec_sim_eih(name):
    #     xarm_hec_sim_eih_example/000000
    items = name.split("_")
    scene = items[4:]
    scene = "_".join(scene)
    return dict(
        factory='XarmHecSimEihDataset',
        args={'data_dir': osp.join('data/sim_for_hec_eih', scene),
              'ds_len': -1,
              }
    )


def get_xarm_hec_sim(name):
    # xarm_hec_sim_092547
    items = name.split("_")
    scene = items[3:]
    scene = "_".join(scene)
    return dict(
        factory='XarmHecSimDataset',
        args={'data_dir': osp.join('data/sim_for_hec', scene),
              'ds_len': -1,
              }
    )


def get_xarm_real(name):
    # xarm_real_rb_solver_iter/real_xarm/example
    items = name.split("_")[2:]
    data_dir = osp.join("data", "_".join(items))
    return dict(
        factory='XarmRealDataset',
        args={'data_dir': data_dir,
              'ds_len': -1,
              }
    )


def get_xarm_hec_sim_marker(name):
    # xarm_hec_sim_marker_version1_000000
    items = name.split("_")
    scene = items[4:]
    scene = "/".join(scene)
    return dict(
        factory='XarmHecSimMarkerDataset',
        args={'data_dir': osp.join('data/sim_hec_with_marker', scene),
              'ds_len': -1,
              }
    )


def get_xarm_hec_sim_marker_eih(name):
    # xarm_hec_sim_marker_version1_000000
    items = name.split("_")
    scene = items[5:]
    scene = "/".join(scene)
    return dict(
        factory='XarmHecSimMarkerEIHDataset',
        args={'data_dir': osp.join('data/sim_hec_with_marker_eih', scene),
              'ds_len': -1,
              }
    )


def get_xarm_hec_real_marker_eih(name):
    # xarm_hec_real_marker_eih_20231111_1323
    items = name.split("_")
    scene = items[5:]
    scene = "_".join(scene)
    return dict(
        factory='XarmHecSimMarkerEIHDataset',
        args={'data_dir': osp.join('data/hand_eye_solver', scene),
              'ds_len': -1,
              }
    )


def get_dream_xarm(name):
    # "dream_xarm_v4_train"
    version, split = name.split("_")[-2:]
    if "mini" in name:
        ds_len = 1
    else:
        ds_len = -1
    return dict(
        factory='DreamXArmDataset',
        args={'data_dir': f"data/sim/render_for_mrcnn_ring_rt_{version}",
              'split': split,
              'ds_len': ds_len,
              }
    )


def get_dream_kuka(name):
    # "dream_kuka_train"
    split = name.split("_")[-1]
    return dict(
        factory='DreamKukaDataset',
        args={'data_dir': f"/home/linghao/Datasets/DREAM/synthetic/kuka_synth_{split}_dr",
              'split': split,
              'ds_len': -1,
              }
    )


def get_dream_real_panda3cam_realsense(name):
    # dream_real_panda3cam_realsense
    return dict(
        factory='DreamRealPanda3CamRealsenseDataset',
        args={'data_dir': osp.expanduser(f'./DREAM/real/panda-3cam_realsense/processed'),
              'ds_len': -1,
              }
    )


def get_dream_synthetic_panda_train_dr(name):
    # dream_synthetic_panda_train_dr_0
    root_dir = osp.expanduser(f'./DREAM/synthetic/panda_synth_train_dr/processed')
    scene = int(name.split("_")[-1])
    data_dir = osp.join(root_dir, f"scene_{scene:06d}")
    return dict(
        factory='DreamSyntheticPandaTrainDrDataset',
        args={'data_dir': data_dir,
              'ds_len': -1,
              }
    )


def get_dream_synthetic_panda_test_dr(name):
    # dream_synthetic_panda_train_dr_0
    root_dir = osp.expanduser(f'./DREAM/synthetic/panda_synth_test_dr/processed')
    scene = int(name.split("_")[-1])
    data_dir = osp.join(root_dir, f"scene_{scene:06d}")
    return dict(
        factory='DreamSyntheticPandaTestDrDataset',
        args={'data_dir': data_dir,
              'ds_len': -1,
              }
    )


def get_dream_synthetic_panda_test_photo(name):
    # dream_synthetic_panda_train_dr_0
    root_dir = osp.expanduser(f'./DREAM/synthetic/panda_synth_test_photo/processed')
    scene = int(name.split("_")[-1])
    data_dir = osp.join(root_dir, f"scene_{scene:06d}")
    return dict(
        factory='DreamSyntheticPandaTestPhotoDataset',
        args={'data_dir': data_dir,
              'ds_len': -1,
              }
    )


def get_baxter_real(name):
    scene = name.split("_")[-1]
    data_dir = osp.expanduser("./baxter_real/baxter-real-dataset/processed")
    scene_dir = osp.join(data_dir, scene)
    return dict(
        factory='BaxterRealDataset',
        args={'data_dir': scene_dir,
              'ds_len': -1,
              }
    )


def get_baxter_real_se(name):
    scene = "_".join(name.split("_")[3:])
    data_dir = "data/baxter_real"
    scene_dir = osp.join(data_dir, scene)
    return dict(
        factory='BaxterRealDataset',
        args={'data_dir': scene_dir,
              'ds_len': -1,
              }
    )


def get_realman(name):
    #     realman/24xxxx-xxxxxx
    items = name.split("/")
    scene = items[1]
    return dict(
        factory='RealmanDataset',
        args={'data_dir': osp.join('data/realsense/', scene),
              'ds_len': -1,
              }
    )


def get_realman_hec_marker_eih(name):
    scene = name[len("realman_hec_marker_eih_"):]
    return dict(
        factory='RealmanHecMarkerEIHDataset',
        args={'data_dir': osp.join('data/realsense/', scene),
              'ds_len': -1,
              }
    )
