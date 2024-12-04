import os
import subprocess

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
from crc.utils.timer import EvalTime


def run_colmap(basedir, match_type, remote=False, cam_model='OPENCV'):
    evaltime = EvalTime()
    evaltime("begin")
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    # import time
    # start = time.time()
    if remote:
        use_gpu = '0'
    else:
        use_gpu = '1'
    feature_extractor_args = [
        'colmap', 'feature_extractor',
        '--ImageReader.camera_model', cam_model,
        '--SiftExtraction.num_threads', '16',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.single_camera', '1',
        '--SiftExtraction.use_gpu', use_gpu,
        '--SiftExtraction.max_num_features', "1500",
    ]
    feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
    logfile.write(feat_output)
    print('Features extracted')
    evaltime("Colmap feature extraction")
    # start = time.time()
    exhaustive_matcher_args = [
        'colmap', match_type,
        '--database_path', os.path.join(basedir, 'database.db'),
        '--SiftMatching.use_gpu', use_gpu,
        '--SiftMatching.num_threads', '16',
    ]

    match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True))
    logfile.write(match_output)
    print('Features matched')
    evaltime("Colmap feature matching")
    # start = time.time()
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)
    # hierarchical_mapper
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', os.path.join(basedir, 'sparse'),  # --export_path changed to --output_path in colmap 3.6
        '--Mapper.num_threads', '16',
        '--Mapper.init_min_tri_angle', '4',
        '--Mapper.extract_colors', '0',
    ]

    map_output = (subprocess.check_output(mapper_args, universal_newlines=True))
    logfile.write(map_output)
    print('Sparse map created')
    evaltime("Colmap feature mapping")
    # start = time.time()
    model_convert_args = [
        'colmap', 'model_converter',
        '--input_path', os.path.join(basedir, 'sparse/0'),
        '--output_path', os.path.join(basedir, 'sparse/0'),
        '--output_type', 'TXT'
    ]
    map_output = subprocess.check_output(model_convert_args, universal_newlines=True)
    logfile.write(map_output)
    # print(time.time() - start)
    # start = time.time()

    '''model_ba_args = [
            'colmap', 'bundle_adjuster',
            '--input_path',os.path.join(basedir, 'sparse/0'),
            '--output_path', os.path.join(basedir, 'sparse/0'),
            '--BundleAdjustment.refine_principal_point','1'
    ]
    map_output =subprocess.check_output(model_ba_args, universal_newlines=True)
    logfile.write(map_output)
    print('finish bundle adjustment')'''
    '''model_undistort_args = [
            'colmap', 'image_undistorter',
            '--input_path',os.path.join(basedir, 'sparse/0'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', basedir
    ]
    map_output =subprocess.check_output(model_undistort_args, universal_newlines=True
    logfile.write(map_output)
    print(time.time()-start)
    start = time.time()'''
    logfile.close()

    print('Finished running COLMAP, see {} for logs'.format(logfile_name))
