import os

import numpy as np
import cv2


def get_K(camera_id):
    cmd = "curl -X POST http://192.168.31.175:2525/capture -o tmp.zip"
    os.system('rm -rf tmp/*')
    os.system(cmd)
    outdir = 'tmp'
    os.system(f"unzip -o tmp.zip -d {outdir}")
    # img = cv2.imread(f"{outdir}/color_image_{camera_id}.png")
    # qpos = np.loadtxt(f"{outdir}/qpos.txt")
    K = np.loadtxt(f"{outdir}/K_{camera_id}.txt")
    return K


def get_color_image(camera_id):
    cmd = "curl -X POST http://192.168.31.175:2525/capture -o tmp.zip"
    os.system('rm -rf tmp/*')
    os.system(cmd)
    outdir = 'tmp'
    os.system(f"unzip -o tmp.zip -d {outdir}")
    img = cv2.imread(f"{outdir}/color_image_{camera_id}.png")
    # qpos = np.loadtxt(f"{outdir}/qpos.txt")
    # K = np.loadtxt(f"{outdir}/K_{camera_id}.txt")
    return img


def get_depth_image(camera_id):
    cmd = "curl -X POST http://192.168.31.175:2525/capture -o tmp.zip"
    os.system('rm -rf tmp/*')
    os.system(cmd)
    outdir = 'tmp'
    os.system(f"unzip -o tmp.zip -d {outdir}")
    img = cv2.imread(f"{outdir}/depth_image_{camera_id}.png", 2).astype(np.float32) / 1000.0
    # qpos = np.loadtxt(f"{outdir}/qpos.txt")
    # K = np.loadtxt(f"{outdir}/K_{camera_id}.txt")
    return img


def get_qpos():
    cmd = "curl -X POST http://192.168.31.175:2525/capture -o tmp.zip"
    os.system(cmd)
    outdir = 'tmp'
    os.system("rm -rf tmp/*")
    os.system(f"unzip -o tmp.zip -d {outdir}")
    # img = cv2.imread(f"{outdir}/color_image_{camera_id}.png")
    qpos = np.loadtxt(f"{outdir}/qpos.txt")
    # K = np.loadtxt(f"{outdir}/K_{camera_id}.txt")
    return qpos


def set_qpos(qpos, speed=1, is_radian=True, ignore_left=False, ignore_right=False):
    assert len(qpos) == 15
    if is_radian is False:
        qpos = np.deg2rad(qpos)
    np.savetxt('tmp_qpos.txt', qpos)
    cmd = (
        f"curl -X POST http://192.168.31.175:2525/move_to_qpos -F file=@tmp_qpos.txt -F speed={speed} -F ignore_left={int(ignore_left)} -F ignore_right={int(ignore_right)}")
    os.system(cmd)
    os.system("rm tmp_qpos.txt")


def drive_to_qpos(qposes, speed=1, is_radian=True):
    assert isinstance(qposes, np.ndarray)
    assert qposes.ndim == 2
    assert qposes.shape[1] == 15
    if is_radian is False:
        qposes = np.deg2rad(qposes)
    np.savetxt('tmp_qposes.txt', qposes)
    cmd = (f"curl -X POST http://192.168.31.175:2525/move_to_qposes_by_file -F file=@tmp_qposes.txt -F speed={speed}")
    os.system(cmd)
    os.system("rm tmp_qposes.txt")


def set_gripper(width, force=1):
    cmd = f"curl -X POST http://192.168.31.175:2525/set_gripper -F width={width} -F force={force}"
    os.system(cmd)


def main():
    qpos = get_qpos()
    print(np.array2string(qpos, separator=',', precision=6, suppress_small=True))


if __name__ == '__main__':
    main()
