import sapien.core as sapien
import warnings

import numpy as np
import sapien


class SAPIENKinematicsModelStandalone:

    def __init__(self, urdf_path, srdf_path=None):
        self.engine = sapien.Engine()

        self.scene = self.engine.create_scene()

        loader = self.scene.create_urdf_loader()
        # loader.scale = 10
        builder = loader.load_file_as_articulation_builder(urdf_path, srdf_path)
        # builder = loader.load(urdf_path)

        self.robot: sapien.Articulation = builder.build(fix_root_link=True)

        self.robot.set_pose(sapien.Pose())

        self.robot.set_qpos(np.zeros(self.robot.dof))

        self.scene.step()

        self.model: sapien.PinocchioModel = self.robot.create_pinocchio_model()

    def compute_forward_kinematics(self, qpos, link_index) -> sapien.Pose:
        # assert len(qpos) == self.robot.dof, f"qpos {len(qpos)} != {self.robot.dof}"
        if len(qpos) != self.robot.dof:
            warnings.warn("qpos length not match")
        qpos = np.array(qpos).tolist() + [0] * (self.robot.dof - len(qpos))
        self.model.compute_forward_kinematics(np.array(qpos))

        return self.model.get_link_pose(link_index)

    def compute_inverse_kinematics(self, link_index, pose, initial_qpos, *args, **kwargs):
        if len(initial_qpos) != self.robot.dof:
            warnings.warn("initial_qpos length not match")
        initial_qpos = np.array(initial_qpos).tolist() + [0] * (self.robot.dof - len(initial_qpos))

        return self.model.compute_inverse_kinematics(link_index, pose, initial_qpos=initial_qpos,
                                                     *args, **kwargs)

    def release(self):
        self.scene = None

        self.engine = None


def main():
    sk = SAPIENKinematicsModelStandalone("data/sapien_packages/xarm7/xarm_urdf/xarm7_gripper.urdf")
    qpos = np.deg2rad([-37.6, -7.2, -22.7, 27.5, -4.9, 34.2, -56.1])
    print(sk.compute_forward_kinematics(qpos, 8).to_transformation_matrix())
    # print(sk.compute_forward_kinematics(qpos))


if __name__ == '__main__':
    main()
