import numpy as np
import sapien.core as sapien
import transforms3d
import trimesh

from crc.structures.xarm_mapping import link_name_mesh_path_mapping
from crc.utils import utils_3d
from crc.utils.utils_3d import Rt_to_pose
from crc.utils.vis3d_ext import Vis3D


class XArmModel:
    def __init__(self, urdf_path):
        self.engine = sapien.Engine()

        self.scene = self.engine.create_scene()

        loader = self.scene.create_urdf_loader()

        builder = loader.load_file_as_articulation_builder(urdf_path)

        # if add_dummy_rotation or add_dummy_translation:
        #     dummy_joint_indicator = (add_dummy_translation,) * 3 + (add_dummy_rotation,) * 3
        #
        #     add_dummy_free_joint(builder, dummy_joint_indicator)

        self.robot: sapien.Articulation = builder.build(fix_root_link=True)

        self.robot.set_pose(sapien.Pose())

        self.robot.set_qpos(np.zeros(self.robot.dof))

        self.scene.step()
        # self._add_constraints()

        self.model: sapien.PinocchioModel = self.robot.create_pinocchio_model()

    def _add_constraints(self):
        # Add constraints
        outer_knuckle = next(
            j for j in self.robot.get_active_joints() if j.name == "right_outer_knuckle_joint"
        )
        outer_finger = next(
            j for j in self.robot.get_active_joints() if j.name == "right_finger_joint"
        )
        inner_knuckle = next(
            j for j in self.robot.get_active_joints() if j.name == "right_inner_knuckle_joint"
        )

        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        T_pw = pad.pose.inv().to_transformation_matrix()
        p_w = (
                outer_finger.get_global_pose().p
                + inner_knuckle.get_global_pose().p
                - outer_knuckle.get_global_pose().p
        )
        T_fw = lif.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

        right_drive = self.scene.create_drive(lif, sapien.Pose(p_f), pad, sapien.Pose(p_p))
        right_drive.lock_motion(1, 1, 1, 0, 0, 0)

        outer_knuckle = next(
            j for j in self.robot.get_active_joints() if j.name == "drive_joint"
        )
        outer_finger = next(
            j for j in self.robot.get_active_joints() if j.name == "left_finger_joint"
        )
        inner_knuckle = next(
            j for j in self.robot.get_active_joints() if j.name == "left_inner_knuckle_joint"
        )

        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        T_pw = pad.pose.inv().to_transformation_matrix()
        p_w = (
                outer_finger.get_global_pose().p
                + inner_knuckle.get_global_pose().p
                - outer_knuckle.get_global_pose().p
        )
        T_fw = lif.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

        left_drive = self.scene.create_drive(lif, sapien.Pose(p_f), pad, sapien.Pose(p_p))
        left_drive.lock_motion(1, 1, 1, 0, 0, 0)

        left_link = next(l for l in self.robot.get_links() if l.name == "left_outer_knuckle")
        right_link = next(l for l in self.robot.get_links() if l.name == "right_outer_knuckle")
        self.scene.create_gear(left_link, sapien.Pose(), right_link, sapien.Pose())

        # Cache robot link ids
        self.robot_link_ids = [link.get_id() for link in self.robot.get_links()]

    def compute_forward_kinematics(self, qpos, link_index):
        self.model.compute_forward_kinematics(np.array(qpos))

        return self.model.get_link_pose(link_index)

    def release(self):
        self.scene = None

        self.engine = None


def main():
    sk = XArmModel("data/xarm7_with_gripper.urdf")

    qpos = np.random.randn(13)
    # qpos = np.array([0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0.0, 0.0])
    qpos = np.array([0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0.0, 0.0])
    sk.robot.set_qpos(qpos)
    # qpos = sk.robot.get_qpos()
    print()
    vis3d = Vis3D(xyz_pattern=('x', 'y', 'z'), out_folder="dbg", sequence_name="xarm_model")
    sk.robot.set_qpos()
    links = sk.robot.get_links()
    for link in range(len(links)):
        link_name = links[link].name
        pq = sk.compute_forward_kinematics(qpos, link)
        pose = Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
        mesh_path = link_name_mesh_path_mapping[link_name]
        if mesh_path == "": continue
        mesh = trimesh.load_mesh(mesh_path)
        vis3d.add_mesh(utils_3d.transform_points(mesh.vertices, pose), mesh.faces)

    # for link in range(20):
    #     print('link', link)
    #     sk.compute_forward_kinematics(qpos, link)


if __name__ == '__main__':
    main()
