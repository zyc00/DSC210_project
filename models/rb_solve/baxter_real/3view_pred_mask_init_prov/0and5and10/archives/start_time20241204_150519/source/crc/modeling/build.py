from crc.modeling.models.rb_solve.rb_solver import RBSolver
from crc.modeling.models.rb_solve.rb_solver_realman_head import RBSolverRealmanHead
from crc.modeling.models.rb_solve.rb_solver_realman_lefthandcam_b2b import RBSolverRealmanLeftHandCamB2B
from crc.modeling.models.rb_solve.rb_solver_realman_righthandcam_b2b import RBSolverRealmanRightHandCamB2B
from crc.modeling.models.rb_solve.rb_mesh_pose_solver import RBMeshPoseSolver
from crc.modeling.models.rb_solve.space_explorer import SpaceExplorer
from crc.modeling.models.hand_eye_solver import HandEyeSolver
from crc.modeling.models.hand_eye_solver_eih import HandEyeSolverEIH
from crc.modeling.models.dream.resnet import DreamNetwork
from crc.modeling.models.hand_eye_solver_gn import HandEyeSolverGN

_META_ARCHITECTURES = {
    'RBSolver': RBSolver,
    'RBSolverRealmanHead': RBSolverRealmanHead,
    'RBSolverRealmanLeftHandCamB2B': RBSolverRealmanLeftHandCamB2B,
    'RBSolverRealmanRightHandCamB2B': RBSolverRealmanRightHandCamB2B,
    'RBMeshPoseSolver': RBMeshPoseSolver,
    "SpaceExplorer": SpaceExplorer,
    "HandEyeSolver": HandEyeSolver,
    "HandEyeSolverEIH": HandEyeSolverEIH,
    "DreamNetwork": DreamNetwork,
    'HandEyeSolverGN': HandEyeSolverGN
}


def build_model(cfg):
    print("building model...", end='\r')
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    model = meta_arch(cfg)
    return model
