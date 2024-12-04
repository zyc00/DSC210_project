from crc.registry import TRAINERS
from crc.trainer.base import BaseTrainer
from crc.trainer.rbsolver_trainer import RBSolverTrainer
from crc.trainer.rbsolve_iter import RBSolverIterTrainer
from crc.trainer.rbsolve_iter_eih import RBSolverIterEiHTrainer
from crc.trainer.hand_eye_solver import HandEyeSolverTrainer
from crc.trainer.rbsolve_iter_dvqf import RBSolverIterTrainerDVQF
from crc.trainer.solve_iter_nbv import SolverIterTrainerNBV
from crc.trainer.rbsolve_iter_realman import RBSolverIterRealmanTrainer


@TRAINERS.register('base')
def build_base_trainer(cfg):
    return BaseTrainer(cfg)


@TRAINERS.register('rbsolver')
def build_rbsolver_trainer(cfg):
    return RBSolverTrainer(cfg)


@TRAINERS.register('rbsolver_iter')
def build_rbsolveriter_trainer(cfg):
    return RBSolverIterTrainer(cfg)


@TRAINERS.register('rbsolver_iter_realman')
def build_rbsolveriter_realman_trainer(cfg):
    return RBSolverIterRealmanTrainer(cfg)


@TRAINERS.register('rbsolver_iter_dvqf')
def build_rbsolveriter_trainer(cfg):
    return RBSolverIterTrainerDVQF(cfg)


@TRAINERS.register('solver_iter_nbv')
def build_solveriter_trainer(cfg):
    return SolverIterTrainerNBV(cfg)


@TRAINERS.register('rbsolver_iter_eih')
def build_rbsolveritereih_trainer(cfg):
    return RBSolverIterEiHTrainer(cfg)


@TRAINERS.register('hand_eye_solver')
def build_hand_eye_solver_trainer(cfg):
    return HandEyeSolverTrainer(cfg)


def build_trainer(cfg) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](cfg)
