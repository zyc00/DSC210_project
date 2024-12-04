import csv
import numpy as np
import os
import os.path as osp
import random
import sys
import warnings

import loguru
import torch
import torch.multiprocessing
from dl_ext.timer import EvalTime

from crc.config import cfg
from crc.engine.defaults import default_argument_parser
from crc.trainer.build import build_trainer
from crc.utils.comm import synchronize, get_rank, get_world_size
from crc.utils.logger import setup_logger
from crc.utils.os_utils import archive_runs, make_source_code_snapshot, deterministic
from crc.utils.vis3d_ext import Vis3D
from crc.evaluators import build_evaluators
from crc.visualizers import build_visualizer
from crc.utils.os_utils import isckpt

warnings.filterwarnings('once')
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def get_preds(trainer):
    world_size = get_world_size()
    distributed = world_size > 1
    if distributed:
        trainer.to_distributed()
    preds = trainer.get_preds()
    trainer.to_base()
    return preds


def eval_one_ckpt(trainer):
    trainer.resume()
    preds = get_preds(trainer)
    if get_rank() == 0:
        if cfg.test.do_evaluation:
            evaluators = build_evaluators(cfg)
            for evaluator in evaluators:
                evaluator(preds, trainer)
        if cfg.test.do_visualization:
            visualizer = build_visualizer(cfg)
            visualizer(preds, trainer)


def eval_all_ckpts(trainer):
    if cfg.test.do_evaluation:
        evaluators = build_evaluators(cfg)
    if cfg.test.do_visualization:
        visualizer = build_visualizer(cfg)
    if cfg.test.ckpt_dir == '':
        ckpt_dir = cfg.output_dir
    else:
        ckpt_dir = cfg.test.ckpt_dir
    tb_writer = trainer.tb_writer
    csv_results = {'fname': []}
    for fname in sorted(os.listdir(ckpt_dir)):
        if isckpt(fname) and int(fname[-10:-4]) > cfg.test.eval_all_min:
            cfg.defrost()
            cfg.solver.load_model = osp.join(ckpt_dir, fname)
            cfg.solver.load = ''
            # cfg.solver.load = fname[:-4]
            cfg.freeze()
            trainer.resume()
            preds = get_preds(trainer)
            all_metrics = {}
            if cfg.test.do_evaluation:
                for evaluator in evaluators:
                    eval_res = evaluator(preds, trainer)
                    all_metrics.update(eval_res)
            # save results
            csv_results['fname'].append(fname)
            for k, v in all_metrics.items():
                tb_writer.add_scalar(f'eval/{k.replace("@", "_")}', v, int(fname[-10:-4]))
                if k not in csv_results: csv_results[k] = []
                csv_results[k].append(v)
            if cfg.test.do_visualization:
                visualizer(preds, trainer)
    # write csv file
    csv_out_path = osp.join(trainer.output_dir, 'inference', cfg.datasets.test, 'eval_all_ckpt.csv')
    os.makedirs(osp.dirname(csv_out_path), exist_ok=True)
    with open(csv_out_path, 'w', newline='') as csvfile:
        fieldnames = list(csv_results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(csv_results['fname'])):
            writer.writerow({k: v[i] for k, v in csv_results.items()})


def train(cfg, local_rank, distributed, resume):
    trainer = build_trainer(cfg)
    if resume:
        trainer.resume()
    if distributed:
        trainer.to_distributed()
    trainer.fit()
    return trainer


def main():
    parser = default_argument_parser()
    # parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--init_method', default='env://', type=str)
    parser.add_argument('--no_archive', default=False, action='store_true')
    args = parser.parse_args()
    evaltime = EvalTime()
    evaltime("")
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.init_method,
            rank=local_rank, world_size=num_gpus)
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.output_dir == '':
        assert args.config_file.startswith('configs') and args.config_file.endswith('.yaml')
        cfg.output_dir = args.config_file[:-5].replace('configs', 'models')
    if 'PYCHARM_HOSTED' in os.environ:
        loguru.logger.warning("fix random seed!!!!")
        cfg.dataloader.num_workers = 0
        cfg.backup_src = False
        random_seed = 999
        np.random.seed(random_seed)
        random.seed(random_seed)
        # torch.use_deterministic_algorithms(True)
        torch.random.manual_seed(random_seed)
    if cfg.solver.load != '':
        args.no_archive = True
    cfg.mode = 'train'
    cfg.freeze()
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    Vis3D.default_out_folder = osp.join(output_dir, 'dbg')

    # archive previous runs
    if not args.no_archive:
        archive_runs(output_dir)
    logger = setup_logger(output_dir)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    path = os.path.join(output_dir, "config.yaml")
    logger.info("Running with full config:\n{}".format(cfg.dump(), ".yaml"))
    with open(path, "w") as f:
        f.write(cfg.dump())
    if get_rank() == 0 and cfg.backup_src is True:
        make_source_code_snapshot(output_dir)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.remove(2)
        logger.info(config_str)
        format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        logger.add(sys.stdout, format=format, level="INFO")
    logger.info("Running with config:\n{}".format(cfg))
    # if cfg.deterministic is True:
    #     deterministic()
    # seed_everything(777)
    # np.random.default_rng(777)
    # np.random.seed(0)
    # torch.random.manual_seed(0)
    # torch.cuda.manual_seed(0)
    evaltime("init")
    print(cfg)
    exit()
    trainer = train(cfg, local_rank, args.distributed, cfg.solver.resume)

    if cfg.do_eval_after_train:
        evaltime("train")
        if cfg.test.eval_all:
            eval_all_ckpts(trainer)
        else:
            eval_one_ckpt(trainer)
        evaltime("eval")


if __name__ == "__main__":
    main()
