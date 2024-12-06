import os
import argparse
import sys

from crc.utils.os_utils import archive_runs, make_source_code_snapshot
from dl_ext.timer import EvalTime

from crc.utils.comm import get_rank, get_world_size
from crc.utils.logger import setup_logger
from crc.utils.miscellaneous import save_config


def default_argument_parser(add_help=True, default_config_file=""):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--config-file", '-c', default=default_config_file, metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg, args):
    evaltime = EvalTime(disable=True)
    evaltime("")
    logger = setup_logger(cfg.output_dir)
    evaltime("setup logger")
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("\n" + open(args.config_file, "r").read())
    logger.info("Running with config:\n{}".format(cfg))
    evaltime("info")
    output_config_path = os.path.join(cfg.output_dir, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)
    evaltime("save cfg")
    make_source_code_snapshot(cfg.output_dir)


def setup(args, freeze=True, do_archive=False):
    """
    Create configs and perform basic setups.
    """
    evaltime = EvalTime(disable=True)
    evaltime("")
    from crc.config import cfg
    cfg = cfg.clone()
    evaltime("clone")
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    evaltime("merge")
    if cfg.output_dir == '':
        assert args.config_file.startswith('configs') and args.config_file.endswith('.yaml')
        cfg.output_dir = args.config_file[:-5].replace('configs', 'models')
    # cfg.model.mode = 'train'
    evaltime("output_dir")
    if freeze:
        cfg.freeze()
    os.makedirs(cfg.output_dir, exist_ok=True)
    if do_archive:
        archive_runs(cfg.output_dir)
    # logger = setup_logger(output_dir, get_rank())
    default_setup(cfg, args)
    evaltime("default setup")
    return cfg
