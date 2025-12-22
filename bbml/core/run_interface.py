import argparse
from pathlib import Path
from typing import Any, Callable
import uuid

from bbml.core.utils.config_compose import config_compose
from bbml.utils.storage import download_url


def parse_run_args():
    parser = argparse.ArgumentParser(description="Run training.")
    parser.add_argument("-c", "--config", type=str, action="append", required=True, help="Add YAML configs to compose (path or URL)")
    parser.add_argument("--where", type=str, help="Where to run", default="local")
    args = parser.parse_args()

    if args.where == "local":
        updated_config_paths = []
        for config_path_or_url in args.config:
            if config_path_or_url.startswith("http"):
                config_stem = config_path_or_url.split("/")[-1].split(".")[0]
                local_path = Path("/tmp") / f"{config_stem}_{str(uuid.uuid4())[:8]}.yaml"
                download_url(config_path_or_url, local_path)
                updated_config_paths.append(local_path)
            else:
                updated_config_paths.append(config_path_or_url)
        args.config = updated_config_paths
    
    return args


def run_interface(run_func: Callable[[dict], Any]):
    args = parse_run_args()
    composed_config = config_compose(args.config)
    run_func(composed_config)