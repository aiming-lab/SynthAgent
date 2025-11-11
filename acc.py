from dataclasses import dataclass
import json
from simpleArgParser import parse_args
from loguru import logger
import syn.utils

@dataclass
class Config:
    # input folder
    input: str

def stat_accuracy(execute_status: dict[str, dict]) -> tuple:
    auto_eval_cnt = sum(1 for status in execute_status.values() if status['auto-eval'] != 'NA')
    auto_eval_success = sum(1 for status in execute_status.values() if status['auto-eval'] == 'success')
    total_cnt = len(execute_status)

    return auto_eval_success, auto_eval_cnt, total_cnt

if __name__ == '__main__':
    args: Config = parse_args(Config)
    status = json.load(open(f"{args.input}/tasks_done_unique.json", 'r'))

    envs =  ['shopping', 'shopping_admin', 'reddit', 'gitlab', 'map']
    env2acc = {env: {} for env in envs}
    for env in env2acc.keys():
        env_status = {k: v for k, v in status.items() if k.endswith(f"@{env}")}
        auto_eval_success, auto_eval_cnt, total_cnt = stat_accuracy(env_status)

        env2acc[env] = {
            'auto_eval_success': auto_eval_success,
            'auto_eval_cnt': auto_eval_cnt,
            'total_cnt': total_cnt
        }

    
    total = {k: sum(item[k] for item in env2acc.values()) for k in env2acc['shopping'].keys()}
    envs.append('total')
    env2acc['total'] = total

    accs = {env: env2acc[env]['auto_eval_success'] / env2acc[env]['auto_eval_cnt'] * 100 if env2acc[env]['auto_eval_cnt'] > 0 else 0 for env in env2acc.keys()}
    accs = ";".join(f"{accs[env]:.2f}" for env in envs)

    auto_eval_cnts = ";".join(f"{env2acc[env]['auto_eval_cnt']}" for env in env2acc.keys())
    total_cnts = ";".join(f"{env2acc[env]['total_cnt']}" for env in env2acc.keys())


    logger.info(";".join(envs))
    logger.info(f"Accuracy (%): {accs}")
    logger.info(f"Counts (has auto-eval): {auto_eval_cnts}")
    logger.info(f"Total Counts: {total_cnts}")


    