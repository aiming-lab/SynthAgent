from syn.tools import tools_jsonl_load, tools_jsonl_save, tools_deserialize_dataclass
from syn.data import ExplorationTraj
from syn.consts import const_target_port_placeholder, const_is_load_screenshot_image
import os
import random
import re
from multiprocessing import Pool, Manager, Lock
from simpleArgParser import parse_args
from dataclasses import dataclass

@dataclass
class Config:
    start_folder: str
    output: str
    limit_per_env: int = 10000
    endswith: str = ""


os.environ['SHOPPING'] = f"http://127.0.0.1:{const_target_port_placeholder}"
os.environ['SHOPPING_ADMIN'] = f"http://127.0.0.1:{const_target_port_placeholder}/admin"
os.environ['REDDIT'] = f"http://127.0.0.1:{const_target_port_placeholder}"
os.environ['GITLAB'] = f"http://127.0.0.1:{const_target_port_placeholder}/explore"
os.environ['MAP'] = f"http://127.0.0.1:{const_target_port_placeholder}"
os.environ[const_is_load_screenshot_image] = '0'  # do not load screenshot image for speed


def replace_with_port(url: str, env: str):
    url = url.strip('/')
    current_port = re.findall('127.0.0.1:\d+', url)
    if current_port:
        port = current_port[0]
    else:
        env_url = os.environ[env.upper()]
        print('error: cannot find port in url', url, 'thus using default url = ', env_url)
        return env_url
    return url.replace(port, f'127.0.0.1:{const_target_port_placeholder}') 

def load_data(folder: str) -> tuple[list[str], str]:
    db_path = f'{folder}/db.simplified.jsonl'
    db = tools_jsonl_load(db_path)

    status2idx = {}

    for idx, item in enumerate(db):
        sta = item['status']
        if sta not in status2idx:
            status2idx[sta] = set()
        status2idx[sta].add(task := item["high_level_tasks"][-1])

    unique = {task for task in status2idx.get('END', set())}
    return list(unique)

def process_environment(args):
    env, config, shared_unique, lock = args
    config: Config
    os.environ['DISABLE_FROM_DICT'] = '1'  # disable for faster serialization
    
    results = []
    folder = f'{config.start_folder}/{env}'
    if not os.path.exists(folder): return results
    for sub in os.listdir(folder):
        if sub.endswith(config.endswith):
            path = f"{folder}/{sub}"
            db_path = f'{path}/db.jsonl'
            data = tools_deserialize_dataclass(tools_jsonl_load(db_path), list[ExplorationTraj])
            random.shuffle(data)
            cnt = 0
            for exptaj in data:
                task = exptaj.high_level_tasks[-1].task
                task_env = f"{task}@{env}"
                
                # Use lock to safely check and update shared list
                with lock:
                    if task_env not in shared_unique:
                        shared_unique.append(task_env)
                        is_unique = True
                    else:
                        is_unique = False
                
                if is_unique:
                    start_url = exptaj.high_level_tasks[-1].trajectories[0].curr_state.raw_state.url
                    start_url = replace_with_port(start_url, env)
                    results.append(
                        {
                            'task': task,
                            'start_url': start_url,
                            'sites': [env],
                        }
                    )
                    cnt += 1
                    if cnt >= config.limit_per_env: break
        
            print(f"Loaded {cnt} unique tasks from {path}")
            for i in range(1, min(3, len(results))):
                print(i, results[-i])
            print('-'*100)
            break
    
    return results



if __name__ == '__main__':
    args: Config = parse_args(Config)

    results = []
    # Use Manager to create a shared list for tracking unique tasks across processes
    with Manager() as manager:
        shared_unique = manager.list()
        lock = manager.Lock()
        
        # Prepare arguments for each environment
        envs = ['reddit', 'shopping', 'shopping_admin', 'gitlab', 'map']
        args_list = [(env, args, shared_unique, lock) for env in envs]
        
        # Use multiprocessing to process environments in parallel
        with Pool(processes=len(envs)) as pool:
            env_results = pool.map(process_environment, args_list)
        
        # Combine results from all environments
        existed_env = 0
        for env_result in env_results:
            results.extend(env_result)
            if len(env_result) > 0:
                existed_env += 1

    print('got', len(results), f'unique tasks for {existed_env} envs\nsaving to={args.output}')

    tools_jsonl_save(results, args.output)