import multiprocessing as mp
import os
import json
import math
import random
import copy
import time
from dataclasses import dataclass, field
from loguru import logger


from syn.args import ExeAgentConfig
from syn.tools import (
    tools_get_time,
    tools_elapsed_time_print,
    tools_jsonl_save,
    tools_jsonl_load,
    tools_deserialize_dataclass,
)
from refineagent import ExeAgent
from simpleArgParser import parse_args
from tqdm import tqdm



@dataclass
class MultiExeAgentConfig(ExeAgentConfig):
    """Multi-process ExeAgent configuration that extends ExeAgentConfig"""
    num_processes: int = field(default=4, kw_only=True)
    
    def pre_process(self):
        super().pre_process()
        # Ensure we have at least 1 process
        self.num_processes = max(1, self.num_processes)

    
def run_single_agent(idx: int, shared_single_agent_configs: list[ExeAgentConfig]):
    config = shared_single_agent_configs[idx]
    
    if idx != 0: logger.remove()
    log_file = f"{config.output}/run.log"
    logger.add(log_file, format='<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>', level='DEBUG', colorize=False, rotation=None)

    logger.info(f"Running single ExeAgent={idx}/{len(shared_single_agent_configs)-1} with config=\n{config}")

    agent = ExeAgent(config)
    agent.run_episode()

    logger.info(f"Single ExeAgent={idx}/{len(shared_single_agent_configs)-1} finished with output={config.output}")

def run_monitoring_process(multi_agent_config: MultiExeAgentConfig, shared_single_agent_configs: list[ExeAgentConfig], interval_minutes: int = 10):
    """Monitoring process that gathers results and saves every interval_minutes"""

    logger.remove()
    log_file = f"{multi_agent_config.output}/run_monitor.log"
    logger.add(log_file, format='<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>', level='DEBUG', colorize=False, rotation=None)

    logger.info(f"Starting monitoring process with {interval_minutes} minute intervals")
    
    # Create a temporary MultiExeAgent instance for gathering results
    temp_multiagent = MultiExeAgent(multi_agent_config)
    temp_multiagent.shared_single_agent_configs = shared_single_agent_configs
    
    interval_seconds = interval_minutes * 60
    
    while True:
        try:
            for _ in tqdm(range(interval_seconds), desc="Monitoring process sleeping", disable=True):
                time.sleep(1)

            temp_multiagent.gather_results()
            temp_multiagent.save()

            auto_eval_success, auto_eval_cnt, complete_cnt, total_cnt = temp_multiagent._stat_accuracy(temp_multiagent.tasks_done_unique)

            logger.info(f"Auto-eval accuracy: {auto_eval_success}/{auto_eval_cnt}={auto_eval_success / auto_eval_cnt if auto_eval_cnt > 0 else 0:.4f}\nComplete rate: {complete_cnt}/{total_cnt}={complete_cnt / total_cnt if total_cnt > 0 else 0:.4f}\nExecute progress: {total_cnt}/{len(temp_multiagent.tasks_todo)}={total_cnt / len(temp_multiagent.tasks_todo) if len(temp_multiagent.tasks_todo) > 0 else 0:.4f}")
            
            logger.info(f"Monitoring process: completed gather and save cycle")

        except Exception as e:
            logger.error(f"Monitoring process error: {e}")
            continue

class MultiExeAgent(ExeAgent):
    def __init__(self, config: MultiExeAgentConfig):
        self.multi_agent_config = config
        temp_params = copy.deepcopy(config.__dict__)
        del temp_params['num_processes']
        config_for_single_agent = ExeAgentConfig(**temp_params)
        super().__init__(config_for_single_agent)
        del temp_params['output']
        self.shared_single_agent_configs = [
            ExeAgentConfig(**temp_params, output=f"{self.multi_agent_config.output}/multiagent/{i}")
            for i in range(self.multi_agent_config.num_processes)
        ]

    def gather_results(self,):
        # gather results
        gpt_client_result = {
            'count': 0,
            'call': 0,
            'usage': {

            }
        }

        self.base_unclickable_elements = {}
        self.tasks_done_unique = {}
        self.tasks_todo = []
        self.tasks_done_buffer = []

        for idx, config in enumerate(self.shared_single_agent_configs):

            # gpt stats: gpt_client_token_usage.json
            if os.path.exists(path := f"{config.output}/gpt_client_token_usage.json"):
                temp = json.load(open(path, 'r'))
                gpt_client_result['count'] += temp.get('count', 0)
                gpt_client_result['call'] += temp.get('call', 0)
                for key, value in temp.get('usage', {}).items():
                    if key not in gpt_client_result['usage']:
                        gpt_client_result['usage'][key] = {}
                    
                    for uk, uv in value.items():
                        if uk not in gpt_client_result['usage'][key]:
                            gpt_client_result['usage'][key][uk] = 0
                        gpt_client_result['usage'][key][uk] += uv

            # base_unclickable_elements.jsonl
            if os.path.exists(path := f"{config.output}/base_unclickable_elements.jsonl"):
                temp = tools_deserialize_dataclass(json.load(open(path, 'r')), dict[str, set[tuple[str, tuple[int, int, int, int]]]])
                for key, value in temp.items():
                    if key not in self.base_unclickable_elements:
                        self.base_unclickable_elements[key] = set()
                    self.base_unclickable_elements[key].update(value)
            
            # tasks_done_unique.json
            if os.path.exists(path := f"{config.output}/tasks_done_unique.json"):
                temp = json.load(open(path, 'r'))
            
                self.tasks_done_unique.update(temp)
            
            # tasks_todo.jsonl
            if os.path.exists(path := f"{config.output}/tasks_todo.jsonl"):
                temp = tools_jsonl_load(path)
                self.tasks_todo.extend(temp)
            
        
        # load gpt
        if gpt_client_result['count'] > 0:
            with open(path := f"{self.multi_agent_config.output}/gpt_client_token_usage.json", 'w') as f:
                json.dump(gpt_client_result, f, indent=4)
            self.gpt_client.token_usage.load_from_json(path)

        # tasks todo
        if len(self.tasks_todo) == 0:
            self.tasks_todo = tools_jsonl_load(self.config.tasks_path)
            logger.info(f"load {len(self.tasks_todo)} tasks from {self.config.tasks_path}")
        else:
            logger.info(f"load {len(self.tasks_todo)} existing tasks")

    def distribute_tasks(self):
        random.shuffle(self.tasks_todo)
        tasks_per_process = math.ceil(len(self.tasks_todo) / self.multi_agent_config.num_processes)
        for i in range(self.multi_agent_config.num_processes):
            config = self.shared_single_agent_configs[i]
            os.makedirs(config.output, exist_ok=True)
            start = i * tasks_per_process
            end = (i + 1) * tasks_per_process
            task_chunk = self.tasks_todo[start:end]
            if task_chunk:
                tools_jsonl_save(task_chunk, f"{config.output}/tasks_todo.jsonl")

            json.dump(self.tasks_done_unique, open(f"{config.output}/tasks_done_unique.json", 'w'), indent=4)    

    def save(self,):
        for idx, config in enumerate(self.shared_single_agent_configs):

            # screenshots
            # already saved to the parent folder

            # run.log
            os.system(f"cp {config.output}/run.log {self.multi_agent_config.output}/run_{idx}.log")
        
        super().save()

    def load(self,):
        super().load()

    def run_episode(self,):
        error_code = 0
        start_time = tools_get_time()
        logger.info(f"Starting MultiExeAgent with {self.multi_agent_config.num_processes} processes")
        logger.info(f"Config: {self.multi_agent_config}")
        logger.info(f"Shared output directory: {self.multi_agent_config.output}")
        
        self.gather_results()
        self.save()
        self.load()
        self.distribute_tasks()

        processes = []
        
        
        # Start worker processes
        for process_id in range(self.multi_agent_config.num_processes):
            p = mp.Process(
                target=run_single_agent,
                args=(process_id, self.shared_single_agent_configs)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started process {process_id} with PID {p.pid}")
        
        # Start monitoring process
        monitoring_process = mp.Process(
            target=run_monitoring_process,
            args=(self.multi_agent_config, self.shared_single_agent_configs, 10)
        )
        monitoring_process.start()
        logger.info(f"Started monitoring process with PID {monitoring_process.pid}")

        for idx, p in enumerate(processes):
            p.join()
            logger.info(f"MultiExeagent Process {idx}/{len(processes)-1} pid={p.pid} finished with exit code {p.exitcode}")
            if p.exitcode != 0:
                error_code = p.exitcode

        monitoring_process.kill()

        # final
        self.gather_results()
        self.save()
        # print
        auto_eval_success, auto_eval_cnt, complete_cnt, total_cnt = self._stat_accuracy(self.tasks_done_unique)
        logger.info(f"Auto-eval accuracy: {auto_eval_success}/{auto_eval_cnt}={auto_eval_success / auto_eval_cnt if auto_eval_cnt > 0 else 0:.4f}\nComplete rate: {complete_cnt}/{total_cnt}={complete_cnt / total_cnt if total_cnt > 0 else 0:.4f}\nExecute progress: {total_cnt}/{len(self.tasks_todo)}={total_cnt / len(self.tasks_todo) if len(self.tasks_todo) > 0 else 0:.4f}")

        elapsed_time = tools_elapsed_time_print(start_time)
        logger.info(f"MultiExeAgent with config={self.multi_agent_config} completed! Started at {start_time}, Elapsed time: {elapsed_time}")

        return error_code


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)    
    args: MultiExeAgentConfig = parse_args(MultiExeAgentConfig)
    multiagent = MultiExeAgent(args)
    retry_cnt = 3
    error_code = 0
    while retry_cnt > 0:
        retry_cnt -= 1
        error_code = multiagent.run_episode()

        if error_code == 0:
            break
        else:
            logger.error(f"MultiExeAgent run_episode failed with error_code={error_code}, retrying... {retry_cnt} retries left")
    
    logger.info(f"MultiExeAgent finished with error_code={error_code}, exiting...")
