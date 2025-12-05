from syn.base_explore import Explorer
from syn.data import StateInfo, Action, RawState, ActionType, HighLevelTask, Element, LowLevelTask, ExplorationTraj, ExplorationTrajStatus, ActionExecuteStatus, LowTaskStatus

from syn.args import ExeAgentConfig
from syn.prompts import (
    prompt_action_from_observation_adapt_from_webarena,
    prompt_refine_during_execution,
)
from syn.tools import (
    tools_get_time,
    tools_elapsed_time_print,
    tools_jsonl_save,
    tools_jsonl_load,
    tools_serialize_dataclass,
    tools_deserialize_dataclass,
    tools_robust_json_loads,
)
from syn.gpt import GPTClient
from syn.utils import stat_time, stat_time_block
from syn.consts import (
     const_undefined_category,
     const_uninteractive_category,
     const_target_port_placeholder,
)



import random
import json
import time
import os
import numpy as np
from loguru import logger
from simpleArgParser import parse_args
from tqdm import tqdm
from collections import defaultdict
import re
import copy



def replace_with_env(item: dict, env_target_port: str) -> dict:

    raw = json.dumps(item)

    REDDIT = os.environ["REDDIT"]
    SHOPPING = os.environ["SHOPPING"]
    SHOPPING_ADMIN = os.environ["SHOPPING_ADMIN"]
    GITLAB = os.environ["GITLAB"]
    MAP = os.environ["MAP"]
    HOMEPAGE = os.environ["HOMEPAGE"]

    raw = raw.replace("__GITLAB__", GITLAB)
    raw = raw.replace("__REDDIT__", REDDIT)
    raw = raw.replace("__SHOPPING__", SHOPPING)
    raw = raw.replace("__SHOPPING_ADMIN__", SHOPPING_ADMIN)
    raw = raw.replace("__MAP__", MAP)

    task_dict = json.loads(raw)
    task_dict['start_url'] = task_dict['start_url'].replace(const_target_port_placeholder, env_target_port).strip('/')

    return task_dict


class ExeAgent(Explorer):
    def __init__(self, config: ExeAgentConfig):
        super().__init__(config)
        self.config: ExeAgentConfig = config

        self.tasks_done_buffer: list[HighLevelTask] = []  # Store trajectories of executed tasks
        self.tasks_todo: list[dict] = []  # High-level tasks that to be executed {'task': str, 'start_url': str}
        self.tasks_done_unique: dict[str, str] = {}  # task: status

        self.eval_gpt_client = GPTClient(provider=config.eval_gpt.provider, base_url=config.eval_gpt.openai_api_base, api_key=config.eval_gpt.openai_api_key)
        self.load()


    def save(self):
        super().save()
        tools_jsonl_save(self.tasks_todo, f"{self.config.output}/tasks_todo.jsonl")
        tools_jsonl_save(tools_serialize_dataclass(self.tasks_done_buffer), f"{self.config.output}/tasks_done.jsonl", append=True)
        self.tasks_done_buffer = []

        with open(f"{self.config.output}/tasks_done_unique.json", 'w') as f:
            json.dump(self.tasks_done_unique, f, indent=4)


    def load(self):
        super().load()
        if os.path.exists(path := f"{self.config.output}/tasks_todo.jsonl"):
            self.tasks_todo = tools_jsonl_load(path)
            logger.info(f"Loaded {len(self.tasks_todo)} high-level tasks to be executed from {path}, skipping loading from input={self.config.tasks_path}")
        else:
            self.tasks_todo = tools_jsonl_load(self.config.tasks_path)
            logger.info(f"Loaded {len(self.tasks_todo)} high-level tasks to be executed from input={self.config.tasks_path}")
        
        random.shuffle(self.tasks_todo)

        if os.path.exists(path := f"{self.config.output}/tasks_done_unique.json"):
            self.tasks_done_unique = json.load(open(path, 'r'))

            logger.info(f"Loaded {len(self.tasks_done_unique)} high-level tasks DONE from {path}")


    def _are_screenshots_identical(self, screenshot1: np.ndarray, screenshot2: np.ndarray):
        """Check if two screenshots are identical"""
        return np.array_equal(screenshot1, screenshot2)

    @stat_time
    def _cot_step(self, task: str, current_state: StateInfo, previous_traj: list[LowLevelTask]) -> LowLevelTask:
        excluding_elements = self.base_unclickable_elements.get(current_state.raw_state.url, set())
        
        message = prompt_action_from_observation_adapt_from_webarena(
            url=current_state.raw_state.url,
            page_context=self._format_page_context(current_state),
            elements=self._format_elements_for_llm(current_state.elements, excluding_elements=excluding_elements),
            previous_state_action=self._format_previous_observation_and_action(previous_traj, last_k=self.config.history_last_k),
            screenshot=current_state.raw_state.screenshot if self.config.enable_vision else None,
            high_level_task=task,
            history_last_k=self.config.history_last_k,
        )
        eleid2element = {e.id: e for e in current_state.elements}

        # logger.debug(f"cot_step message=\n{message[0]['content'][0]['text']}")

        failed_low_level_task = LowLevelTask(
            task="failed during cot_step",
            action=Action(
                element=None,
                action_type=ActionType.STOP,
                value="error during cot_step",
            ),
            curr_state=current_state,
            task_status=LowTaskStatus.IN_PROGRESS,
        )

        error_msg = None

        try:
            response = self.gpt_client.request(
                messages=message,
                json_mode=True,
                **self.config.gpt.__dict__,
            )
            response_text = response.message.content
            data = tools_robust_json_loads(response_text)
            
            if not isinstance(data, dict):
                error_msg = f"Expected dict, got {type(data)} from response: {response_text}"
            
            if error_msg is None and not ('next_action' in data and isinstance(data['next_action'], dict) and 'action' in data['next_action'] and isinstance(data['next_action']['action'], dict)):
                error_msg = f"Expected 'next_action' with 'action' dict in response: {response_text}"

            if error_msg is None:
                next_action = data['next_action']
                if not all(x in next_action['action'] for x in ['type', 'element_id', 'value']):
                    error_msg = f"error in parsing next_action from response: {response_text}"

            if isinstance(error_msg, str):
                logger.error(f"Error in parsing cot response: {error_msg}\nresponse={response_text}")
                failed_low_level_task.action.value = error_msg
                return failed_low_level_task

            action_dict = next_action['action']
            action_type = ActionType(action_dict['type'].lower())

            if 'state_observation_summary' in data:
                state_summary = data['state_observation_summary']
            else:
                state_summary = None
            current_state.summary = str(state_summary)
            
            if Action._is_required_element(action_type):
                eleid = str(action_dict['element_id'])
                if eleid not in eleid2element:
                    error_msg = f"Element ID {eleid} not found in current state elements. Available IDs: {list(eleid2element.keys())}\n{current_state.raw_state.accessibility_tree}\ninput_gpt_message={message[0]['content'][0]}\nresponse_text={response_text}"
                    failed_low_level_task.action.value = error_msg
                    logger.error(error_msg)
                    return failed_low_level_task
                else:    
                    target_element = eleid2element[eleid]
            else:
                target_element = None

            action = Action(
                action_type=action_type,
                element=target_element,
                value=action_dict.get('value', None),
            )
            low_level_task = next_action.get('low-level_instruction', action.get_action_str())
            
            low_level_task = LowLevelTask(
                task=low_level_task,
                curr_state=current_state,
                action=action,
                task_status=LowTaskStatus.IN_PROGRESS,
                reasoning=data.get('reasoning'),
            )

            logger.debug(f"next_action low_level_task={low_level_task.task}, action={low_level_task.action}, state_summary={current_state.summary}\nresponse_text={response_text}")

        except Exception as e:
            error_msg = f"error during cot_step for task={task} with error={e}\nresponse={response}"

        if isinstance(error_msg, str):
            failed_low_level_task.action.value = error_msg
            logger.error(error_msg)
            return failed_low_level_task

        return low_level_task


    @stat_time
    def _refine_step(self, task: str, current_state: StateInfo, previous_traj: list[LowLevelTask], previous_high_level_tasks: list[str]) -> str:
        
        message = prompt_refine_during_execution(
            curr_url=current_state.raw_state.url,
            curr_state_context=self._format_page_context(current_state),
            previous_state_action=self._format_previous_observation_and_action(previous_traj, include_all_steps=True, last_k=self.config.history_last_k),
            curr_screenshot=current_state.raw_state.screenshot if self.config.enable_vision else None,
            current_high_level_task=task,
            previous_high_level_tasks='\n'.join(previous_high_level_tasks),
            history_last_k=self.config.history_last_k,
        )

     
        try:
            response = self.gpt_client.request(
                messages=message,
                json_mode=True,
                **self.config.gpt.__dict__,
            )
            response_text = response.message.content
            data = tools_robust_json_loads(response_text)
            
            if not isinstance(data, dict):
                logger.error(f"Expected dict, got {type(data)} from response: {response_text}")
                return task
            
            if 'Need-to-Refine' in data and data['Need-to-Refine'].lower() == 'yes':
                if 'High-Level-Task' in data and len(data['High-Level-Task'].strip()) > 0:
                    return data['High-Level-Task'].strip()
                
            logger.debug(response_text)
            return task
        except Exception as e:
            logger.error(f"Error during refinement step for task={task} with error={e}\nresponse={response}")

            
        return task

    def _stat_accuracy(self, execute_status: dict[str, str]) -> tuple[int, int, int, int]:
        auto_eval_cnt = sum(1 for status in execute_status.values() if status['auto-eval'] != 'NA')
        auto_eval_success = sum(1 for status in execute_status.values() if status['auto-eval'] == 'success')
        complete_cnt = sum(1 for status in execute_status.values() if status['end_reason'] == 'completed')
        total_cnt = len(execute_status)
        return auto_eval_success, auto_eval_cnt, complete_cnt, total_cnt

    def run_episode(self):
        from syn.evaluators import evaluator_router
            
        env, current_state = self._init_env_for_episode(self.config.target_start_url)


        task_exe_cnt = len(self.tasks_done_unique)

        for task_cnt in tqdm(range(len(self.tasks_todo)), desc=f'exeagent-{self.config.output}', initial=task_exe_cnt, total=len(self.tasks_todo)):
            task_dict = self.tasks_todo[task_cnt]            

            if 'sites' in task_dict:
                assert len(task_dict['sites']) == 1, f"Expected exactly one site in task_dict['sites'], got {task_dict}"
                self.config.target_env = task_dict['sites'][0]
                self.config.post_process()

            env_target_port = os.environ[f"{self.config.target_env.upper()}_PORT"]
            task_dict = replace_with_env(task_dict, env_target_port)

            task = task_dict['task']
            original_task_env = f"{task}@{self.config.target_env}"
            
            if self.config.ignore_start_url:
                start_url = self.config.target_start_url
            else:
                start_url = task_dict['start_url']

            if original_task_env in self.tasks_done_unique:
                logger.info(f"Task '{task}' already done, skipping execution.")
                continue

            # reinit env to avoid logout
            observation, info =self._reset_env(env, start_url=start_url, require_login=self.config.env.auto_login)
            observation_metadata = info['observation_metadata']
            current_state = self._get_env_state(env, obs=observation, observation_metadata=observation_metadata)      

            logger.info(f"--- Start Executing {task_exe_cnt}/{len(self.tasks_todo)}---\nTask={task}\nstart_url={start_url}\n")
            logger.info(f"total gpt usage:\n{self.gpt_client.token_usage}")
            logger.info(f"per iteration gpt usage:\n{self.gpt_client.token_usage.per_iteration_str()}")
            logger.info(f"per call gpt usage:\n{self.gpt_client.token_usage.per_iteration_str(self.gpt_client.token_usage.call_num)}")


            # Reset all tabs and navigate to seed URL after each task completion
            logger.info(f"Resetting tabs after task completion...")
            current_state = self._reset_all_tabs_and_open_seed_url(env, start_url)
            
            if not current_state.elements:
                logger.warning(f"No interactive elements found on the start_url={current_state.raw_state.url}. Reset to the hompepage={self.config.target_start_url}")
                current_state = self.goto_url(env, current_state, self.config.target_start_url)
            
            # execute the task
            high_level_task = HighLevelTask(
                task=task,
                start_url=start_url,
                trajectories=[],
            )
            exp_traj = ExplorationTraj(
                curr_state=current_state,
                high_level_tasks=[high_level_task]
            )


            step_idx = 0
            failed_attempt = 0
            while step_idx < self.config.max_steps and failed_attempt < self.config.failed_retry:
                logger.info(f"Executing step {step_idx + 1}/max={self.config.max_steps} for task={task}, currrent_state_url={current_state.raw_state.url}, env_page_url={env.page.url}. failed_attempt={failed_attempt}/{self.config.failed_retry}")

                next_low_level_task: LowLevelTask = self._cot_step(high_level_task.task, current_state, high_level_task.trajectories)

                exp_traj.add_low_level_task(next_low_level_task)

                if next_low_level_task.action.action_type is ActionType.NONE:
                    next_low_level_task.task_status = LowTaskStatus.END
                    logger.info(f"Task {high_level_task.task} completed with status: {next_low_level_task.task_status}")
                    exp_traj.end_exploration()
                    break

                elif next_low_level_task.action.action_type is ActionType.STOP:

                    next_low_level_task.task_status = LowTaskStatus.NOTACHIEVEABLE
                    logger.info(f"Task {high_level_task.task} cannot be achieved with status: {next_low_level_task.task_status}")
                    failed_attempt += 1

                    # refine process
                    if self.config.refine:
                        new_task = self._refine_step(
                            task=high_level_task.task,
                            current_state=current_state,
                            previous_traj=high_level_task.trajectories,
                            previous_high_level_tasks=[t.task for t in exp_traj.high_level_tasks],
                        )
                        if not isinstance(new_task, str) or len(new_task) == 0 or new_task == high_level_task.task:
                            new_task = None
                    else:
                        new_task = None

                    
                    if new_task:
                        old_task = high_level_task.task
                        exp_traj.add_high_level_task(new_task, current_state)
                        high_level_task: HighLevelTask = exp_traj.high_level_tasks[-1]

                        logger.info(f"Refined task from={old_task} to new-task={new_task}")
                        failed_attempt = 0

                    else:
                        logger.info(f"Refinement={self.config.refine} did not change the task, keeping current task={high_level_task.task}")
                        failed_attempt += 1
                    
                    # remove the recent low-level action, and retry

                    if failed_attempt < self.config.failed_retry:

                        if new_task:
                            # remove last try
                            high_level_task.trajectories = high_level_task.trajectories[:-1]
                        
                        else:
                            low_task = high_level_task.trajectories[-1]
                            low_task.task_status = LowTaskStatus.IN_PROGRESS
                            low_task.action.action_type = ActionType.REFLECT
                            low_task.action.value = f"**Failed Analysis**: {low_task.action.value}.\n**Reflection**: Maybe I should consider goto actions to resume to an intermediate step and try DIFFERENT approaches to achieve the task."
                            high_level_task.trajectories[-1] = low_task

                else:
                    next_state = self._execute_single_low_level_task(next_low_level_task, env, curr_state=current_state)
                    next_low_level_task.state_after = next_state
                    current_state = next_state
                    exp_traj.curr_state = current_state
                
                step_idx += 1

            # finalize
            task_exe_cnt += 1
            self.gpt_client.token_usage.iteration_count += 1


            task_status = {
                'steps': len(high_level_task.trajectories),
                'max_steps': self.config.max_steps,
                'refine_cnt': len(exp_traj.high_level_tasks) - 1,
                'high_level_tasks': [t.task for t in exp_traj.high_level_tasks],
                'retry_failed': failed_attempt,
                'end_reason': 'unknown',
                'auto-eval': 'NA',
            }

            if step_idx >= self.config.max_steps:
                s = f"exceeded_max_steps"
            elif high_level_task.trajectories[-1].task_status == LowTaskStatus.END:
                s = f"completed"
            elif high_level_task.trajectories[-1].task_status == LowTaskStatus.NOTACHIEVEABLE:
                s = f"not_achievable"
            else:
                s = "unknown"

            task_status['end_reason'] = s

            # potential evaluation
            if 'eval' in task_dict and isinstance(task_dict['eval'], dict):
                eval_config = task_dict['eval']
                last_low_task = high_level_task.trajectories[-1]
                

                if last_low_task.action.action_type in {ActionType.NONE, ActionType.STOP}:
                    last_action_summary = last_low_task.action.value
                else:
                    last_action_summary = None

                if last_action_summary is None or last_action_summary.strip() == "":
                    if last_low_task.state_after and isinstance(last_low_task.state_after.summary, str):
                        last_action_summary = last_low_task.state_after.summary
                    elif last_low_task.task:
                        last_action_summary = last_low_task.task
                    else:
                        last_action_summary = "No summary available"

                evaluator = evaluator_router(
                    eval_config=eval_config,
                    task=high_level_task.task,
                    last_action_summary=last_action_summary,
                    page=env.page,
                    gpt_client=self.eval_gpt_client,
                )
                score = evaluator()

                if score == 0:
                    task_status['auto-eval'] = 'failed'
                elif score == 1:
                    task_status['auto-eval'] = 'success'
                else:
                    raise ValueError(f"Unexpected score={score} from evaluator: {evaluator}")
                
            
            
            # record
            self.tasks_done_unique[original_task_env] = task_status
            self.tasks_done_buffer.append(high_level_task)

            logger.info(f"Task {original_task_env} executed done with {len(high_level_task.trajectories)} steps. Total tasks done: {len(self.tasks_done_unique)}")

            # accuracy
            auto_eval_success, auto_eval_cnt, complete_cnt, total_cnt = self._stat_accuracy(self.tasks_done_unique)
            logger.info(f"Auto-eval accuracy: {auto_eval_success}/{auto_eval_cnt}={auto_eval_success / auto_eval_cnt if auto_eval_cnt > 0 else 0:.4f}\nComplete rate: {complete_cnt}/{total_cnt}={complete_cnt / total_cnt if total_cnt > 0 else 0:.4f}")
            
            if len(self.tasks_done_buffer) > 0:
                self.save()

            

        env.close()
        logger.info(f"Episode finished. Got done {len(self.tasks_done_unique)} unique tasks.")

        # accuracy
        auto_eval_success, auto_eval_cnt, complete_cnt, total_cnt = self._stat_accuracy(self.tasks_done_unique)
        logger.info(f"Auto-eval accuracy: {auto_eval_success}/{auto_eval_cnt}={auto_eval_success / auto_eval_cnt if auto_eval_cnt > 0 else 0:.4f}\nComplete rate: {complete_cnt}/{total_cnt}={complete_cnt / total_cnt if total_cnt > 0 else 0:.4f}")

        logger.info(f"Total GPT usage:\n{self.gpt_client.token_usage}")
        logger.info(f"Per iteration GPT usage:\n{self.gpt_client.token_usage.per_iteration_str()}")
        logger.info(f"Per call GPT usage:\n{self.gpt_client.token_usage.per_iteration_str(self.gpt_client.token_usage.call_num)}")


if __name__ == "__main__":
    args: ExeAgentConfig = parse_args(ExeAgentConfig)
    start_time = tools_get_time()
    logger.info(f"Starting ExeAgent with config\n{args}\nStart time: {start_time}")
    exeagent = ExeAgent(args)
    exeagent.run_episode()
    logger.info(f"ExeAgent done! started at {start_time} Elapsed time: {tools_elapsed_time_print(start_time)}\n{args}")



