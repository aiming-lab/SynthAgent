from syn.base_explore import Explorer
from syn.data import StateInfo, Action, RawState, ActionType, HighLevelTask, Element, LowLevelTask, ExplorationTraj, ExplorationTrajStatus, ActionExecuteStatus, LowTaskStatus

from syn.args import SynthAgentConfig
from syn.prompts import (
    prompt_osgenesis_generate_high_level_task,
)
from syn.tools import (
    tools_get_time,
    tools_elapsed_time_print,
    tools_jsonl_save,
    tools_jsonl_load,
    tools_serialize_dataclass,
    tools_deserialize_dataclass,
    tools_is_local_url,
)
from syn.utils import stat_time, stat_time_block
from syn.consts import (
     const_undefined_category,
     const_uninteractive_category
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

class SynthAgent(Explorer):
    def __init__(self, config: SynthAgentConfig):
        super().__init__(config)
        self.config: SynthAgentConfig = config
        self.url_visit_count: dict[str, int] = defaultdict(int)  # Track visit counts for each URL

        self.unclick_elem_pool: set[tuple[str, Action]] = set()
        self.click_elem_pool: set[tuple[str, Action]] = set()
        self.explored_elem_pool: set[tuple[str, Action]] = set()

        # save
        # before_state, action, new_state
        self.exp_traj_buffer: list[tuple[StateInfo, Action, ExplorationTraj]] = []
        
        # Element pools file paths
        self.element_pools_dir = os.path.join(config.output, "element_pools")
        os.makedirs(self.element_pools_dir, exist_ok=True)
        
        # Load existing element pools
        self.load()
        self.iteration_count = self.gpt_client.token_usage.iteration_count


    def save(self):
        super().save()
        self._save_element_pools()
        self._save_url_visit_counts()

    def load(self):
        super().load()
        self._load_element_pools()
        self._load_url_visit_counts()

    def _hash_item_in_set(self, url: str, action: Action, pool: set[tuple[str, Action]]) -> bool:
        return (url, action) in pool

    def _load_element_pools(self):
        pools = ['unclick', 'click', 'explored']
        for pool_name in pools:
            pool_path = os.path.join(self.element_pools_dir, f"{pool_name}_pool.json")
            if os.path.exists(pool_path):
                pool_data = tools_jsonl_load(pool_path)
                pool_data = tools_deserialize_dataclass(pool_data, set[tuple[str, Action]])
                setattr(self, f"{pool_name}_elem_pool", pool_data)
                logger.info(f"Loaded {len(pool_data)} elements from {pool_path} pool.")

    def _load_url_visit_counts(self):
        visit_count_path = os.path.join(self.element_pools_dir, "url_visit_counts.json")
        if os.path.exists(visit_count_path):
            with open(visit_count_path, 'r') as f:
                self.url_visit_count = json.load(f)
                self.url_visit_count = defaultdict(int, self.url_visit_count) 
            logger.info(f"Loaded visit counts for {len(self.url_visit_count)} URLs.")

    def _save_url_visit_counts(self):
        visit_count_path = os.path.join(self.element_pools_dir, "url_visit_counts.json")
        with open(visit_count_path, 'w') as f:
            json.dump(self.url_visit_count, f, indent=4)

    def _save_element_pools(self):
        """Save element pools to disk"""
        pools = [
            ('unclick', list(self.unclick_elem_pool)),
            ('click', list(self.click_elem_pool)),
            ('explored', list(self.explored_elem_pool))
        ]
        for pool_name, pool_set in pools:
            pool_path = os.path.join(self.element_pools_dir, f"{pool_name}_pool.json")
            tools_jsonl_save(tools_serialize_dataclass(pool_set), pool_path)
        
        # Also save URL visit counts
        self._save_url_visit_counts()

    def _are_screenshots_identical(self, screenshot1: np.ndarray, screenshot2: np.ndarray):
        """Check if two screenshots are identical"""
        return np.array_equal(screenshot1, screenshot2)

    @stat_time
    def _weighted_select_element_by_category(self, url: str, category_to_tasks: dict[str, list[LowLevelTask]], new_elements: set[Element]) -> list[LowLevelTask]:

        capped: dict[str, list[LowLevelTask]] = {}
        for cat, tasks in category_to_tasks.items():
            if cat in {const_uninteractive_category, const_undefined_category} or len(tasks) == 0:
                continue

            if cat in {'scroll'}:
                 capped[cat] = tasks
                 continue

            # filter out unclickable elements 
            weighted_tasks = [task for task in tasks if not self._hash_item_in_set(url, task.action, self.unclick_elem_pool)]
            
            if len(weighted_tasks) <= self.config.max_ele_per_category:
                capped[cat] = weighted_tasks
            else:
                capped[cat] = self._weighted_sample_tasks(url, weighted_tasks, self.config.max_ele_per_category, new_elements)

        categories = list(capped.keys())
        T = self.config.max_ele_for_sampling

        selected: list[LowLevelTask] = []
        selected_per_cat: dict[str, int] = defaultdict(int)

        # If we have at least T categories, pick one task from T distinct categories
        if len(categories) >= T:
            chosen_cats = random.sample(categories, T)
            for cat in chosen_cats:
                # Use weighted sampling with num_samples=1
                tasks = self._weighted_sample_tasks(url, capped[cat], 1, new_elements)
                if tasks:
                    selected.extend(tasks)
            return selected

        # Otherwise, first pick one per category to maximize coverage
        for cat in categories:
            tasks = self._weighted_sample_tasks(url, capped[cat], 1, new_elements)
            if tasks:
                selected.extend(tasks)
                selected_per_cat[cat] = 1

        # Fill the remaining slots from the leftover capped candidates
        remaining_slots = T - len(selected)
        pool: list[LowLevelTask] = []
        for cat, tasks in capped.items():
            quota = self.config.max_ele_per_category - selected_per_cat[cat]
            if quota <= 0:
                continue
            # collect tasks not yet selected
            for task in tasks:
                if task not in selected:
                    pool.append(task)

        if len(pool) <= remaining_slots:
            selected += pool
        else:
            # Use weighted sampling for remaining slots
            extra = self._weighted_sample_tasks(url, pool, remaining_slots, new_elements)
            selected += extra

        random.shuffle(selected)
        return selected

    def _weighted_sample_tasks(self, url: str, tasks: list[LowLevelTask], num_samples: int, new_elements: set[Element]) -> list[LowLevelTask]:
        """
        Sample tasks using weighted selection WITHOUT replacement.
        Handles both single selection (num_samples=1) and multiple selection cases.
        Uses 4-level weighting: unexplored+new=4, explored+new=3, unexplored+old=3, explored+old=1
        """
        if len(tasks) <= num_samples:
            return tasks.copy()
        
        task_weights = []

        for task in tasks:
            element = task.action.target_element
            is_new = element in new_elements
            is_explored = self._hash_item_in_set(url, task.action, self.explored_elem_pool)
            
            # Use the same 4-level weighting system for consistency
            if (not is_explored) and is_new:
                weight = 4  # Unexplored and newly appeared, highest weight
            elif is_explored and is_new:
                weight = 3  # Explored but newly appeared, second highest weight
            elif (not is_explored) and (not is_new):
                weight = 3  # Unexplored but not newly appeared, second highest weight
            else:
                weight = 1  # Explored and not newly appeared, lowest weight
            
            task_weights.append(weight)

        task_weights = np.array(task_weights, dtype=np.float64)
        task_weights /= task_weights.sum()  # Normalize weights
        selected = np.random.choice(tasks, size=num_samples, replace=False, p=task_weights)

        return list(selected)
                
    @stat_time
    def batch_generate_high_level_task(self, before_state: list[StateInfo], action: list[Action], new_state: list[StateInfo]) -> list[list[str|None, str|None]]:
        """Prompts the LLM to generate a high-level task and low-level instruction based on a state transition."""
        
        batch_message = [
            prompt_osgenesis_generate_high_level_task(
                website_intro=self.config.target_env_description,
                curr_state_screenshot=before_state.raw_state.screenshot,
                new_state_screenshot=new_state.raw_state.screenshot,
                current_action_str=str(action),
                bounding_box=action.target_element.union_bound if action.target_element else None,
                website_name=self.config.target_env,
            )
            for before_state, action, new_state in zip(before_state, action, new_state)
        ]

        batch_message: list[dict] = [
            {'messages': msg, **self.config.gpt.__dict__}
            for msg in batch_message
        ]

        tasks = [[None, None] for _ in range(len(batch_message))]

        for idx, response in enumerate(self.gpt_client.batch_requests(batch_message, json_mode=True)):
            try:
                response_text = response.message.content
                data = json.loads(response_text)

                if isinstance(data, dict) and 'High-Level-Instruction' in data:
                    tasks[idx][0] = data['High-Level-Instruction']
                else:
                    logger.warning(f"cannot parse high-level instruction from {data}")

                if isinstance(data, dict) and 'Sub-Instruction' in data:
                    tasks[idx][1] = data['Sub-Instruction']
                else:
                    logger.warning(f"cannot parse sub-instruction from {data}")

                logger.info(f"Generated high-level task and low-level instruction: {tasks[idx]}")

            except Exception as e:
                logger.error(f"Error generating high-level task: {e}")

        return tasks

    def _weighted_select_url(self) -> str:
        """
        Select URL from available URLs with weighted sampling based on inverse visit counts.
        URLs with fewer visits have higher probability of being selected.
        """
        available_urls = list(self.url_visit_count.keys())
        if not available_urls:
            raise ValueError("No URLs available")
        
        # Calculate weights based on inverse visit counts
        weights = []
        for url in available_urls:
            visit_count = self.url_visit_count[url]
            weight = 1.0 / (visit_count + 1) + 0.1
            weights.append(weight)
        
        # Select URL using weighted random choice
        selected_url = random.choices(available_urls, weights=weights, k=1)[0]
        
        # Update visit count
        self.url_visit_count[selected_url] *= 2
        self.url_visit_count[selected_url] = max(self.url_visit_count[selected_url], 2)
        
        logger.info(f"Selected URL: {selected_url} (visit count: {self.url_visit_count[selected_url]})")
        return selected_url

    def run_episode(self, seed_url: str):
        """
        Runs an episode of OS-Genesis exploration with intelligent element selection.
        Incorporates random walk strategies for better exploration.
        """
        # Initialize visit count for seed URL
        self.url_visit_count[seed_url] = 0
            
        env, current_state = self._init_env_for_episode(seed_url)

        for iteration_cnt in tqdm(range(self.iteration_count, self.config.max_iteration), desc=f'synthagent-{self.config.target_env}-{self.config.env.env_start_port}'):
            
            # 1. Use weighted selection to choose URL based on visit counts
            if not self.url_visit_count:
                logger.warning("No URLs available. Ending episode.")
                break
            target_url = self._weighted_select_url()
            current_state = self._reset_all_tabs_and_open_seed_url(env, target_url)

            task_cnt = self._cnt_unique_tasks_by_load_db()

            logger.info(f"--- Start Iteration {iteration_cnt + 1}/{self.config.max_iteration} --- visit={self.url_visit_count[target_url]}, URL={target_url}\nstats=\n{self.db_status}\nunique_task_count={task_cnt}/{self.config.synth_until_tasks}")
            logger.info(f"url pool = {len(self.url_visit_count)}\n{self.url_visit_count}")
            logger.info(f"total gpt usage:\n{self.gpt_client.token_usage}")
            logger.info(f"per iteration gpt usage:\n{self.gpt_client.token_usage.per_iteration_str()}")
            logger.info(f"per call gpt usage:\n{self.gpt_client.token_usage.per_iteration_str(self.gpt_client.token_usage.call_num)}")

            if task_cnt >= self.config.synth_until_tasks:
                logger.info(f"Reached target task count {task_cnt} >= {self.config.synth_until_tasks}. Ending episode.")
                break

            if not current_state.elements:
                logger.warning(f"No interactive elements found on the page={current_state.raw_state.url}. Skipping interaction.")
                self.url_visit_count[target_url] = int(1e18) #  to avoid re-selection
                continue

            prev_state_elements: set[Element] = set()
            # find new elements
            current_state_elements = set(current_state.elements)
            new_elements = current_state_elements - prev_state_elements

            category2tasks = self.categorize_tasks_for_action(current_state, excluding_elements=self.unclick_elem_pool)
            selected_tasks: list[LowLevelTask] = self._weighted_select_element_by_category(current_state.raw_state.url, category2tasks, new_elements)


            for idx, task in enumerate(selected_tasks):
                current_state = self._reset_all_tabs_and_open_seed_url(env, target_url)

                hash_item = (current_state.raw_state.url, task.action)

                logger.info(f"Selected low-level task {idx + 1}/{len(selected_tasks)}, executing {task.action}")

                next_state = self._execute_single_low_level_task(task, env, curr_state=current_state)

                self.explored_elem_pool.add(hash_item)

                # judge if the action caused a page change
                logger.debug(f"next state url = {next_state.raw_state.url}, current state url = {current_state.raw_state.url}")
                if not self._states_different(current_state, next_state):
                    logger.info(f"No page change detected. Adding {hash_item} to unclickable pool.")
                    if task.action.target_element:
                        self.unclick_elem_pool.add(hash_item)

                elif not tools_is_local_url(next_state.raw_state.url):
                    logger.info(f"page change detected, however leading to external url={next_state.raw_state.url}, skipping interaction.")
                    if task.action.target_element:
                        self.unclick_elem_pool.add(hash_item)

                else:
                    self.click_elem_pool.add(hash_item)
                    self.url_visit_count[next_state.raw_state.url] += 1
                    exp_traj = ExplorationTraj(curr_state=current_state)
                    exp_traj.add_high_level_task("empty", next_state)
                    exp_traj.add_low_level_task(task)
                    self.exp_traj_buffer.append((current_state, task.action, exp_traj))
                
            
            # clear buffered trajectories for generating high-level tasks
            if len(self.exp_traj_buffer) >= 8 or (iteration_cnt + 1) >= self.config.max_iteration:
                logger.info(f"Generating high-level tasks for {len(self.exp_traj_buffer)} buffered trajectories.")
                before_states = [item[0] for item in self.exp_traj_buffer]
                actions = [item[1] for item in self.exp_traj_buffer]
                after_states = [item[2].curr_state for item in self.exp_traj_buffer]
                high_low_tasks: list[tuple[str|None, str|None]] = self.batch_generate_high_level_task(before_states, actions, after_states)

                for idx, (high_str, low_str) in enumerate(high_low_tasks):
                    exp_traj: ExplorationTraj = self.exp_traj_buffer[idx][2]
                    if isinstance(low_str, str) and len(low_str) > 0:
                        exp_traj.high_level_tasks[-1].trajectories[-1].task = low_str

                    if isinstance(high_str, str) and len(high_str) > 0:
                        exp_traj.high_level_tasks[-1].task = high_str
                        exp_traj.end_exploration(ExplorationTrajStatus.END)
                    else:
                        exp_traj.end_exploration(ExplorationTrajStatus.EARLY_END_DURING_SYNTHESIS)
                    
                    self.exploration_traj_save_db.append(exp_traj)
                
                self.exp_traj_buffer.clear()
                logger.info(f"Iteration={iteration_cnt+1} done! Saving {len(self.exploration_traj_save_db)} exploration trajectories to the database. Current Iteration Stats={self.stat_db(self.exploration_traj_save_db)}")
                self.save()
            
            self.iteration_count += 1
            self.gpt_client.token_usage.iteration_count += 1

        env.close()
        logger.info(f"Episode finished. Generated\n{self.db_status}\noutput={self.config.output}\nunique_task_count={self._cnt_unique_tasks_by_load_db()}/{self.config.synth_until_tasks}")
        logger.info(f"Element pool stats - Clickable: {len(self.click_elem_pool)}, "
                   f"Unclickable: {len(self.unclick_elem_pool)}, Explored: {len(self.explored_elem_pool)}")
        logger.info(f"URL stats - Total discovered: {len(self.url_visit_count)}")
        logger.info(f"Total GPT usage:\n{self.gpt_client.token_usage}")
        logger.info(f"Per iteration GPT usage:\n{self.gpt_client.token_usage.per_iteration_str()}")
        logger.info(f"Per call GPT usage:\n{self.gpt_client.token_usage.per_iteration_str(self.gpt_client.token_usage.call_num)}")


if __name__ == "__main__":
    args: SynthAgentConfig = parse_args(SynthAgentConfig, pass_in=[])
    start_time = tools_get_time()
    logger.info(f"Starting SynthAgent with config\n{args}\nStart time: {start_time}")
    synthagent = SynthAgent(args)
    synthagent.run_episode(args.target_start_url)
    logger.info(f"synthagent done! started at {start_time} Elapsed time: {tools_elapsed_time_print(start_time)}\n{args}")






























     