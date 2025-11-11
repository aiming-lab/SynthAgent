from syn.utils import stat_time
import json
import os
import re
import time
import numpy as np
from tqdm import tqdm
from simpleArgParser import to_json

from syn.gpt import GPTClient
from syn.data import (
    Element, LowLevelTask, HighLevelTask, LowTaskStatus, Action, StateInfo, 
    ActionType, RawState, ExplorationTraj,
    ActionExecuteStatus
)
from syn.prompts import (
    prompt_task_categorization_for_actions,
)
from syn.args import ExploreConfig
from syn.consts import (
    const_uninteractive_category,
    const_undefined_category,
)
from syn.tools import (
    tools_jsonl_save,
    tools_jsonl_load,
    tools_serialize_dataclass,
    tools_get_time,
    tools_is_local_url,
    tools_deserialize_dataclass,
)
from collections import defaultdict
from loguru import logger

class Explorer:    
    def __init__(
        self,
        config: ExploreConfig,
    ):
        self.config = config

        self.gpt_client =  GPTClient(provider=self.config.gpt.provider)

        self.exploration_traj_save_db: list[ExplorationTraj] = []
        self.db_status = {
            'total_count': 0,
            'status_counts': defaultdict(int),
            'avg_depth_per_traj': defaultdict(int),
            'avg_low_actions_per_task': defaultdict(int),
        }
        self.visisted_node = set()
        self.iteration_count = 0
        self._env: None | "ScriptBrowserEnv" = None

        self.base_unclickable_elements: dict[str, set[tuple[str, tuple[int, int, int, int]]]] = {}  # url: [name, union_bound]

    @stat_time
    def save(self):
        db = tools_serialize_dataclass(self.exploration_traj_save_db)
        tools_jsonl_save(db, f"{self.config.output}/db.jsonl", append=True)
        
        # simplified version
        db = [item.to_dict(simplified=True) for item in self.exploration_traj_save_db]
        tools_jsonl_save(db, f"{self.config.output}/db.simplified.jsonl", append=True)

        json.dump(tools_serialize_dataclass(self.base_unclickable_elements), open(f"{self.config.output}/base_unclickable_elements.jsonl", 'w'))

        
        # save config
        with open(f"{self.config.output}/config.json", 'w') as f:
            f.write(to_json(self.config))
        
        # save gpt client token usage
        self.gpt_client.token_usage.to_json(f"{self.config.output}/gpt_client_token_usage.json")

        # save db_status
        with open(f"{self.config.output}/db_status.json", 'w') as f:
            json.dump(self.db_status, f, indent=2)


        stats = self.stat_db(self.exploration_traj_save_db)
        self.db_status['total_count'] += len(self.exploration_traj_save_db)
        for metric, v in stats.items():
            if not metric.startswith('avg_'): continue
            for s, new_avg in v.items():
                old_cnt = self.db_status['status_counts'][s]
                new_cnt = stats['status_counts'][s]
                total_cnt = old_cnt + new_cnt
                self.db_status['status_counts'][s] = total_cnt

                if total_cnt > 0:
                    old_avg = self.db_status[metric][s]
                    self.db_status[metric][s] = (old_cnt * old_avg + new_avg * new_cnt) / total_cnt

        self.exploration_traj_save_db = []

    @stat_time
    def load(self):
        if os.path.exists(path := f"{self.config.output}/gpt_client_token_usage.json"):
            self.gpt_client.token_usage.load_from_json(path)
        
        # load db_status
        if os.path.exists(path := f"{self.config.output}/db_status.json"):
            with open(path, 'r') as f:
                self.db_status = json.load(f)
                # Convert defaultdict keys back to defaultdict
                self.db_status['status_counts'] = defaultdict(int, self.db_status.get('status_counts', {}))
                self.db_status['avg_depth_per_traj'] = defaultdict(int, self.db_status.get('avg_depth_per_traj', {}))
                self.db_status['avg_low_actions_per_task'] = defaultdict(int, self.db_status.get('avg_low_actions_per_task', {}))
        pass

        if os.path.exists(path := f"{self.config.output}/base_unclickable_elements.jsonl"):
            self.base_unclickable_elements = tools_deserialize_dataclass(json.load(open(path, 'r')), dict[str, set[tuple[str, tuple[int, int, int, int]]]])

    @staticmethod    
    def stat_db(list_of_trajectories: list[ExplorationTraj]) -> dict:
        """
        Return statistics of the exploration trajectories.
        """
        stats = {
            'status_counts': defaultdict(int),
            'avg_low_actions_per_task': defaultdict(int),
            'avg_depth_per_traj': defaultdict(int),
        }
        
        if not list_of_trajectories:
            return stats
        
        for traj in list_of_trajectories:
            status = traj.status.value
            stats['status_counts'][status] += 1
            stats['avg_depth_per_traj'][status] += len(traj.high_level_tasks)
            stats['avg_low_actions_per_task'][status] += len(traj.high_level_tasks[-1].trajectories)


        for status, count in stats['status_counts'].items():
            if count > 0:
                stats['avg_depth_per_traj'][status] /= count
                stats['avg_low_actions_per_task'][status] /= count
            
        return stats

    @stat_time
    def goto_url(self, env: "ScriptBrowserEnv", curr_state: StateInfo, url: str) -> StateInfo:
        from browser_env import create_goto_url_action

        obs, _, _, _, info = env.step(create_goto_url_action(url))
        time.sleep(3)
        info_meta = info['observation_metadata']
        return self._get_env_state(env, obs, info_meta)


    @stat_time
    def extract_elements(self, raw_state: RawState) -> list[Element]:
        """Extract interactive elements from the current page state."""
        elements = [
            Element(
                accessibility_tree_content=ele['text'],
                union_bound=ele['union_bound'],
                element_id=ele_id
            )
            for ele_id, ele in raw_state.observation_metadata['text']['obs_nodes_info'].items()
        ]
        
        # Filter out explored elements based on similarity
        return elements
    
    @stat_time
    def categorize_tasks_for_action(self, state: StateInfo, excluding_elements: set[tuple[str, Action]]=set()) -> dict[str, list[LowLevelTask]]:
        """[category: [action]]"""
        
        elements: str = self._format_elements_for_llm(state.elements, return_dict=False)
        page_context = self._format_page_context(state)
        message = prompt_task_categorization_for_actions(state.raw_state.url, len(state.elements), elements, page_context, state.raw_state.screenshot)
        url = state.raw_state.url

        try:
            response = self.gpt_client.request(
                messages=message,
                json_mode=True,
                **self.config.gpt.__dict__ | {'max_completion_tokens': 8192},
            )
            response_text = response.message.content
        except Exception as e:
            logger.error(f"Error in requesting GPT: {e}")

        if '//' in response_text:
            response_text = re.sub(r'//.*', '', response_text)

        try:
            data = json.loads(response_text)
            assert isinstance(data, dict) and 'Categorization' in data
        except Exception as e:
            logger.error(f"Invalid JSON response: {e} from response {response_text}")
            return {const_undefined_category: [e.id for e in state.elements]}

        # logger.debug(f"Categorization response data:\n{data}")
        # Convert to {category: [Action]}
        results = {}
        visisted_eids = set()
        eid2element = {str(ele.id): ele for ele in state.elements} | {int(ele.id): ele for ele in state.elements}
        
        for category, list_of_items in data['Categorization'].items():
            results[category] = []
            if category == const_uninteractive_category:
                for eleid in list_of_items:
                    eleid = str(eleid)
                    if eleid in eid2element and eleid not in visisted_eids:
                        visisted_eids.add(eleid)
                        ele = eid2element[eleid]
                        ele.action_type = ActionType.NONE
                        results[category].append(eleid)
                
            elif category == "scroll": # as of July 26, remove the scroll category in the prompt; we defaultly add scroll down
                if isinstance(list_of_items, dict) and 'value' in list_of_items and list_of_items['value']:
                    scroll_value = list_of_items['value'].lower()
                    if scroll_value in {'up', 'down'}:
                        action = Action(None, scroll_value, action_type=ActionType.SCROLL)
                    else:
                        # default to 'down' if value is not recognized
                        action = Action(None, 'down', action_type=ActionType.SCROLL)
                else:
                    action = Action(None, 'down', action_type=ActionType.SCROLL)
                task_str = list_of_items.get('low-level_instruction', f"Scroll {action.value} on the page to explore more content.")
                task = LowLevelTask(
                    task=task_str,
                    curr_state=state,
                    action=action,
                    task_status=LowTaskStatus.IN_PROGRESS,
                )

                if (url, action) not in excluding_elements:
                    results[category].append(task)

            else:
                for item in list_of_items:
                    if isinstance(item, dict) and all(x in item for x in ['action', 'element_id', 'value']):
                        eleid = str(item['element_id'])
                        action_type = item['action'].upper()
                        action_value = item['value']
                        if eleid in eid2element and eleid not in visisted_eids and action_type in ActionType.__members__:
                            visisted_eids.add(eleid)
                            ele = eid2element[eleid]
                            ele.action_type = ActionType(action_type.lower())
                            try:
                                action = Action(ele, action_value, action_type=ele.action_type)
                            except Exception as e:
                                logger.error(f"Error creating Action for element={ele} with action type={action_type} and value='{action_value}'\nerror={e}")
                                continue
                            task_str = item.get('low-level_instruction', str(action))
                            task = LowLevelTask(
                                task=task_str,
                                curr_state=state,
                                action=action,
                                task_status=LowTaskStatus.IN_PROGRESS,
                            )
                            if (url, action) not in excluding_elements:
                                results[category].append(task)

        # add scroll down
        if self._judge_current_page_can_scroll_down(self._env):
            action = Action(None, 'down', action_type=ActionType.SCROLL)
            task_str = f"Scroll down on the page to explore more content."
            task = LowLevelTask(
                task=task_str,
                curr_state=state,
                action=action,
                task_status=LowTaskStatus.IN_PROGRESS,
            )
            if (url, action) not in excluding_elements:
                results['scroll'] = [task]

        # process undefined category

        recognized_num_by_llm = len(visisted_eids)
        if const_undefined_category not in results: results[const_undefined_category] = []
        if const_uninteractive_category not in results: results[const_uninteractive_category] = []
        for eleid, ele in eid2element.items():
            eleid = str(eleid)
            if eleid not in visisted_eids:
                visisted_eids.add(eleid)
                results[const_undefined_category].append(eleid)


        static_elements = len([ele for ele in state.elements if ele.name.startswith('StaticText')])

        logger.info(f"Categorized {len(state.elements)} (containing {static_elements} static elements) elements into {len(results)} categories: {list(results.keys())} with {recognized_num_by_llm}/{len(state.elements) - static_elements} elements recognized by LLM and finally have {len(results[const_undefined_category])} undefined elements, {len(results[const_uninteractive_category])} non-interactive elements")

        return results


    @stat_time
    def _init_env_for_episode(self, start_url: str) -> tuple["ScriptBrowserEnv", StateInfo]:
        from browser_env import ScriptBrowserEnv
        env = ScriptBrowserEnv(
                headless=self.config.browser.headless,
                slow_mo=self.config.browser.slow_mo,
                observation_type=self.config.browser.observation_type,
                current_viewport_only=self.config.browser.current_viewport_only,
                viewport_size=self.config.browser.viewport_size,
                sleep_after_execution=self.config.sleep_after_action,
            )
        self._env = env

        observation, info =self._reset_env(env, start_url=start_url, require_login=self.config.env.auto_login)

        observation_metadata = info['observation_metadata']
        current_state = self._get_env_state(env, obs=observation, observation_metadata=observation_metadata)
        if current_state.raw_state.url == self.config.target_start_url:
            current_state.summary = self.config.target_env_description
        
        return env, current_state

    def _reset_all_tabs_and_open_seed_url(self, env, seed_url: str) -> StateInfo:
        """Close all tabs and open a new tab with the seed URL"""
        from browser_env import create_page_close_action, create_new_tab_action, create_goto_url_action, create_page_focus_action
        try:
            # Get the number of open tabs
            num_tabs = len(env.context.pages)
            logger.info(f"Resetting tabs: currently {num_tabs} tabs open")
            
            # Close all tabs except the first one
            for i in range(num_tabs - 1, 0, -1):  # Close from last to second tab
                try:
                    env.step(create_page_focus_action(i))  # Focus on tab i
                    env.step(create_page_close_action())   # Close the focused tab
                    logger.debug(f"Closed tab {i}")
                except Exception as e:
                    logger.warning(f"Failed to close tab {i}: {e}")
            
            # Navigate the remaining tab (tab 0) to the seed URL
            obs, _, _, _, info = env.step(create_goto_url_action(seed_url))
            logger.info(f"Reset all tabs and navigated to seed URL: {seed_url}")
            
            # Get the new state
            info_meta = info['observation_metadata']
            return self._get_env_state(env, obs, info_meta)
            
        except Exception as e:
            logger.error(f"Error resetting tabs: {e}")
            # Fallback: just navigate to seed URL
            obs, _, _, _, info = env.step(create_goto_url_action(seed_url))
            info_meta = info['observation_metadata']
            return self._get_env_state(env, obs, info_meta)    

    # Helper methods
    def _reset_env(self, env: "ScriptBrowserEnv", start_url: str, require_login: bool=False):
        storage_state = None
        if self.config.target_env in {'reddit', 'shopping', 'gitlab', 'shopping_admin'} and require_login:
            cookie_file_folder = f"{self.config.output}/cookie"
            os.makedirs(cookie_file_folder, exist_ok=True)
            cookie_file_name = f"{cookie_file_folder}/{self.config.target_env}_state.json"
            assert os.path.exists('../webarena-official/browser_env/auto_login.py'), "please check the files for auto login"
            os.system(
                f"cd ../webarena-official && python browser_env/auto_login.py --auth_folder {os.path.abspath(cookie_file_folder)} --site_list {self.config.target_env}"
            )
            assert os.path.exists(cookie_file_name), f"Cookie file {cookie_file_name} does not exist. Please check auto-login."
            storage_state = cookie_file_name

        with open(f"{self.config.output}/init_env.json", 'w') as f:
            state = {'start_url': start_url, 'storage_state': storage_state}
            json.dump(state, f)
            logger.info(f"Resetting environment with state: {state}")

        observation, info = env.reset(options={'config_file': f"{self.config.output}/init_env.json"})

        env.context.set_default_timeout(120000)
        env.context.set_default_navigation_timeout(120000)

        return observation, info

    @staticmethod
    def _format_elements_for_llm(elements: list[Element], return_dict: bool = False, excluding_elements: set[tuple[str, tuple]]=set()) -> str | list:
        """Format elements for LLM prompt."""
        # statictext elements are not interactive, so we skip them

        elements = [
            e
            for e in elements
            if (not e.name.startswith('StaticText')) and
            (not (e.name, e.union_bound) in excluding_elements)
        ]

        actual_excluding = [e for e in elements if (e.name, e.union_bound) in excluding_elements]


        temp = [{'element_id': element.id, 'text': element.name} for element in elements]

        if return_dict: return temp 

        formatted = [str(item) for item in temp]

        if len(excluding_elements) > 0:
            
            logger.debug(f"pass_in_excluding_elements({len(excluding_elements)})={[n for (n, _) in excluding_elements]} \nactual excluding elements ({len(actual_excluding)}): {[e.name for e in actual_excluding]}")

            if len(actual_excluding) == 0:
                logger.warning(f"No actual excluding elements found in this page\n{[e.name for e in elements]}")

            if len(actual_excluding) > 0:
                formatted.append("You MUST NOT choose from the following elements:")
                formatted.extend([str({'element_id': e.id, 'text': e.name}) for e in actual_excluding])
            else:
                formatted.append("You MUST NOT choose from the following elements with text:")
                formatted.extend([str({'text': n}) for (n, _) in excluding_elements])

        return "\n".join(formatted)
    
    @staticmethod
    def _format_page_context(state: StateInfo) -> str:
        return state.raw_state.accessibility_tree

    @staticmethod
    def _format_previous_observation_and_action(trajectories: list[LowLevelTask], return_dict: bool = False, include_all_steps: bool = False, last_k: int | None = None) -> list[dict] | str:
        """
        previous_state_action = [
            {
                "state_observation_summary": "xxxx",
                "reasoning-for-action": xxxxx,
                "low-level_instruction: "xxxx",
                "action_str": "xxxx",
            },
        ]
        """
        previous_state_action = []
        if last_k is None:
            last_k = len(trajectories)

        for task in trajectories[-last_k:]:
            # executed
            if task.state_after is not None or include_all_steps:
                previous_state_action.append({
                    "current_state_observation": task.curr_state.summary,
                    "current_state_url": task.curr_state.raw_state.url,
                    # "reasoning-for-action": task.reasoning,
                    "low-level_instruction": task.task,
                    "action_str": str(task.action),
                })

        if return_dict:
            return previous_state_action

        return "\n\n".join([str(item) for item in previous_state_action])

    @stat_time
    def _execute_single_low_level_task(self, task: LowLevelTask, env: "ScriptBrowserEnv", curr_state: StateInfo | None = None) -> StateInfo:
        """Execute a single action in the environment. return new state
        make sure the current state in the newest one and accurate!!!
        """
        from browser_env import (
            create_id_based_action,
            create_none_action,
            create_page_close_action,
            create_page_focus_action,
        )

        env.page.set_default_navigation_timeout(120000)  # temp 120 seconds

        if task.state_after is not None:
            if task.state_after.raw_state.url == env.page.url:
                logger.warning(f"skip executing low-level task {task.action} because the state_after is not None and matches the current env page URL {env.page.url}.")
                return task.state_after  # already executed
            else:
                logger.warning(f"Low-level task state_after URL {task.state_after.raw_state.url} does not match current env page URL {env.page.url}. Re-executing the task.")

        else:
            if curr_state is None:
                curr_state = self._get_env_state(env, None, None)

            if task.curr_state != curr_state:
                logger.warning(f"Low-level task current state doee not match the pass in current state)")
                task.curr_state = curr_state
            
            name2ele = {e.name: e for e in curr_state.elements}
            accessible_elements = {e.accessibility_tree_content for e in curr_state.elements}
            if task.action.target_element and task.action.target_element.accessibility_tree_content not in accessible_elements:
                # the current state does not match the target element (might have been refreshed or goto)

                if task.action.target_element.name in name2ele:
                    task.action.target_element = name2ele[task.action.target_element.name]
                    task.action.target_element.action_type = task.action.action_type
                else:
                    task.task_status = LowTaskStatus.NOTACHIEVEABLE
                    task.action.status = ActionExecuteStatus.FAILURE
                    task.state_after = curr_state
                    return curr_state

        initial_tabs_count = len(env.context.pages)
        for page in env.context.pages:
            page.set_default_navigation_timeout(120000)  # temp 120 seconds
        
        if task.action.action_type == ActionType.NONE:
            # do nothing
            if task.curr_state.raw_state.url != env.page.url:
                logger.warning(f"Low-level task with NONE action type, but current state URL {task.curr_state.raw_state.url} does not match env page URL {env.page.url}.")

            obs = env._get_obs()
            observation_metadata = env._get_obs_metadata()
            info = {'observation_metadata': observation_metadata}
        
        else:
            obs, _, _, _, info = env.step(create_id_based_action(task.action.get_action_str()))
            time.sleep(self.config.sleep_after_action)

        task.action.status = ActionExecuteStatus.SUCCESS
        logger.info(f"Executed action_str={task.action.get_action_str()}, detail={str(task.action)}")

        current_tabs_count = len(env.context.pages)
        current_tabs_urls = [page.url for page in env.context.pages]
        tabs_to_close = []
        if current_tabs_count > initial_tabs_count:
            logger.warning(f"New tab(s) detected after action! Tabs count: {initial_tabs_count} -> {current_tabs_count}")
            logger.warning("All current tab URLs:")
            for i, url in enumerate(current_tabs_urls):
                logger.warning(f"  Tab {i}: {url}")
            
            
            for i, page in enumerate(env.context.pages):
                if not tools_is_local_url(page.url):
                    tabs_to_close.append(i)
                    logger.warning(f"Marking tab {i} for closure (non-localhost): {page.url}")
            
            for tab_index in reversed(tabs_to_close):
                try:
                    env.step(create_page_focus_action(tab_index))
                    env.step(create_page_close_action())
                    logger.info(f"Closed non-localhost tab {tab_index}")
                except Exception as e:
                    logger.error(f"Failed to close tab {tab_index}: {e}")
            
            if not tools_is_local_url(env.page.url):
                for i, page in enumerate(env.context.pages):
                    if tools_is_local_url(page.url):
                        try:
                            env.step(create_page_focus_action(i))
                            logger.info(f"Switched focus to localhost tab {i}: {page.url}")
                            break
                        except Exception as e:
                            logger.error(f"Failed to switch to tab {i}: {e}")
            
            logger.info(f"After cleanup - Active tabs: {[page.url for page in env.context.pages]}")

        # Get new state after execution (and potential tab cleanup)
        new_state = self._get_env_state(env, obs=obs, observation_metadata=info['observation_metadata'])
        task.state_after = new_state

        # adding element to excluding if go to the external url
        if task.action.target_element and (len(tabs_to_close) > 0 or not self._states_different(curr_state, new_state)):
            hash_item = self._add_url_element_to_pool(
                url=curr_state.raw_state.url,
                element=task.action.target_element,
                pool=self.base_unclickable_elements,
            )
            logger.info(f"Added element {hash_item} to base_unclickable_elements due to tab closure.")

        return new_state
     
    @stat_time
    def _get_env_state(self, env: "ScriptBrowserEnv", obs: dict | None, observation_metadata: dict | None) -> StateInfo:
        if obs and observation_metadata:
            pass
        else:
            obs = env._get_obs()
            observation_metadata = env._get_obs_metadata()
        

        raw_state = RawState(url=env.page.url, accessibility_tree=obs['text'], observation_metadata=observation_metadata, screenshot=obs['image'], timestamp=time.time())
        elements = self.extract_elements(raw_state)

        return StateInfo(
            raw_state=raw_state,
            elements=elements,
        )

    def _states_different(self, state1: StateInfo, state2: StateInfo) -> bool:
        """Check if two states are different (page changed)."""
        return (state1.raw_state.url != state2.raw_state.url) or \
                (state1.raw_state.accessibility_tree != state2.raw_state.accessibility_tree) or \
                (not np.array_equal(state1.raw_state.screenshot, state2.raw_state.screenshot))
    
    def _judge_current_page_can_scroll_down(self, env: "ScriptBrowserEnv") -> bool:
        return env.page.evaluate(
            """() => {
                const doc = document.documentElement;
                const scrolled = doc.scrollTop + window.innerHeight;
                const total   = doc.scrollHeight;
                return scrolled < total;
            }"""
        )

    def _cnt_unique_tasks_by_load_db(self, status: str = 'END') -> int:
        path = f"{self.config.output}/db.simplified.jsonl"
        unique = set()

        if os.path.exists(path):
            for item in tools_jsonl_load(path):
                if item['status'] == status:
                    unique.add(item["high_level_tasks"][-1])
        
        return len(unique)
    
    def _add_url_element_to_pool(self, url: str, element: Element, pool: dict[str, set[tuple[str, tuple[int, int, int, int]]]]) -> tuple[str, tuple[str, tuple[int, int, int, int]]]:
        if url not in pool:
            pool[url] = set()
        pool[url].add((element.name, element.union_bound))

        return (url, (element.name, element.union_bound))

if __name__ == '__main__':
    pass