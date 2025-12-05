from simpleArgParser import parse_args
from dataclasses import dataclass, field
from enum import Enum
from syn.consts import (
    const_web_description_reddit,
    const_web_description_shopping,
    const_web_description_gitlab,
    const_web_description_shopping_admin,
    const_web_description_openstreetmap,
)

class APIProvider(Enum):
    openai = "openai"

@dataclass
class BrowserConfig:
    headless: bool = True
    slow_mo: int = 0
    observation_type: str = "accessibility_tree"
    viewport_size = {"width": 1280, "height": 720}
    current_viewport_only: bool = True
    screenshot_full_page: bool = False
    fill_instead_of_type: bool = True

@dataclass
class EnvConfig:
    env_start_port: int
    env_domain: str = "127.0.0.1"
    auto_login: bool = True
    
    def pre_process(self):
        import os
        base_port = self.env_start_port
        shopping_port = base_port
        shopping_admin_port = base_port + 1
        reddit_port = base_port + 2
        gitlab_port = base_port + 3
        wikipedia_port = base_port + 4
        map_port = base_port + 5
        homepage_port = base_port + 6
        reset_port = base_port + 7

        os.environ['ENV_DOMAIN'] = self.env_domain

        os.environ['SHOPPING_PORT'] = str(shopping_port)
        os.environ['SHOPPING_ADMIN_PORT'] = str(shopping_admin_port)
        os.environ['REDDIT_PORT'] = str(reddit_port)
        os.environ['GITLAB_PORT'] = str(gitlab_port)
        os.environ['MAP_PORT'] = str(map_port)
        os.environ['HOMEPAGE_PORT'] = str(homepage_port)
        os.environ['WIKIPEDIA_PORT'] = str(wikipedia_port)
        

        os.environ['SHOPPING'] = f"http://{self.env_domain}:{shopping_port}"
        os.environ['SHOPPING_ADMIN'] = f"http://{self.env_domain}:{shopping_admin_port}/admin"
        os.environ['REDDIT'] = f"http://{self.env_domain}:{reddit_port}"
        os.environ['GITLAB'] = f"http://{self.env_domain}:{gitlab_port}/explore"
        os.environ['MAP'] = f"http://{self.env_domain}:{map_port}"
        os.environ['HOMEPAGE'] = f"http://{self.env_domain}:{homepage_port}"
        os.environ['WIKIPEDIA'] = f"http://{self.env_domain}:{wikipedia_port}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"


@dataclass
class GPTConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.7
    max_completion_tokens: int = 4096
    provider: APIProvider = APIProvider.openai
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str = "dummy"

    def pre_process(self):
        import os        
        if self.provider == APIProvider.openai:
            os.environ['OPENAI_API_BASE'] = self.openai_api_base
            os.environ['OPENAI_API_KEY'] = self.openai_api_key

        self.model = GPTConfig.model_map(self.model, get_simplified=False)

    def post_process(self):
        from loguru import logger
        if self.model.startswith('o'):
            self.temperature = 1.0
            logger.warning(f"Model {self.model} is o series, thus setting temperature to 1.0")

    @staticmethod
    def model_map(model: str, get_simplified=False) -> str:
        mappings = {
            'Qwen/Qwen2.5-VL-7B-Instruct': 'qwen7b',
            'ByteDance-Seed/UI-TARS-1.5-7B': 'tars7b',
        }

        if not get_simplified:
            mappings = {v: k for k, v in mappings.items()}
        
        reverse = {v: k for k, v in mappings.items()}

        if model in reverse:
            return model

        if (not model.startswith('gpt')) and (not model.startswith('o')):
            if model in mappings:
                return mappings[model]
            else:
                raise ValueError(f"Unknown model={model} not in allowed {mappings}")    
        else:
            return model

@dataclass
class DebugConfig:
    enable_stat_time_logging: bool = False
    enable_stat_time_block_logging: bool = False
    debug: bool = False
    

@dataclass
class ExploreConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    sleep_after_action: float = 0.5
    target_env: str = "reddit" # choose from "shopping", "reddit", "shopping_admin", "gitlab", "map"
    target_start_url: str | None = None
    target_env_description: str | None = None

    # exploration config
    max_iteration: int = 128
    max_ele_for_sampling: int = 10 # max number of elements to sample for interaction
    max_ele_per_category: int = 3  # max number of elements to execute per category

    # output directory for the exploration results
    output: str | None = None
    note: str | None = None


    def _set_output(self, subfolder: str='explore'):
        import os
        from syn.consts import const_screenshot_save_path
        # set output directory
        from syn.tools import tools_get_time
        if self.output is None:
            self.output = f"outputs/{subfolder}/{self.target_env}/{tools_get_time()}"

            if isinstance(self.note, str):
                self.output += f"_{self.note}"
        
        if self.debug.debug and not self.output.startswith('outputs/debug/'):
            self.output = self.output.replace('outputs/', 'outputs/debug/')

        # set log file
        from loguru import logger
        log_file = f"{self.output}/run.log"
        logger.add(log_file, format='<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>', level='DEBUG', colorize=False, rotation=None)

        # set screenshot save path
        os.environ[const_screenshot_save_path] = f"{self.output}/screenshots"
        os.makedirs(os.environ[const_screenshot_save_path], exist_ok=True)

    def _set_env(self):
        # set debug env
        import os
        from syn.consts import const_enable_logging_stat_time, const_enable_logging_stat_time_block, const_playwright_fill_instead_of_type, const_playwright_screenshot_full_page
        os.environ[const_enable_logging_stat_time] = str(int(self.debug.enable_stat_time_logging))
        os.environ[const_enable_logging_stat_time_block] = str(int(self.debug.enable_stat_time_block_logging))

        # set environment for playwright
        os.environ[const_playwright_fill_instead_of_type] = str(int(self.browser.fill_instead_of_type))
        os.environ[const_playwright_screenshot_full_page] = str(int(self.browser.screenshot_full_page))

    def pre_process(self):
        self._set_env()


    def post_process(self):
        import os
        from loguru import logger
        assert self.target_env in ["shopping", "reddit", "shopping_admin", "gitlab", "map"], "Invalid target environment"
        if self.target_env_description is None:
            self.target_env_description = {
                "shopping": const_web_description_shopping,
                "reddit": const_web_description_reddit,
                "gitlab": const_web_description_gitlab,
                "shopping_admin": const_web_description_shopping_admin,
                "map": const_web_description_openstreetmap,
            }[self.target_env]
        
        self.target_start_url = os.environ[self.target_env.upper()]

        logger.info(f"Target start URL is set to {self.target_start_url} for environment {self.target_env}")



@dataclass
class SynthAgentConfig(ExploreConfig):
    synth_until_tasks: int | None = None

    def pre_process(self):
        from loguru import logger
        if isinstance(self.synth_until_tasks, int):
            logger.info(f"synth_until_tasks={self.synth_until_tasks} is set, will synthesize until this number of tasks is reached, thus the max_iteration will be ignored")
            self.max_iteration = int(1e18)
        else:
            self.synth_until_tasks = int(1e18)


        self._set_env()
        self._set_output(subfolder='synthagent')




@dataclass
class ExeAgentConfig(ExploreConfig):
    tasks_path: str = field(kw_only=True)
    max_steps: int = 30
    enable_vision: bool = True # enable vision for the agent
    # refine the tasks during execution
    refine: bool = False
    # failed task retry
    failed_retry: int = 2
    ignore_start_url: bool = False # whether ignore the start URL in the config data

    # params for agent inference
    history_last_k: int | None = 3

    # eval gpt client
    eval_gpt: GPTConfig = field(default_factory=GPTConfig)




    def pre_process(self):
        import os
        assert os.path.exists(self.tasks_path), f"Tasks path={self.tasks_path} does not exist"
        from syn.tools import tools_get_time

        self._set_env()

        tasks_path_name = os.path.basename(self.tasks_path)
        if tasks_path_name.startswith('webarena'):
            assert self.ignore_start_url is False, "webarena tasks must use the start_url in the tasks file"

        
        if self.output is None:
            name = os.path.basename(self.tasks_path.strip('.jsonl'))
            # if any(t in name for t in ['5_single_sites', 'webarena.750', 'webarena.226', 'synthagent', 'osgenesis']):
            env_folder_name = 'webarena'
            # else:
            #     env_folder_name = self.target_env

            model_name = GPTConfig.model_map(self.gpt.model, get_simplified=True)
            self.output = f"outputs/exeagent/{env_folder_name}/{name}.{model_name}.vision-{self.enable_vision}.refine-{self.refine}.last_k-{self.history_last_k}.{tools_get_time()}"

            if isinstance(self.note, str):
                self.output += f"_{self.note}"

        self._set_output(subfolder='exeagent')

        self.sleep_after_action = max(self.sleep_after_action, 2.0)


if __name__ == "__main__":
    import os
    args: ExeAgentConfig = parse_args(ExeAgentConfig)
    print(args)

    EnvConfig(env_start_port=9999).pre_process()
    print("shopping env=", os.environ['SHOPPING_PORT'])