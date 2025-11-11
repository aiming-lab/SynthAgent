from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import json
import numpy as np
import copy
import time
import re
import os
from syn.tools import tools_hash, tools_ndarray_image_save, tools_serialize_dataclass
import hashlib
from syn.consts import const_screenshot_save_path, const_is_load_screenshot_image, const_disable_screenshot_path_check
from uuid import uuid4
from loguru import logger




class ActionType(Enum):
    """Types of actions that can be performed on elements."""
    """
    noop Do nothing
    click(elem) Click at an element
    hover(elem) Hover on an element
    type(elem, text) Type to an element
    press(key_comb) Press a key comb
    scroll(dir) Scroll up and down
    tab_focus(index) focus on i-th tab
    new_tab Open a new tab
    tab_close Close current tab
    go_back Visit the last URL
    go_forward Undo go_back
    goto(URL) Go to URL
    """
    CLICK = "click"
    TYPE = "type"
    HOVER = "hover"
    SCROLL = "scroll"
    PRESS = "press"
    NONE = "none" # do nothing
    # update
    GOTO = "goto"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    STOP = "stop"  # stop the current task, often for non-achievable tasks
    # reflection
    REFLECT = "reflect"

class SynthesizeStatus(Enum):
    """Status of high level task in the synthesis loop."""
    KEEP = "KEEP"
    NEW = "NEW"
    REFINED = "REFINED"
    EXPANDED = "EXPANDED"
    DONE = "DONE"

class ActionExecuteStatus(Enum):
    """Status of executing a low level task in the environment."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    NOT_EXECUTED = "NOT_EXECUTED"

class LowTaskStatus(Enum):
    """status of a low level task"""

    BEGIN = "BEGIN"
    IN_PROGRESS = "IN_PROGRESS"
    END = "END"
    NOTACHIEVEABLE = "NOT_ACHIEVABLE"

class ExplorationTrajStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    END = "END"
    DROP = "DROP"
    OVER_DEPTH_END = "OVER_DEPTH_END"
    EARLY_END_NO_ELEMENTS_TO_INTERACT = "EARLY_END_NO_ELEMENTS_TO_INTERACT"
    EARLY_END_DURING_SYNTHESIS = "EARLY_END_DURING_SYNTHESIS"
    EARLY_END_INTERACTIVE_FAILED = "EARLY_END_INTERACTIVE_FAILED"
    EARLY_END_BEING_CANCELLED = "EARLY_END_BEING_CANCELLED"



@dataclass
class Element:
    union_bound: tuple[int, int, int, int] # (x, y, width, height) bounding box

    # metadata
    id: str
    name: str
    role: str # html role, link, button, input, etc.

    accessibility_tree_content: str

    action_type: ActionType | None = None
    value: str | None = None  # text content, attributes, etc. if it is an input

    
    def __init__(self, accessibility_tree_content: str, union_bound: tuple[int, int, int, int], element_id: str):
        self.accessibility_tree_content = accessibility_tree_content
        self.union_bound = tuple(union_bound)
        self.id = element_id
        

        if len(accessibility_tree_content) == 0:
            self.name = "empty"
            self.role = "empty"
            self.action_type = ActionType.NONE
        else:
            self.name = re.sub(r'^\[\d+\]\s*', '', accessibility_tree_content)
            self.role = self.name.split()[0]
            self.action_type = self.determine_action_type(self.role)

    @staticmethod
    def determine_action_type(role: str) -> ActionType | None:
        """Determine action type from element role. Note that many elements can map to multiple action types"""

        role_lower = role.lower()

        if role_lower in {
            'button', 'link', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
            'option', 'tab', 'checkbox', 'radio', 'switch', 'treeitem', 'rowheader'
        }:
            return ActionType.CLICK

        # 2. HOVER: reveal submenu, tooltip, or list of items
        elif role_lower in {
            'tooltip', 'menubar', 'menu', 'tablist'
        }:
            return ActionType.HOVER

        # 3. TYPE: text input or editable field
        elif role_lower in {
            'textbox', 'searchbox', 'combobox', 'spinbutton'
        }:
            return ActionType.TYPE

        # 4. PRESS: keyboard-driven adjustments or controls
        elif role_lower in {
            'slider', 'timer'
        }:
            return ActionType.PRESS

        # 5. SCROLL: scrollable content regions
        elif role_lower in {
            'document', 'main', 'article', 'feed', 'region', 'group',
            'list', 'listbox', 'tree', 'treegrid', 'grid', 'gridcell',
            'rowgroup', 'row', 'log', 'search', 'table', 'scrollbar', 'progressbar'
        }:
            return ActionType.SCROLL

        # default: no specific action
        return ActionType.NONE

    def is_need_a_value_input(self) -> bool:
        return self.action_type in {ActionType.TYPE, ActionType.PRESS, ActionType.SCROLL, ActionType.GOTO}

    @staticmethod
    def create_empty_element() -> "Element":
        """Create an empty element with no action type."""
        return Element(
            accessibility_tree_content="",
            union_bound=(0, 0, 0, 0),
            element_id="",
        )

    def __hash__(self):
        # follow os-genesis
        hash_str = f"{self.name}_{self.union_bound or None}"
        return tools_hash(hash_str)


@dataclass
class RawState:
    """Raw state information for the environment."""
    url: str
    accessibility_tree: str
    observation_metadata: dict[str, Any]
    screenshot: np.ndarray
    timestamp: float

    def __init__(self, url, accessibility_tree, observation_metadata, screenshot, timestamp: float | None = None):
        if url.endswith('/'):
            url = url[:-1]
        self.url = copy.deepcopy(url)
        self.accessibility_tree = copy.deepcopy(accessibility_tree)
        self.observation_metadata = copy.deepcopy(observation_metadata)
        self.screenshot = copy.deepcopy(screenshot)
        self.timestamp = timestamp if isinstance(timestamp, float) else time.time()
    
    def __hash__(self):
        # currently, we don't consider the status change after executing other tasks
        # that means, if the url, accessibility tree and screenshot are the same, we consider the state is the same
        hash_str = f"{self.url}_{self.accessibility_tree}"
        h = hashlib.sha256()
        h.update(hash_str.encode('utf-8'))
        img_bytes = self.screenshot.tobytes()
        h.update(img_bytes)
        return int(h.hexdigest(), 16)
    
    def hash_by_screenshot(self) -> int:
        img_bytes = self.screenshot.tobytes()
        h = hashlib.sha256()
        h.update(img_bytes)
        return int(h.hexdigest(), 16)

    def to_dict(self) -> dict:
        """Convert the raw state to a dictionary."""
        path = str(self.hash_by_screenshot()) + '.jpg'
        path = f"{os.environ[const_screenshot_save_path]}/{path}"
        if not os.path.exists(path): tools_ndarray_image_save(self.screenshot, path)

        return {
            "url": self.url,
            "accessibility_tree": self.accessibility_tree,
            "observation_metadata": self.observation_metadata,
            "screenshot": path,
            "timestamp": self.timestamp,
        }
    
    @staticmethod
    def from_dict(data: dict) -> "RawState":
        """Convert a dictionary back to RawState, loading screenshot from path."""
        from syn.tools import tools_load_png_rgba
        
        screenshot_path = data["screenshot"]
        if isinstance(screenshot_path, str) and os.path.exists(screenshot_path):
            if os.environ.get(const_is_load_screenshot_image, "1") == "1":
                screenshot = tools_load_png_rgba(screenshot_path)
            else:
                screenshot = screenshot_path
        elif os.environ.get(const_disable_screenshot_path_check, "0") == "0":
            raise ValueError(f"Screenshot path does not exist: {screenshot_path}")
        else:
            screenshot = screenshot_path
        
        return RawState(
            url=data["url"],
            accessibility_tree=data["accessibility_tree"],
            observation_metadata=data["observation_metadata"],
            screenshot=screenshot,
            timestamp=data["timestamp"]
        )
    
    def __eq__(self, other):
        if not isinstance(other, RawState):
            return False
        
        return (
            self.url == other.url and
            self.accessibility_tree == other.accessibility_tree and
            self.observation_metadata == other.observation_metadata
        )


@dataclass
class StateInfo:
    """Contains information about the current state of the environment."""
    raw_state: RawState
    elements: list[Element]
    summary: str | None = None
    
    def should_terminate(self) -> bool:
        return len(self.elements) == 0

    def __str__(self):
        return f"StateInfo(URL={self.raw_state.url}, Elements={len(self.elements)}, Summary={self.summary})"

    def __hash__(self):
        return hash(self.raw_state)

@dataclass
class Action:
    """Represents a low-level action executed in the environment."""
    action_type: ActionType # None represents for not assigned yet, not decided yet; for non-interactive actions, such as summary, it can be ActionType.NONE
    target_element: Element | None # can be None for non-interactive actions
    value: str | None = None
    coordinates: Optional[Tuple[int, int]] = None
    status: ActionExecuteStatus = ActionExecuteStatus.NOT_EXECUTED

    def __init__(self, element: Element | None, value: str | None, action_type: ActionType | None):
        if element is None:
            assert action_type is not None, "If element is None, action_type must be provided"
            assert not self._is_required_element(action_type), "If element is None, action_type cannot be CLICK, TYPE or HOVER"
            self.action_type = action_type
        
        elif action_type is not None:
            element.action_type = action_type

        self.action_type = action_type
        self.target_element = element
        self.coordinates = None
        self.status = ActionExecuteStatus.NOT_EXECUTED

        """value is the text to be typed, or the key to be pressed"""
        default_empty_value = " "
        if value is None: value = default_empty_value
        elif isinstance(value, str) and len(value) == 0: value = default_empty_value
        self.value = self._value_ascil(value)

        if self._is_required_value(self.action_type) and not self.value.strip():
            raise ValueError(f"Action {self.action_type} requires a value, but got an empty value\n{self}")
            
        elif not self._is_required_value(self.action_type) and self.value.strip():
            logger.warning(f"Action {self.action_type} does not require a value, but got={self.value}, thus reset to empty string")
            self.value = default_empty_value


    def __hash__(self):
        hash_str = str(self)
        return tools_hash(hash_str)

    @staticmethod
    def _is_required_value(action_type: ActionType) -> bool:
        return action_type in {ActionType.TYPE, ActionType.PRESS, ActionType.SCROLL, ActionType.GOTO, ActionType.STOP, ActionType.NONE}

    @staticmethod
    def _is_required_element(action_type: ActionType) -> bool:
        return action_type in {ActionType.CLICK, ActionType.HOVER, ActionType.TYPE}

    def get_action_str(self) -> str:
        """return action str for create_id_based_action format."""

        match self.action_type:
            case ActionType.CLICK:
                return f"click [{self.target_element.id}]"
            case ActionType.TYPE:
                text = self.value if self.value else ""
                return f"type [{self.target_element.id}] [{text}]"
            case ActionType.HOVER:
                return f"hover [{self.target_element.id}]"
            case ActionType.SCROLL:
                direction = self.value if self.value in ["up", "down"] else "down"
                return f"scroll [{direction}]"
            case ActionType.PRESS:
                key_comb = self.value if self.value else ""
                return f"press [{key_comb}]"
            case ActionType.NONE | ActionType.STOP:
                return f"none"
            case ActionType.GOTO:
                return f"goto [{self.value}]"
            case ActionType.GO_BACK:
                return f"go_back"
            case ActionType.GO_FORWARD:
                return f"go_forward"
            case _:
                raise RuntimeError(f"Unknown action type: {self.action_type}, target element={self.target_element}")
    
    def __str__(self):

        """
        CLICK = "click"
        TYPE = "type"1
        HOVER = "hover"
        SCROLL = "scroll"1
        PRESS = "press"1
        NONE = "none" # do nothing1
        """

        if self.action_type is None:
            logger.warning("Action type is None (not assigned yet), but you are trying to use it as a string")
            action_str = "NOT-ASSIGNED"
            target_str = self.target_element.accessibility_tree_content
            value_str = "None"
        
        elif self.action_type == ActionType.NONE:
            action_str = "none (non-interactive action such as summary)"
            value_str = f"{self.value}"
            target_str = "None"
            
        elif self.target_element is None: # can only be scroll or press
            assert isinstance(self.value, str) and self.value.strip(), f"If target_element is None, value must be a non-empty string for action_type={self.action_type}"
            action_str = self.action_type.value
            target_str = "None"
            value_str = f"{self.value}"
        
        elif self.action_type is ActionType.TYPE: # type
            action_str = self.action_type.value
            target_str = self.target_element.accessibility_tree_content
            value_str = f"{self.value}"
        
        elif self.action_type in {ActionType.CLICK, ActionType.HOVER}:
            action_str = self.action_type.value
            target_str = self.target_element.accessibility_tree_content
            value_str = "None"
        
        elif self.action_type in {ActionType.GO_BACK, ActionType.GO_FORWARD, ActionType.GOTO}:
            action_str = self.action_type.value
            target_str = "None"
            value_str = f"{self.value}"
        
        elif self.action_type == ActionType.STOP:
            action_str = "stop (find the task is inpossible to complete)"
            target_str = "None"
            value_str = f"{self.value}"

        
        else:
            raise RuntimeError(f"Unknown action type: {self.action_type}, target element={self.target_element}, Value='{self.value}'")

        return f"{{Action: '{action_str}', Target: '{target_str}', Value: '{value_str}'}}"

    def _value_ascil(self, input_value: str) -> str:
        if len(input_value) > 0:
            input_value = input_value.replace("’", "'").replace("‘", "'")
            input_value = input_value.replace("“", '"').replace("”", '"')

            input_value = input_value.replace("–", "-").replace("—", "-")

            input_value = input_value.replace("…", "...")

            input_value = input_value.replace(" ", " ").replace(" ", " ")

            input_value = input_value.replace("•", "*")
        return input_value

@dataclass
class LowLevelTask:
    task: str | None # description of the low level task, such as ""Type 'OpenAI' into the search bar to find relevant articles."", more like the reason for the action
    curr_state: StateInfo
    action: Action
    state_after: StateInfo | None = None # none indicates unknown
    task_status: LowTaskStatus = LowTaskStatus.NOTACHIEVEABLE
    reasoning: str | None = None

    def __hash__(self):
        hash_str = f"{self.task}_{hash(self.curr_state)}_{hash(self.action)}_{self.task_status.name}"
        return tools_hash(hash_str)

    def is_executed(self) -> bool:
        """Check if the task has been executed."""
        return self.state_after is not None

@dataclass
class HighLevelTask:
    """High-level task to be accomplished in the exploration."""
    task: str # high level task description, such as "find a product with price less than 100"
    trajectories: list[LowLevelTask]
    start_url: str | None = None  # the url where the high-level task starts, can be None if not specified

    def __hash__(self):
        hash_str = f"{self.task}_" + "_".join([str(hash(t)) for t in self.trajectories])
        return tools_hash(hash_str)

@dataclass
class ExplorationTraj:
    """represent the whole life of an exploration"""
    # todo: record the execution like states

    curr_state: StateInfo  # to record the environment state, in general, this equals to the next state of the last low-level task of the last high-level task
    high_level_tasks: list[HighLevelTask] = field(default_factory=list)  # high-level tasks evolved during exploration
    status: ExplorationTrajStatus = ExplorationTrajStatus.IN_PROGRESS
    forked_nodes: list["ExplorationTraj"] = field(default_factory=list)  # related nodes in the exploration graph, used for visualization and debugging
    parent_node: Optional["ExplorationTraj"] = None  # parent node in the exploration graph, used for visualization and debugging
    uuid: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self, simplified: bool = False) -> dict:
        """not simplified, that means you can restore all from the saved dict"""
        if not simplified:
            return {
                'curr_state': tools_serialize_dataclass(self.curr_state),
                'high_level_tasks': tools_serialize_dataclass(self.high_level_tasks),
                'status': tools_serialize_dataclass(self.status),
                'forked_nodes': [node.uuid for node in self.forked_nodes],
                'parent_node': self.parent_node.uuid if self.parent_node else None,
            }
        else:

            states = []
            for low_level_task in self.high_level_tasks[-1].trajectories:
                if low_level_task.state_after is None:
                    after_state = None
                else:
                    after_state = low_level_task.state_after.raw_state.to_dict()['screenshot']
                states.append(
                    {
                        'task': low_level_task.task,
                        'action': str(low_level_task.action),
                        'curr_state': low_level_task.curr_state.raw_state.to_dict()['screenshot'],
                        'curr_state_summary': low_level_task.curr_state.summary,
                        'state_after': after_state,
                    }
                )

            return {
                'curr_state': (self.curr_state.raw_state.url, self.curr_state.summary),
                'high_level_tasks': [task.task for task in self.high_level_tasks],
                'status': tools_serialize_dataclass(self.status),
                "low_level_trajectory": states,
                'forked_nodes': [node.uuid for node in self.forked_nodes],
                'parent_node': self.parent_node.uuid if self.parent_node else None,
            }

        
    def clone(self, ) -> "ExplorationTraj":
        """Create a deep copy of the current high-level tasj synthesis Exploration trajectory."""
        temp = ExplorationTraj(
            curr_state=copy.deepcopy(self.curr_state),
            high_level_tasks=copy.deepcopy(self.high_level_tasks),
            status=self.status,
            parent_node=self,
        )
        self.forked_nodes.append(temp)
        return temp
    
    def add_low_level_task(self, task: LowLevelTask):
        if len(self.high_level_tasks) == 0:
            self.high_level_tasks.append(HighLevelTask(task="empty task.", trajectories=[]))

        self.high_level_tasks[-1].trajectories.append(task)
    
    def add_high_level_task(self, task_str: str, next_state: StateInfo) -> "ExplorationTraj":
        """Add a new high-level task to the exploration trajectory."""
        trajectories = [] if len(self.high_level_tasks) == 0 else self.high_level_tasks[-1].trajectories
        task = HighLevelTask(task=task_str, trajectories=trajectories)
        self.high_level_tasks.append(task)
        self.curr_state = next_state
        return self

    def get_current_low_level_task(self) -> LowLevelTask:
        assert len(self.high_level_tasks) > 0, "No high-level tasks available"
        assert len(self.high_level_tasks[-1].trajectories) > 0, "No low-level tasks available in the current high-level task"
        assert self.high_level_tasks[-1].trajectories[-1].is_executed() is False, "The last low-level task has already been executed"
        return self.high_level_tasks[-1].trajectories[-1]

    def end_exploration(self, status: ExplorationTrajStatus = ExplorationTrajStatus.END):
        assert status not in {ExplorationTrajStatus.IN_PROGRESS}, f"Invalid status={status} for ending exploration"
        self.status = status

        if self.status in {ExplorationTrajStatus.END}:
            self.high_level_tasks[-1].trajectories[-1].task_status = LowTaskStatus.END
        else:
            self.high_level_tasks[-1].trajectories[-1].task_status = LowTaskStatus.IN_PROGRESS
