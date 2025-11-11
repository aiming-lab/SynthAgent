import os
import json
from dataclasses import dataclass, field
from syn.args import GPTConfig
from syn.prompts import prompt_refine_trajectory
from syn.tools import (
    tools_get_time,
    tools_elapsed_time_print,
    tools_jsonl_save,
    tools_jsonl_load,
    tools_robust_json_loads,
)
from syn.gpt import GPTClient
from loguru import logger
from simpleArgParser import parse_args


@dataclass
class TrajScoreConfig:
    input: str
    gpt: GPTConfig = field(default_factory=GPTConfig)

    def pre_process(self):
        assert os.path.exists(self.input), f"Input folder={self.input} does not exist"
        if not os.path.exists(f"{self.input}/multiagent"):
            logger.warning(f"Input folder {self.input} does not contain 'multiagent' subfolder, you may want to check your input path")



class TrajScoreAgent:
    def __init__(self, config: TrajScoreConfig):
        self.config: TrajScoreConfig = config
        os.makedirs(self.config.input, exist_ok=True)

        self.tasks_done_buffer: list[dict] = []
        self.task2item: dict[str, dict] = {}
        self.decisions_done: list[dict] = []   # raw model decisions per task
        self.refined_done: list[dict] = []     # refined tasks (keep/refine only)

        self.gpt_client = GPTClient(
            provider=config.gpt.provider,
            base_url=config.gpt.openai_api_base,
        )
        self.load()

    def save(self):
        # Save decisions
        if self.decisions_done:
            tools_jsonl_save(self.decisions_done, f"{self.config.input}/tasks_done_decision.jsonl", append=False)
        if self.refined_done:
            tools_jsonl_save(self.refined_done, f"{self.config.input}/tasks_done_refined.jsonl", append=False)

    def load(self):
        # Load dicts straight from JSONL
        src_path = f"{self.config.input}/tasks_done.jsonl"
        if os.path.exists(src_path):
            done_buffer = tools_jsonl_load(src_path)  # -> list[dict]
        else:
            raise FileNotFoundError(f"tasks_done.jsonl not found: {src_path}")
            
        if os.path.exists(f"{self.config.input}/tasks_done_decision.jsonl"):
            self.decisions_done = tools_jsonl_load(f"{self.config.input}/tasks_done_decision.jsonl")

        self.tasks_done_buffer = done_buffer
        self.task2item = {t['task']: t for t in done_buffer}
        logger.info(f"Loaded {len(self.tasks_done_buffer)} high-level tasks from {src_path}")

    @staticmethod
    def reorder_list(original: list, order: list[int] | int) -> list:
        """Reorder `original` according to 0-based `order`."""
        if isinstance(order, int):
            order = [order]
        assert all(isinstance(i, int) for i in order), "order contains non-integers"
        assert all(0 <= i < len(original) for i in order), "order has out-of-range indices"
        return [original[i] for i in order]

    @staticmethod
    def _ensure_final_none(trajectory: list[dict], final_none_value: str, modify_end: bool, append_end: bool) -> list[dict]:
        """Ensure the final step is a NONE action with the provided non-empty value, either by modifying the last step or appending a new one."""
        assert isinstance(final_none_value, str) and final_none_value.strip(), "final_none_value must be non-empty"
        end_action = {'action_type': 'none', 'target_element': None, 'value': final_none_value, 'coordinates': None, 'status': 'NOT_EXECUTED'}
        def _to_str(x):
            if isinstance(x, str): return x
            try: return json.dumps(x, ensure_ascii=False)
            except Exception: return str(x)

        if not trajectory:
            trajectory = []
            append_end = True

        if modify_end and trajectory:
            last = dict(trajectory[-1])
            last["action"] = end_action
            trajectory = trajectory[:-1] + [last]
        elif append_end:
            last = trajectory[-1]
            curr_state = (last.get("curr_state") or None)
            
            step = {
                "task": last.get("task") or "",
                "curr_state": curr_state,
                "action": end_action,
                "state_after": None,
                "task_status": "END",
                "reasoning": f"Finalized with NONE: {_to_str(final_none_value)}",
            }

            trajectory = trajectory + [step]
        else:
            last = dict(trajectory[-1])
            last["action"] = end_action
            trajectory = trajectory[:-1] + [last]

        return trajectory

    def _format_traj_for_prompt(self, sample: dict) -> str:
        """Format one task dict into the prompt block."""
        steps = sample.get("trajectories", []) or []
        formatted_steps = []
        for step in steps:
            obs = (((step or {}).get("curr_state") or {}).get("summary")) or ""
            act = (step or {}).get("action", "")
            aft = (((step or {}).get("state_after") or {}).get("summary")) or ""
            if not isinstance(act, str):
                try:
                    act = json.dumps(act, ensure_ascii=False)
                except Exception:
                    act = str(act)
            formatted_steps.append(f"Observation: {obs}\nAction: {act}\nObservation After Action: {aft}")
        trajectory_str = "\n\n".join(formatted_steps)
        high_level_task = sample.get("task") or "<UNKNOWN_TASK>"
        return f"Length of trajectories: {len(steps)}\nHigh-Level Task: {high_level_task}\nTrajectory:\n{trajectory_str}"

    def _format_request(self, traj_block: str) -> dict:
        return {
            "messages": prompt_refine_trajectory(traj_block),
            "model": self.config.gpt.model,
            "temperature": self.config.gpt.temperature,
            "max_tokens": self.config.gpt.max_completion_tokens,
        }

    def _apply_decision(self, decision_obj: dict) -> dict | None:
        task_name = decision_obj.get("task", "")
        decision = decision_obj.get("decision", "")
        order = decision_obj.get("order", [])
        final_none_value = decision_obj.get("final_none_value", "")
        modify_end = bool(decision_obj.get("modify_end", False))
        append_end = bool(decision_obj.get("append_end", False))

        task_dict = self.task2item.get(task_name)
        if task_dict is None:
            logger.error(f"Task '{task_name}' not found in buffer.")
            return None

        if decision == "drop":
            return None

        # keep/refine path: reorder and ensure final NONE with non-empty value
        traj = task_dict.get("trajectories", []) or []
        if not isinstance(traj, list):
            logger.error(f"Task '{task_name}' has invalid 'trajectories' (not a list).")
            return None

        try:
            if order:
                traj = self.reorder_list(traj, order)
            # Enforce final NONE with non-empty value
            traj = self._ensure_final_none(traj, final_none_value, modify_end=modify_end, append_end=append_end)
        except AssertionError as e:
            logger.error(f"Task '{task_name}': order/final_none enforcement failed: {e}")
            return None

        refined = dict(task_dict)
        refined["trajectories"] = traj
        return refined

    def run(self):
        decision_path = f"{self.config.input}/tasks_done_decision.jsonl"
        if self.decisions_done:
            logger.info(f"{decision_path} already exists; skipping.")
            refineds = []
            for d in self.decisions_done:
                refined = self._apply_decision(d)
                if refined is not None:
                    refineds.append(refined)
            self.refined_done = refineds
            self.save()
            return
            

        if not self.tasks_done_buffer:
            logger.error("No tasks in buffer.")
            return

        # Build requests
        requests = []
        for task in self.tasks_done_buffer:
            traj_block = self._format_traj_for_prompt(task)
            requests.append(self._format_request(traj_block))

        try:
            responses = self.gpt_client.batch_requests(
                requests=requests,
                json_mode=True,
            )
            for resp in responses:
                response_text = getattr(resp, "message", None).content if hasattr(resp, "message") else resp["message"]["content"]
                data = tools_robust_json_loads(response_text)
                if not isinstance(data, dict) or len(data) == 0:
                    logger.error(f"Expected dict from model, got {type(data)}: {response_text!r}")
                    continue

                # Keep the raw decision object
                decision_obj = {
                    "task": data.get("task", ""),
                    "score": data.get("score"),
                    "decision": data.get("decision"),
                    "order": data.get("order", []),
                    "modify_end": data.get("modify_end", False),
                    "append_end": data.get("append_end", False),
                    "final_none_value": data.get("final_none_value", ""),
                    "drop_reason": data.get("drop_reason", ""),
                    "modification_reason": data.get("modification_reason", ""),
                }
                self.decisions_done.append(decision_obj)

                # Apply decision for keep/refine to produce refined trajectory
                refined = self._apply_decision(decision_obj)
                if refined is not None:
                    self.refined_done.append(refined)

            self.save()

        except Exception:
            logger.exception("Error during trajectory scoring/refinement")


if __name__ == "__main__":
    args: TrajScoreConfig = parse_args(TrajScoreConfig)
    start_time = tools_get_time()
    logger.info(f"Starting TrajScoreAgent with config\n{args}\nStart time: {start_time}")

    if os.path.exists(m := f"{args.input}/multiagent"):
        for sub in os.listdir(m):
            args.input = f"{m}/{sub}"
            logger.info(f"Processing subfolder: {args.input}")
            agent = TrajScoreAgent(args)
            agent.run()
    else:
        agent = TrajScoreAgent(args)
        agent.run()

    logger.info(f"TrajScoreAgent done! started at {start_time} Elapsed time: {tools_elapsed_time_print(start_time)}\n{args}")
