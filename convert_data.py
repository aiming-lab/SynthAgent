from syn.tools import tools_jsonl_load, tools_deserialize_dataclass
from syn.data import HighLevelTask, LowLevelTask, StateInfo, ActionType
from syn.prompts import prompt_action_from_observation_adapt_from_webarena
from syn.base_explore import Explorer
from syn.consts import const_is_load_screenshot_image, const_enable_logging_stat_time_block
from syn.utils import stat_time_block
import os
import re
import json
from concurrent.futures import ProcessPoolExecutor
from simpleArgParser import parse_args
from dataclasses import dataclass
from tqdm import tqdm
from functools import partial, reduce
from transformers import AutoProcessor
import tarfile
import copy
import random
from enum import Enum

class FilterStrategy(Enum):
    rule_correct = "rule_correct" # test set only
    judge_complete = "judge_complete"
    duplicate_task = "duplicate_task" 

@dataclass
class Config:
    # input folder
    input: str
    output: str
    only_path: bool = True
    limit: int | None = None
    cpu: int = 32
    cutoff_len: int = 32768
    image_len: int = 1198 + 32
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    strategy: list[FilterStrategy] | None = None  # optional filtering strategy
    history_last_k: int | None = 3
    load_refined: bool = True  # load refined data if available

    def pre_process(self):
        assert os.path.exists(self.input) and os.path.isdir(self.input), f"input {self.input} must be an existing folder"
        assert self.output.endswith(".json"), f"output {self.output} must be a json file"
        os.makedirs(os.path.dirname(self.output), exist_ok=True)
        self.cpu = min(self.cpu, os.cpu_count() or 1)

def lambda_de(item):
    return tools_deserialize_dataclass(item, HighLevelTask)

def lambda_de_list(list_item: list) -> list[HighLevelTask]:
    return tools_deserialize_dataclass(list_item, list[HighLevelTask])

def check_element_id_in_current_page(page: str, eid: str) -> bool:
    pat = rf"(?m)^[ \t]*\[{eid}\][^\r\n]*$"
    m = re.search(pat, page)
    return m is not None

def form_message(sample: HighLevelTask, image_folder: str, history_last_k: int | None) -> list[dict]:
#       {
#     "messages": [
#       {
#         "content": "<image>Who are they?",
#         "role": "user"
#       },
#       {
#         "content": "They're Kane and Gretzka from Bayern Munich.",
#         "role": "assistant"
#       },
#       {
#         "content": "What are they doing?<image>",
#         "role": "user"
#       },
#       {
#         "content": "They are celebrating on the soccer field.",
#         "role": "assistant"
#       }
#     ],
#     "images": [
#       "mllm_demo_data/1.jpg",
#       "mllm_demo_data/1.jpg"
#     ]
#   },

    results = []
    # breakpoint()
    for idx, step0 in enumerate(sample.trajectories):

        # filter invalid data
        if step0.action.action_type is ActionType.STOP and step0.task == "failed during cot_step":
            break
        elif step0.action.action_type is ActionType.REFLECT:
            break

        if step0.curr_state.raw_state.screenshot is not None:
            if not os.path.exists(step0.curr_state.raw_state.screenshot):
                print(f"Screenshot not found: {step0.curr_state.raw_state.screenshot}, skipped")
                continue
            screenshot = f"{image_folder}/" + os.path.basename(step0.curr_state.raw_state.screenshot)
        else:
            screenshot = None
        
        if step0.action.target_element is not None and not check_element_id_in_current_page(step0.curr_state.raw_state.accessibility_tree, step0.action.target_element.id):
            print(f"Warning: element {step0.action.target_element} not found in current page, skipped")
            continue
        
        if history_last_k is None: # this need to be modifyed for reordering
            # includes all histories
            history = sample.trajectories[:idx]
        else:
            history = sample.trajectories[max(0, idx - history_last_k):idx]

        input_message: list[dict] = prompt_action_from_observation_adapt_from_webarena(
            url=step0.curr_state.raw_state.url,
            page_context=Explorer._format_page_context(step0.curr_state),
            elements=Explorer._format_elements_for_llm(step0.curr_state.elements, excluding_elements=set()),
            previous_state_action=Explorer._format_previous_observation_and_action(history, include_all_steps=True),
            screenshot=screenshot,
            high_level_task=sample.task,
            return_fine_tune_format=True,
            history_last_k=history_last_k,
        )

    
        output_message = {
            'state_observation_summary': step0.curr_state.summary,
            'reasoning': step0.reasoning,
            'next_action': {
                'low-level_instruction': step0.task,
                'action': {
                    'type': step0.action.action_type.name,
                    'element_id': step0.action.target_element.id if step0.action.target_element else None, # warning, the id map problem
                    'value': step0.action.value,
                }
            }
        }
        output_message = str(output_message)

        input_message.append(
            {
                'role': 'assistant',
                'content': output_message,
            }
        )

        messages = {
            'messages': input_message,
        }
        if screenshot is not None:
            messages['images'] = [screenshot]
        
        txt_msg = json.dumps(messages)
        if (cnt := txt_msg.count('<image>')) != 1:
            print(f"Warning: message does not contain exactly one <image> token, actually have {cnt} tokens, thus skipped")
            continue

        results.append(messages)

        if step0.action.action_type in {ActionType.STOP, ActionType.NONE}:
            break

    
    return results

def form_message_list(samples: list[HighLevelTask], image_folder: str, history_last_k: int | None) -> list[dict]:
    return reduce(lambda acc, x: acc + form_message(x, image_folder, history_last_k), samples, [])

def data_cleaning_by_cutoff_len(data: list, cutoff_len: int, image_len: int, model: str) -> list:
    conversations = []
    BATCH = 128
    for item in data:
        user_text = item['messages'][0]['content'].strip("<image>")
        conv = [
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": item['messages'][1]['content']}
            ]}
        ]
        conversations.append(conv)

    proc = AutoProcessor.from_pretrained(model)

    removed_idx = []
    removed_lens = []

    n = len(conversations)
    num_batches = (n + BATCH - 1) // BATCH

    for b in tqdm(range(num_batches), desc="Tokenizing", total=num_batches):
        start = b * BATCH
        end = min(start + BATCH, n)
        chunk = conversations[start:end]

        texts = [
            proc.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in chunk
        ]
        enc = proc.tokenizer(texts, add_special_tokens=False)

        for local_i, ids in enumerate(enc["input_ids"]):
            L = len(ids)
            if L > cutoff_len - image_len:
                idx = start + local_i
                removed_idx.append(idx)
                removed_lens.append(L)

    removed_set = set(removed_idx)
    filtered_data = [sample for i, sample in enumerate(data) if i not in removed_set]

    return filtered_data

def filter_raw_data(data: list[dict], status: dict, strategy: list[FilterStrategy] | None = None):
    if strategy is None: return data

    filter_status = {}
    cnt = 0
    for task, s in status.items():
        cat = None
        if '@' in task: 
            cat = task.split('@')[1]
            task = task.split('@')[0]
        if 'high_level_tasks' in s:
            last_task = s['high_level_tasks'][-1]
            if task != last_task:
                filter_status[last_task] = s

        if isinstance(s, dict):
            filter_status[task] = s
        elif isinstance(s, str):
            if 'auto_eval=success' in s:
                auto_eval = 'success'
            else:
                auto_eval = 'failed'
            
            if s.startswith('completed'):
                judge = 'completed'
            elif s.startswith('not_completed_with_max_steps'):
                judge = 'exceeded_max_steps'
            else:
                judge = 'not_achievable'
                
            filter_status[task] = {
                'auto-eval': auto_eval,
                'end_reason': judge,
                'category': cat
            }
        
    print(f"found {len(filter_status)} status, {cnt} in 3 single site categories")

    for sta in strategy:
        match sta:
            case FilterStrategy.rule_correct:
                ori_data_num = len(data)
                data = [d for d in data if d['task'] not in filter_status or filter_status[d['task']]['auto-eval'] == 'success']
                print(f"filtered out {ori_data_num - len(data)} samples using {sta}")
            case FilterStrategy.judge_complete:
                ori_data_num = len(data)
                data = [d for d in data if d['task'] not in filter_status or filter_status[d['task']]['end_reason'] == 'completed']
                print(f"filtered out {ori_data_num - len(data)} samples using {sta}")
            case FilterStrategy.duplicate_task:
                seen_tasks = set()
                unique_data = []
                for d in data:
                    if d['task'] not in seen_tasks:
                        unique_data.append(d)
                        seen_tasks.add(d['task'])
                ori_data_num = len(data)
                data = unique_data
                print(f"filtered out {ori_data_num - len(data)} samples using {sta}")
                    
            case _:
                pass

    return data




if __name__ == '__main__':
    args: Config = parse_args(Config)
    os.environ[const_is_load_screenshot_image] = str(int(not args.only_path))  # faster load
    os.environ[const_enable_logging_stat_time_block] = "1"  # enable time logging

        
    status = json.load(open(f"{args.input}/tasks_done_unique.json", 'r'))
    print('loaded status for tasks =', len(status))

    data = []
    if os.path.exists(f"{args.input}/multiagent"):
        for subfolder in os.listdir(f"{args.input}/multiagent"):
            if not os.path.isdir(f"{args.input}/multiagent/{subfolder}"):
                continue
            try:
                int(subfolder)
            except:
                continue

            if args.load_refined:
                p = f"{args.input}/multiagent/{subfolder}/tasks_done_refined.jsonl"

            else:
                p = f"{args.input}/multiagent/{subfolder}/tasks_done.jsonl"
            sliced_data = tools_jsonl_load(p)
            data.extend(sliced_data)
            print(f'loaded {len(sliced_data)} from {subfolder}, total {len(data)}')

            if isinstance(args.limit, int) and len(data) >= args.limit:
                print(f"reach to the data {len(data)} >= limit = {args.limit}, stop loading more")
                data = data[:args.limit]
                break
    else:
        assert os.path.exists(f"{args.input}/tasks_done.jsonl"), f"input {args.input} must contain tasks_done.jsonl"
        data = tools_jsonl_load(f"{args.input}/tasks_done.jsonl")
        assert isinstance(data, list) and len(data) > 0, f"data loaded from {args.input}/tasks_done.jsonl is empty"

    if isinstance(args.limit, int):
        data = data[:args.limit]

    print('loaded tasks =', len(data))

    # filter by strategy
    print(f"filtering data with strategy: {args.strategy}")
    data = filter_raw_data(data, status, args.strategy)
    print('after filtering, tasks =', len(data))
    task_num = len(data)

    # de-seri
    max_workers = max(args.cpu, os.cpu_count() or 1)
    batch_size = len(data) // max_workers + 1
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    with stat_time_block(note=f'de-serializing {len(data)} items with {max_workers} workers in {len(batches)} batches'):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # show progress while consuming the iterator
            batch_results = list(tqdm(ex.map(lambda_de_list, batches), total=len(batches), desc="de-serializing"))
    samples = [item for batch in batch_results for item in batch]  # flatten
    print(type(samples[0]), len(samples))

    # format to training data
    output_base_name = os.path.basename(args.output).rstrip(".json")
    output_image_folder = f"images/{os.path.basename(args.input)}"


    max_workers = args.cpu
    batch_size = len(samples) // max_workers + 1
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    with stat_time_block(note=f'formating {len(samples)} items with {max_workers} workers in {len(batches)} batches'):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # show progress while consuming the iterator
            batch_results = list(tqdm(ex.map(partial(form_message_list, image_folder=output_image_folder, history_last_k=args.history_last_k), batches), total=len(batches), desc="formating"))
    
    results = [item for batch in batch_results for item in batch]  # flatten

    print(f'final formatted {len(results)} items')
    

    # cleaning
    cleaned = data_cleaning_by_cutoff_len(results, cutoff_len=args.cutoff_len, image_len=args.image_len, model=args.model)
    print(f"original = {len(results)}, cleaned = {len(cleaned)}")
    results = cleaned

    # saving
    print(f"saving to {args.output}")
    random.seed(0)
    random.shuffle(results)
    with open(args.output, "w") as f:
        json.dump(results, f)
    print('results num=', len(results))
    sample_num = len(results)
    
    abs_input_path_screenshots = os.path.abspath(f"{args.input}/screenshots")
    os.chdir(os.path.dirname(args.output))
    os.makedirs('./images', exist_ok=True)


    if os.path.exists(c_path := f"dataset_info.json"):
        dataset_info = json.load(open(c_path, 'r'))
        dataset_info[output_base_name] = copy.deepcopy(dataset_info['mllm_demo'])
        dataset_info[output_base_name]['file_name'] = os.path.basename(args.output)
        json.dump(dataset_info, open(c_path, "w"), indent=4)
    else:
        print(f"Warning: dataset_info.json not found in {os.getcwd()}, skipped updating it")
    

    print(f'output_image_folder: {output_image_folder}, cwd={os.getcwd()}')
    if os.path.exists(t := f"{output_image_folder}.tar.gz") or os.path.isdir(output_image_folder):
        print(f"image folder={output_image_folder} exists")
    else:
        print(f"creating image folder={output_image_folder}.tar.gz")
        os.chdir('./images')

        with tarfile.open(f"{os.path.basename(output_image_folder)}.tar.gz", "w:gz") as tar:
            tar.add(abs_input_path_screenshots, arcname=os.path.basename(output_image_folder))
        
        with tarfile.open(f"{os.path.basename(output_image_folder)}.tar.gz", 'r:gz') as tar:
            tar.extractall()

    print('='*100)
    print(args)
    print(f"task_num = {task_num}, sample_num = {sample_num}")
    print('='*100)

