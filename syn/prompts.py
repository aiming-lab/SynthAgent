import numpy as np
from syn.tools import tools_ndarray_to_base64_image, tools_draw_red_bbox
from syn.consts import const_uninteractive_category, const_web_task_examples


def prompt_action_from_observation_adapt_from_webarena(
    url: str,
    elements: list[dict],
    page_context: str,
    screenshot: "np.ndarray",
    high_level_task: str,
    previous_state_action: list[dict],
    return_fine_tune_format: bool = False,
    history_last_k: int | None = None,
) -> list[dict]:
    
    if history_last_k is None:
        hint_for_history = "from oldest to most recent"
    else:
        hint_for_history = f"showing only the {history_last_k} most recent, listed from oldest to newest"

    img_info = "<image is provided in the attachment>" if screenshot is not None else "<image is not available>"

    prompt = f"""You are a GUI (Graphical User Interface) Web Agent expert capable of long-horizon planning and executing high-level tasks on a website. Based on the observations and the high-level task to complete, generate the next low-level instruction.

**Information**

1. High-Level Task (your ultimate goal to finish):
"{high_level_task}"

2. Current Page (only current view, not full page, you may need to scroll to see more):
    - URL:
      {url}

    - Accessibility Tree (Page Context):
      {page_context}

    - Elements (addressable in this view):
      {elements}
      
    - Screenshot (only current view, not full page):
      {img_info}

3. History of Actions ({hint_for_history}):
{previous_state_action}

**Available Low-Level Actions (exact JSON formats)**

- CLICK:
  {{"type": "CLICK", "element_id": <int>, "value": ""}}

- TYPE (default behavior is to press Enter after typing unless you explicitly set press_enter_after to 0):
  {{"type": "TYPE", "element_id": <int>, "value": "text to type <string>"}}

- HOVER:
  {{"type": "HOVER", "element_id": <int>, "value": ""}}

- PRESS (keyboard shortcut, e.g., Ctrl+V / Cmd+V):
  {{"type": "PRESS", "element_id": "", "value": "key_comb <string>"}}

- SCROLL:
  {{"type": "SCROLL", "element_id": "", "value": "up" or "down"}}

- GOTO (navigate directly to a URL):
  {{"type": "GOTO", "element_id": "", "value": "url <string>"}}

- GO_BACK:
  {{"type": "GO_BACK", "element_id": "", "value": ""}}

- GO_FORWARD:
  {{"type": "GO_FORWARD", "element_id": "", "value": ""}}

- NONE (use ONLY when a short summary/finalization step is needed but no browser action is required):
  {{"type": "NONE", "element_id": "", "value": "summary content or final answer <string>"}}

- STOP (use ONLY when you find the given high-level task is impossible to complete, or requires additional information, or has hallucinations, or has untrue information against the observations and action history):
  {{"type": "STOP", "element_id": "", "value": "the reason for the STOP <string>"}}

**Critical Rules for Success**
1. You must issue only actions that are valid given the current observation (elements, accessibility tree, and screenshot).
2. Only propose ONE atomic action per item in your Potential-Actions list; actions must be executable independently.
3. Prefer actions grounded by element IDs present in the current Elements list when using CLICK/TYPE/HOVER.
4. You MUST provide meaningful and non-empty value if the action type is in {{TYPE, SCROLL, GOTO, NONE, STOP}}.
5. When you believe the task is complete (e.g., you have the answer), use NONE with the final answer in value. Do not propose further actions after NONE.
6. Be concise and avoid redundant or risky actions; ensure each proposed action clearly advances the task.
7. The target High-Level Task may be hallucinated or low-quality that cannot be finished, in that case, please cautiously choose "STOP" action to terminate based on your observations and action histories. Please use the following pseudo-code to determine if the task is impossible to complete:
    ```pseudo-code
if high_level_task lacks necessary information to complete (such as "create a user while not providing the required information"):
    return STOP

if high-level task contains hallucinations (such as "delete the credit card ending 0123 while this credit card does not exist"):
    return STOP

if high-level task is not appropriate or not reasonable or harmful (such as "delete the database" or "delete the account"):
    return STOP

if History of Actions contain multiple attempts (>=3) to complete similar low-level instructions and actions:
    return STOP

return NON-STOP Actions (please consider other actions)
```

8. You MUST first generate a "state_observation_summary" to observe the current environment, then take a step-by-step "reasoning" to decide the next action. Finally, you need to decide the details for "next_action". You can also record any potential useful information in "state_observation_summary" for future reference (like a memory).
9. IMPORTANT: The high-level task often requires MULTIPLE steps to complete, so you should NOT expect to finish the task in a single action. You need to carefully reason about the current state and decide the next action which will cause the change to the current state, and then you can continue to exeucte. 
10. You MUST actively decide the next step to finish the task, YOU CANNOT choose "NONE" or "STOP" as the next action unless you are sure that the task is finished or impossible to complete. You cannot wait for something to happen, thus you MUST actively explore and take analysis.
11. The "reasoning" part MUST include your analysis of executing the pseudo-code for deciding "STOP" or "NON-STOP Actions".
12. You MUST actively explore different approaches to complete the task before reaching "STOP" state. For example, if you encounter a situation where the current approach is not yielding results, you should consider alternative strategies or approaches.
13. You MUST choose the element from the provided "Elements (addressable in this view)" list and use the element_id from that chosen element dict. The image screenshot may be not accurate, and the element in "Accessibility Tree (Page Context)" may not be reliable. You MUST include a separate analysis in the "reasoning" part to justify your choice folowing this rule.
14. If you observe that the current page does not change after executing the action, maybe you need to scroll down to see more elements of the page. 
15. Special Notes You MUST Pay Attention To:
 - when you type input the date, use the format "MM/DD/YYYY".


**Output Requirements**
You MUST return a JSON dictionary with the following format (no extra commentary):
{{
    "state_observation_summary": "your 1-3 sentence summary of the current state relevant to the task",
    "reasoning": "your step-by-step reasoning to decide the next action, accurately decide the action type, element_id, and value. decide your action based on the RULES above",
    "next_action": {{
        "low-level_instruction": "A concise instruction that describes the action to be taken, such as "click on the button with ID 'submit'". Format it as a string.",
        "action": {{"type": "XXXX", "element_id": <int or "", if int, make sure it exists in the provided Element list>, "value": <string or empty string per the action spec>}}
    }},
}}
RETURN ONLY THE DICTIONARY I ASKED FOR WITHOUT ANY COMMENTARY."""

    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if return_fine_tune_format:
        prompt = prompt.replace('<image>', '&lt;image&gt;')
        if screenshot is not None:
            message[0]["content"] = f"{prompt}<image>"
        else:
            message[0]['content'] = prompt
            
        return message


    if screenshot is not None:
        message[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": tools_ndarray_to_base64_image(screenshot)},
            },
        )

    return message

def prompt_refine_during_execution(
        curr_state_context: str,
        curr_url: str,
        curr_screenshot: "np.ndarray",
        previous_state_action: list[dict],
        current_high_level_task: str,
        previous_high_level_tasks: list[str],
        history_last_k: int | None = None,
) -> list[dict]:

    if history_last_k is None:
        hint_for_history = "from oldest to most recent"
    else:
        hint_for_history = f"showing only the {history_last_k} most recent, listed from oldest to newest"

    img_info = "<image is provided in the attachment>" if curr_screenshot is not None else "<image is not available>"

    prompt = f"""You are a GUI (Graphical User Interface) Web Agent expert specializing in analyzing interface changes and determining whether a high-level task should be refined based on a series of agent actions and observations.

## High-Level Task Categories

1. **Information Seeking** — User aims to retrieve specific information from the website.
   - Examples:
     - "What is the most expensive product in the 'Electronics' category?"
     - "What are the top 5 posts in the 'Technology' forum?"
     - "Summarize the reviews for the product 'iPhone 11'."

2. **Site Navigation** — User aims to reach a specific page or site state.
   - Examples:
     - "Go to the billing page to check the latest transactions."
     - "Navigate to the 'Contact Us' page and fill out the form to express interest in joining the company."
     - "Find the wiki page of 'the youngest person to receive a Nobel Prize'."

3. **Content Modification** — User aims to change site content or settings.
   - Examples:
     - "Create a user account with username 'bob2134' and password '128nxc18zxv'."
     - "Post a new article titled 'The Future of AI' in the 'Technology' forum."
     - "Create a code repo named 'Agent' and add a README with the text 'This is a code repo for an intelligent agent.'"

---

## Refine Rules

### When to REFINE the task
Refine the task if following situations are observed (cite triggers in "Analysis"):

1. **Invalid or Inconsistent Goal**  
   - The target entity/page/action does not exist, cannot be located, or conflicts with previously observed facts.  
   - Includes site structure changes, unexpected redirects, or missing critical UI elements.

2. **Insufficient Executable Details**  
   - The task is missing essential parameters (e.g., username, product name, date range) and these cannot be inferred from history or current context.  
   - The task wording is too vague to guide concrete next steps.

3. **Stalled or Repetitive Execution**  
   - ≥3 consecutive actions show no meaningful change in page state.  
   - The same error occurs twice (e.g., looped navigation, timeouts, non-interactable elements), suggesting the current task cannot be completed as written.

### When NOT to REFINE
Keep the current high-level task if:
- The goal is valid and consistent with observations.  
- Essential parameters are available or can be inferred.  
- Actions are showing measurable progress toward completion.  
- No persistent or repetitive failure loops are detected.
   
### How to REFINE 
If refinement is required (Your analysis MUST reference the below rules):
1. **Concretize Missing Details** — Add essential parameters from the History of Actions or Current Observation (names, IDs, dates, product info).
2. **Align with Reality** — Remove or replace hallucinated entities/pages with actual ones found on the site.
3. **Downscope the Goal** — If the task is too broad or blocked, adjust it to the next achievable milestone toward the original goal.
4. **Preserve Task Type** — Keep it within the same category (Information Seeking, Site Navigation, Content Modification) unless observations require a category change.
5. The refined high-level task MUST be context-independent — it should not reference the current page, current view, or temporary observations. You MUST avoid wording that ties the task to what is visible "right now" (e.g., "report the number of orders in the current dashboard"). Instead, phrase it as a standalone goal that could be understood and executed without relying on the current observation.
6. The refined task MUST be standalone and self-contained, independent of any current or previous high-level tasks.

## Goal
Ensure the refined task is either already completed within executed actions OR highly likely to be completed within the next 1-2 steps.

---

## Output Requirements

- Format: **JSON dictionary only**, no commentary.
- Fields:
  - `"Analysis"`: Step-by-step reasoning on whether refinement is needed.
  - `"Need-to-Refine"`: `"yes"` or `"no"`.
  - `"High-Level-Task"`: Refined task if `"Need-to-Refine"` is `"yes"`, otherwise empty string.

---

## Information

1. Current High-Level-Task: "{current_high_level_task}"

2. Previous High-Level-Tasks (from oldest to most recent):
<start_previous_high_level_tasks>
{previous_high_level_tasks}
</end_previous_high_level_tasks>

3. History of Actions ({hint_for_history}):
<start_action>
{previous_state_action}
</end_action>

4. Current Page (only current view, not full page):
    - URL: "{curr_url}"

    - Page Context: 
<start_context>
{curr_state_context}
</end_context>

    - Screenshot: "{img_info}"

---

You ONLY need to return a JSON dictionary formatted as follows (no extra commentary):
{{
    "Analysis": "step-by-step reason to analyze whether the given high-level task needs to be refined",
    "Need-to-Refine": "yes or no",
    "High-Level-Task": "provide the refined task if you answer yes, otherwise leave it empty string"
}}
RETURN ONLY THE DICTIONARY I ASKED FOR."""

    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if curr_screenshot is not None:
        message[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": tools_ndarray_to_base64_image(curr_screenshot)},
            },
        )

    return message

def prompt_task_categorization_for_actions(url: str, element_num: int, elements: str, page_context: str, screenshot: "np.ndarray") -> list[dict]:

    prompt = f"""You are a GUI (Graphical User Interface) Web Agent expert capable of grouping interactive elements from a web page into high-level user task categories.

**Information**

Current Page:
    - URL:
{url}

    - Accessibility Tree (only current view, not full page):
{page_context}

    - Elements:
{elements}

    - Screenshot of the page (only current view, not full page):
<image is provided in the attachment>

    
**Your Goal**
1. Your goal is to fully explore the current page state, try to explore different pages and contents of the website.
2. First, you MUST categorize ALL provided {element_num} elements into different categories (list[dict]) based on their natural and purpose.
3. Second, add a "{const_uninteractive_category}" category (list[int]) for elements that cannot be interacted with (i.e., cannot be "CLICK, TYPE, HOVER").
4. For each item in categories except {const_uninteractive_category}, you should decide {{"action": choose from [CLICK, TYPE, HOVER], "element_id": id_of_element (int) to interact, "value": if it is the "TYPE" action, you need to provide the text to type in, else '', "low-level_instruction": concise instruction describing the action}}. The low-level_instruction is a concise instruction that describes the action to be taken, for example:
    - "Click on the 'Add to Cart' button next to the product to add it to your shopping cart."
    - "Type 'OpenAI' into the search bar to find relevant articles."
    - "Scroll down to view the latest blog posts on the homepage."
5. You MUST provide an appropriate and meaningful value for the "value" field if the action is "TYPE", For example:
    - If it's a search box, generate a search query based on realistic user behavior
    - If it's a textbox, generate a plausible text content according to the context

**Output Requirements**
You MUST return a JSON dictionary with the following format (no extra commentary):
{{
    "Analysis": "your analysis of the current page state and elements",
    "Categorization": {{
        "category_1": [{{"action": "xxx", "element_id": "int", "value": "xxx or empty string", "low-level_instruction": "a concise instruction"}}, ...other actions...],
        "category_2": [...],
        "category_3": [...],
        ...different categories...,
        "{const_uninteractive_category}": [element_id_1, element_id_2, ...],
    }}
}}
RETURN ME THE DICTIONARY I ASKED FOR WITHOUT ANY COMMENTARY."""

    message = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
            "url": tools_ndarray_to_base64_image(screenshot),
        }}
        ]}
    ]
    return message

def prompt_osgenesis_generate_high_level_task(
    website_intro: str,
    bounding_box: tuple[int, int, int, int] | None,
    current_action_str: str,
    website_name: str,
    curr_state_screenshot: "np.ndarray",
    new_state_screenshot: "np.ndarray",
) -> list[dict]:
    
    task_examples = const_web_task_examples[website_name]
    task_examples = '\n'.join('- ' + example for example in task_examples)
    
    prompt = f"""You are a GUI (Graphical User Interface) expert capable of analyzing interface changes and envisioning executable tasks or instructions. Given a GUI interface change caused by an action (e.g., clicking or typing) and the corresponding element highlighted in red boxes, you are required to analyze the interface and generate related tasks.

Your task is to envision tasks based on the current action and the resulting changes in the screenshots. The output should include three components:

1. Sub-Instruction: Create a natural language instruction for the current action based on the interface changes it caused. The instruction should be concise, clear, and actionable, incorporating specific details critical to the task, such as elements, file names, timestamps, or other relevant content visible in the screenshots. For example:
   - "Click on the 'Add to Cart' button next to the product to add it to your shopping cart."
   - "Type 'OpenAI' into the search bar to find relevant articles."
   - "Scroll down to view the latest blog posts on the homepage."

2. Analysis: Carefully analyze the before-and-after screenshots step by step, focusing on the changes caused by the action. Then, examine key elements in both screenshots and consider possible operations based on these elements. For example: "The previous screen displayed the main interface of a shopping website, featuring multiple product categories and several showcased items. After clicking the 'Sign Up' button, the interface transitioned to a login page where an email and password can be entered to log into an account. The login page also provides other options, such as recovering a password, creating a new account, or logging in with a Google account."

3. High-Level Instruction: Based on the before-and-after screenshots, the action, and the analysis, generate a high-level task that you believe can be completed within the current interface. There are three types of tasks:
   - Information seeking: The user wants to obtain certain information from the webpage, such as product details, reviews, map information, or route comparisons. Please propose clear and specific questions that need an explicit answer, and avoid asking for summary-type questions, such as "summarize the information about a product."
   - Site navigation: The user wants to navigate to a specific page or state.
   - Content modification: The user wants to modify the content of a webpage or its settings.

The high-level instruction should be creative. You need to deeply analyze the elements and executable actions on the interface to generate realistic, valuable, and executable tasks that can be completed within the current GUI. The instruction should be specific, actionable, and goal-oriented, ensuring the task can be completed on the current GUI by including all critical specifics such as file names, relevant timings, or required details.

Below is a brief description of the current website: {website_intro}

Here are some examples of High-Level Instruction for reference:
{task_examples}

Current Action: {current_action_str}

Website Name: {website_name}

Before-action Screenshot: <image is provided in the first attachment> (the action's target element is highlighted in red box if applicable)

After-action Screenshot: <image is provided in the second attachment>

Please generate tasks that can be completed on the current platform, and avoid tasks that are unrelated to the current website.

You ONLY need to return a JSON dictionary formatted as follows (no extra commentary):
{{
  "Sub-Instruction": "xxx",
  "Analysis": "xxx",
  "High-Level-Instruction": "xxx"
}}
RETURN ONLY THE DICTIONARY I ASKED FOR.
"""


    redbox_screenshot = tools_draw_red_bbox(curr_state_screenshot, bounding_box)
    message = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
            "url": tools_ndarray_to_base64_image(redbox_screenshot),
            }},
            {"type": "image_url", "image_url": {
            "url": tools_ndarray_to_base64_image(new_state_screenshot),
            }}
        ]}
    ]

    return message

def prompt_refine_trajectory(trajectory: str) -> list[dict]:
    prompt = f"""You are a GUI (Graphical User Interface) Web Agent expert. Your job: analyze a high-level task and its trajectory (sequence of states and actions), assign a quality score, and choose one of:
- "keep": keep the trajectory as-is (already minimal, ordered, and ends with a correct NONE action with a non-empty value).
- "refine": reorder or delete steps to make the trajectory succeed (final step must be NONE with a non-empty explanation).
- "drop": discard the trajectory entirely (e.g., irreparable, hallucination, impossible, unsafe, or missing critical information).

A trajectory is structured as:
"Length of trajectory, High-level task, summary of state1, action1, summary of state2, action2, ..."

---

### Available Low-Level Actions (exact JSON formats)

- CLICK:
{{"type": "CLICK", "element_id": <int>, "value": ""}}

- TYPE (default presses Enter unless `press_enter_after=0`):
{{"type": "TYPE", "element_id": <int>, "value": "text <string>"}}

- HOVER:
{{"type": "HOVER", "element_id": <int>, "value": ""}}

- PRESS (keyboard shortcut, e.g., Ctrl+V / Cmd+V):
{{"type": "PRESS", "element_id": "", "value": "key_comb <string>"}}

- SCROLL:
{{"type": "SCROLL", "element_id": "", "value": "up" or "down"}}

- GOTO (navigate directly to a URL):
{{"type": "GOTO", "element_id": "", "value": "url <string>"}}

- GO_BACK:
{{"type": "GO_BACK", "element_id": "", "value": ""}}

- GO_FORWARD:
{{"type": "GO_FORWARD", "element_id": "", "value": ""}}

- NONE (successful finalization; use when the task is complete or no further action is needed):
{{"type": "NONE", "element_id": "", "value": "<non-empty explanation of task completion>"}}

- STOP (only when the high-level task is impossible, inconsistent with observations, or hallucinatory):
{{"type": "STOP", "element_id": "", "value": "reason <string>"}}

---

### Scoring rubric (0-100)
Score the original trajectory on:
1) Goal alignment (0-25): steps relevant to the high-level task.
2) Logical order (0-25): steps follow a sensible sequence.
3) Efficiency (0-25): avoids redundancy or unnecessary actions.
4) Success likelihood (0-25): likely to end successfully with NONE (non-empty value).

**Important**: The score is advisory only. You may still decide to keep, refine, or drop regardless of the numeric score.

---

### Decision policy
- Use your judgment to decide between "keep", "refine", and "drop".
- Always ensure that if a trajectory is **kept** or **refined**, it ends with a **NONE** action whose value is a **non-empty explanation**.
- If refining, reorder/delete steps without inventing new ones. Remove STOP if success is possible and replace with a final NONE.
- If dropping, do not fabricate a NONE. Provide a clear drop_reason.

**Indexing & Deletion**
- Let the trajectory contain K (Observation, Action) pairs. Index them 0..K-1 in the original order.
- Reordering = return indices in a new order.
- Deletion = omit indices.
- No duplicates; all indices must be within 0..K-1. Do not invent new steps.

---

### Input
{trajectory}

---

### Output Requirement (STRICT)
Return **only** one JSON object (no extra text, no code fences):

{{
"task": "<exact high-level task string>",
"score": <int>,                        // 0..100, advisory only
"decision": "keep" | "refine" | "drop",

// For decision in {"keep","refine"}:
"order": [<int>, ...],                 // 0-based indices in final desired order
"modify_end": <true|false>,            // true if final action changed into NONE or its value updated
"append_end": <true|false>,            // true if a NONE action had to be appended
"final_none_value": "<non-empty explanation to place in final NONE.value>",

// For decision == "drop":
"drop_reason": "<brief reason why trajectory should be dropped>",

// Always present:
"modification_reason": "<brief rationale for keep/refine/drop>"
}}

---

### Additional constraints
- "task" must exactly match the high-level task from input.
- If **decision = keep**: "order" must be [0,1,...,K-1], and the existing final step must already be NONE with a non-empty value.
- If **decision = refine**: provide a valid "order" (length ≤ K) that ends with a NONE and non-empty value. Use modify_end/append_end accordingly.
- If **decision = drop**: set "order" = [], leave "final_none_value" empty, and give a clear "drop_reason".
"""

    message = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    return message