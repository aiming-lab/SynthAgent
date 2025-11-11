
import collections
import html
import time
import urllib
import json


from beartype import beartype
from nltk.tokenize import word_tokenize
from playwright.sync_api import Page
from typing import Any
from loguru import logger
from syn.gpt import GPTClient
from syn.tools import tools_robust_json_loads


from urllib.parse import urlparse

import requests

from browser_env.env_config import (
    ACCOUNTS,
    GITLAB,
    MAP,
    REDDIT,
    SHOPPING,
    SHOPPING_ADMIN,
    WIKIPEDIA,
)

def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{SHOPPING}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(
        f"{SHOPPING}/rest/V1/orders", params=params, headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{SHOPPING}/sales/order/view/order_id/{order_id}/"
    return order_url


def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


def gitlab_get_project_memeber_role(page: Page, account_name: str) -> str:
    # get the account index
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;  // Default value if not found

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )

        # get the role
        role: str = page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        )
    except Exception:
        role = ""

    return role




class Evaluator(object):
    def __init__(self, eval_config: dict, task: str, last_action_summary: str, page: Page, gpt_client: GPTClient, gpt_eval_config: dict | None = None):
        self.eval_config = eval_config
        self.task = task
        self.last_action_summary = last_action_summary
        self.page = page
        self.gpt_client = gpt_client
        if gpt_eval_config is None:
            self.gpt_eval_config = {
                "model": "gpt-4.1",
                "temperature": 0.0,
                "max_completion_tokens": 768,
            }
        else:
            self.gpt_eval_config = gpt_eval_config

    @beartype
    def __call__(
        self,
    ) -> float:
        raise NotImplementedError

    def llm_fuzzy_match(self, pred: str, reference: str, question: str) -> float:
        """Check whether the prediction matches the reference with LLM"""

        message = f"""You are grading a student's answer for semantic equivalence.

Task:
Decide if the student's answer is (a) semantically equivalent to the reference answer, or (b) fully contains the reference answer with no contradictions. Ignore wording, phrasing, order, and surface-level differences. Focus only on meaning.

Special rule for "N/A":
- The string "N/A" means "not achievable".
- If and only if the reference answer is "N/A", then a correct student answer can be either "N/A" or a clear, valid reason why the task is not achievable.
- If the reference is not "N/A", then a student answer of "N/A" (or excuses) is incorrect.

Judgment labels (choose exactly one):
- correct
- partially correct
- incorrect

Guidelines:
- correct: Student's answer matches the reference meaning or fully contains all essential information from the reference without introducing contradictions.
- partially correct: Student's answer captures some but not all essential information, or mixes correct info with omissions/uncertainties (no hard contradiction with the reference).
- incorrect: Student's answer conflicts with the reference, misses essential information entirely, or relies on excuses when the reference is not "N/A".

Edge cases:
- Minor numeric rounding (e.g., 3.14 vs 3.1415) → acceptable.
- Additional, non-contradictory details → allowed.
- Any contradiction of key facts in the reference → incorrect.

Now grade the student's answer for this item:

question={question}
reference answer={reference}
student answer={pred}

Respond in the following json format:
{{
    "analysis": "step-by-step analysis of the grading process, explaining why you make the judgment",
    "judgment": "exactly one of 'correct', 'partially correct', or 'incorrect'",
}}
RETURN ONLY THE DICTIONARY I ASKED FOR WITHOUT ANY COMMENTARY."""
        
        messages = [
            {"role": "user", "content": message},
        ]

        response = self.gpt_client.request(
            messages=messages,
            json_mode=True,
            **self.gpt_eval_config,
        )
        response_text = response.message.content.lower()
        try:
            response_data = tools_robust_json_loads(response_text)
        except Exception as e:
            logger.error(f"Failed to parse LLM fuzzy match response as json: {response_text}, error: {e}")
            response_data = {}

        if 'judgment' in response_data:
            response = response_data['judgment']
        else:
            response = response_text

        if "incorrect" in response:
            score =  0.0
        elif 'partially correct' in response or 'correct' in response:
            score = 1.0
        else:
            logger.error(f"Unexpected response from LLM fuzzy match cannot find 'correct', 'incorrect', or 'partially correct'. in {response}, return 0.0")

            score =  0.0

        logger.debug(f"autoeval, LLM fuzzy match score={score}\nquestion={question}\npred={pred}\nreference={reference}\nresponse=\n{response_text}")
        
        return score


    def llm_ua_match(self, pred: str, reference: str, question: str) -> float:
        """Check whether the prediction matches the reference with LLM"""
        messages: list[dict[str, Any]] = []
        # construct the question to ask
        message = ""
        message += f"task: {question}\n"
        message += f"actual unachievable reason: {reference}\n"
        message += f"reported unachievable reason: {pred}\n"
        message += (
            "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
            "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
            "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
            "Determine if the reported reason aligns with the actual reason, even if implicitly. "
            "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": message},
        ]

        response = self.gpt_client.request(
            messages=messages,
            json_mode=False,
            **self.gpt_eval_config,
        )
        response = response.message.content.lower()

        if "different" in response:
            return 0.0
        elif 'same' in response:
            return 1.0        
        else:
            logger.error(f"Unexpected response from LLM UA match cannot find 'same' or 'different'. in {response}, return 0.0")
            return 0.0



class StringEvaluator(Evaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    @staticmethod
    @beartype
    def clean_answer(answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    @beartype
    def exact_match(self, ref: str, pred: str, question: str | None = None) -> float:

        """if question is provided, it will be used to fall back to fuzzy match"""

        if StringEvaluator.clean_answer(pred) == StringEvaluator.clean_answer(ref):
            return 1.0
        elif question is not None:
            # fall back to fuzzy match
            return self.llm_fuzzy_match(reference=ref, pred=pred, question=question)
        else:
            return 0.0


    @staticmethod
    @beartype
    def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0)
        if (
            tokenize
            and len(clean_ref) == 1
            and len(word_tokenize(clean_ref)) == 1
        ):
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)



    def __call__(
        self,
    ) -> float:

        pred = self.clean_answer(self.last_action_summary)
        intent = self.task
        question = self.task

        score = 1.0
        for approach, value in self.eval_config["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred, question=question)

                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                case "fuzzy_match":
                    
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        score *= self.exact_match(ref=value, pred=pred, question=question)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if score != 1:
                            score = 1.0 * self.llm_ua_match(
                                pred=pred,
                                reference=self.eval_config["string_note"],
                                question=self.task,
                            )
                    else:
                        assert isinstance(value, list)
                        for reference in value:
                            score *= self.llm_fuzzy_match(
                                reference=reference, pred=pred, question=intent
                            )
        return score


class URLEvaluator(Evaluator):
    """Check URL matching"""

    @beartype
    def __call__(
        self,
    ) -> float:

        def clean_url(url: str) -> str:
            url = str(url)
            url = url.rstrip("/")
            return url

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            """Parse a URL into its base, path, and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = parsed_url.netloc + parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            return base_path, query

        def parse_urls(
            urls: list[str],
        ) -> tuple[list[str], dict[str, set[str]]]:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries

        pred = clean_url(self.page.url)
        ref_urls = self.eval_config["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        matching_rule = self.eval_config.get("url_note", "GOLD in PRED")
        if matching_rule == "GOLD in PRED":
            ref_base_paths, ref_queries = parse_urls(ref_urls)
            pred_base_paths, pred_query = parse_url(pred)

            base_score = float(
                any(
                    [
                        ref_base_path in pred_base_paths
                        for ref_base_path in ref_base_paths
                    ]
                )
            )
            query_score = 1.0
            for k, possible_values in ref_queries.items():
                query_score *= float(
                    any(
                        possible_ref_value in pred_query.get(k, [])
                        for possible_ref_value in possible_values
                    )
                )
            score = base_score * query_score

        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")

        return score


class HTMLContentEvaluator(Evaluator):
    """Check whether the contents appear in the page"""

    @beartype
    def __call__(
        self,
    ) -> float:

        targets = self.eval_config["program_html"]

        score = 1.0
        for target in targets:
            target_url: str = target["url"]  # which url to check
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", self.page.url)
                target_url = eval(func)

            locator: str = target["locator"]  # js element locator

            # navigate to that url
            if target_url != "last":
                self.page.goto(target_url)
                time.sleep(3)  # TODO [shuyanzh]: fix this hard-coded sleep # so there might be a hard bug for goto???

            # empty, use the full page
            if not locator.strip():
                selected_element = self.page.content()
            # use JS to select the element
            elif locator.startswith("document.") or locator.startswith(
                "[...document."
            ):
                if "prep_actions" in target:
                    try:
                        for prep_action in target["prep_actions"]:
                            self.page.evaluate(f"() => {prep_action}")
                    except Exception:
                        pass
                try:
                    selected_element = str(self.page.evaluate(f"() => {locator}"))
                    if not selected_element:
                        selected_element = ""
                except Exception:
                    # the page is wrong, return empty
                    selected_element = ""
            # run program to call API
            elif locator.startswith("func:"):  # a helper function
                page = self.page
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                selected_element = eval(func)
            else:
                raise ValueError(f"Unknown locator: {locator}")

            selected_element = html.unescape(selected_element)

            if "exact_match" in target["required_contents"]:
                temp_string_evaluator = StringEvaluator(
                    eval_config=self.eval_config,
                    task=self.task,
                    last_action_summary=self.last_action_summary,
                    page=self.page,
                    gpt_client=self.gpt_client,
                    gpt_eval_config=self.gpt_eval_config
                )
                required_contents = target["required_contents"]["exact_match"]
                cur_score = temp_string_evaluator.exact_match(
                    ref=required_contents, pred=selected_element
                )
                score *= float(cur_score)
                # print(f"[exact match] {cur_score}, selected element: {selected_element}, required contents: {required_contents}")
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    content_or = content.split(" |OR| ")
                    cur_score = any(
                        [
                            StringEvaluator.must_include(
                                ref=content,
                                pred=selected_element,
                                tokenize=False,
                            )
                            for content in content_or
                        ]
                    )
                    score *= float(cur_score)
                    # print(f"[must include] {cur_score}, selected element: {selected_element}, required contents: {content_or}")
            else:
                raise ValueError(
                    f"Unknown required_contents: {target['required_contents'].keys()}"
                )
        return score


class EvaluatorComb:
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators

    @beartype
    def __call__(
        self,
    ) -> int:
        score = 1.0
        for evaluator in self.evaluators:
            cur_score = evaluator()
            score *= cur_score
        return int(score)


@beartype
def evaluator_router(eval_config: dict, task: str, last_action_summary: str, page: Page, gpt_client: GPTClient, gpt_request_config: dict | None = None) -> EvaluatorComb:
    """Router to get the evaluator class"""
    eval_types = eval_config['eval_types']
    params = {
        'eval_config': eval_config,
        'task': task,
        'last_action_summary': last_action_summary,
        'page': page,
        'gpt_client': gpt_client,
        "gpt_eval_config": gpt_request_config,

    }
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator(**params))
            case "url_match":
                evaluators.append(URLEvaluator(**params))
            case "program_html":
                evaluators.append(HTMLContentEvaluator(**params))
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)
