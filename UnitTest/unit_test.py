import os
import json
import pytest
from tqdm import tqdm
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from main import llm_surf
import input_filter

os.environ["OPENAI_API_KEY"] = ""


def test_case(llm_input, llm_output, supposed_output, threshold):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=threshold)
    test_case = LLMTestCase(
        input=llm_input,
        actual_output=llm_output,
        retrieval_context=[supposed_output]
    )
    assert_test(test_case, [answer_relevancy_metric])

def unit_test():
    tests = ["detect_trend", "calculate_sr", "calculate_sl", "calculate_tp", "get_bias", "introduction", "long_chat_scenario"]

    input_json = input_filter.front_end_json_sample

    # for each function
    for test in tests:
        print(f"Test:   {test}")
        # loading the prompts and their results
        file = open(f"UnitTest/test_results/{test}_test_results.json")
        data = json.load(file)
        input_json["history_message"] = []
        # testing each prompt for the corresponding function
        for prompt, result in tqdm(data.items()):
            input_json["new_message"] = prompt
            output_json = llm_surf(input_json)
            generated = output_json["response"]
            # running deepeval for each prompt
            try:
                test_case(prompt, generated, result, 0.5)
            except:
                print(f"prompt {prompt} did not executed!")
                f = open("problemed_prompts.txt", "a")
                f.write(f"{prompt}\n")
                f.close()
