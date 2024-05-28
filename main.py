from openai import OpenAI, AzureOpenAI
import os

from dotenv import load_dotenv

from file_processor import FileProcessor
from single_agent import Single_Agent
from multi_agent import Multi_Agent

load_dotenv()
DEBUG = (os.getenv('DEBUG', 'True') == 'True')

class ChatWithOpenai:
    def __init__(
        self,
        system_message,
        model,
        temperature,
        max_tokens,
        client,
        default_user_messages=None,
    ):
        self.system_message = system_message
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        azure_connector_surf = AzureConnectorSurf()
        self.client = azure_connector_surf.client
        self.messages = [{"role": "system", "content": system_message}]
        if default_user_messages:
            for user_message in default_user_messages:
                self.messages += [{"role": "user", "content": user_message}]

    def chat(self, user_input):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages + user_input,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class AzureConnectorSurf:
    def __init__(self) -> None:
        self.api_endpoint = os.getenv("azure_api_endpoint")
        self.api_version = os.getenv("azure_api_version")
        self.api_key = os.getenv("azure_api_key")
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_endpoint,
        )
        self.tts_api_version = os.getenv("tts_api_version")
        self.tts_model1 = os.getenv("tts_model1")
        self.GPT_MODEL = os.getenv("azure_GPT_MODEL_3")
        self.whisper_model = os.getenv("azure_whisper_model")
        self.voice_name = os.getenv("voice_name")
        self.tts_model = os.getenv("tts_model2")


class OpenaiConnectorSurf:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("openai_api_key"))
        self.GPT_MODEL = os.getenv("openai_GPT_MODEL_3")
        self.whisper_model = os.getenv("whisper_model_openai")


def check_relevance(connector_surf, prompt: str):
    messages = [
        {
            "role": "system",
            "content": "Classify if the following prompt is relevant or irrelevant. Guidelines for you as a Trading Assistant:Relevance: Focus exclusively on queries related to trading and financial markets(including stock tickers). If a question falls outside this scope, politely inform the user that the question is beyond the service's focus.Accuracy: Ensure that the information provided is accurate and up-to-date. Use reliable financial data and current market analysis to inform your responses.Clarity: Deliver answers in a clear, concise, and understandable manner. Avoid jargon unless the user demonstrates familiarity with financial terms.Promptness: Aim to provide responses quickly to facilitate timely decision-making for users.Confidentiality: Do not ask for or handle personal investment details or sensitive financial information.Compliance: Adhere to legal and ethical standards applicable to financial advice and information dissemination.Again, focus solely on topics related to trading and financial markets. Politely notify the user if a question is outside this specific area of expertise.",
        },
        {"role": "user", "content": prompt},
    ]
    response = connector_surf.client.chat.completions.create(
        model=connector_surf.GPT_MODEL, temperature=0, messages=messages
    )
    relevance_check = response.choices[0].message.content
    if "irrelevant" in relevance_check.lower():
        return True
    return False


# class llm_surf:
def llm_surf(llm_input: dict) -> str:

    llm_output = {
        "response": "",
        "symbol": llm_input.get("symbol"),
        "file": None,
        "function_call": None,
    }

    azure_connector_surf = AzureConnectorSurf()

    content = ""
    fileProcessor = FileProcessor(azure_connector_surf)
    if "file" in llm_input and llm_input["file"]:
        if type(llm_input["file"]) == str:
            llm_input["file"] = open(llm_input["file"], "r")
        file_content = fileProcessor.get_file_content(llm_input["file"])
        if file_content:
            content += file_content + "\n"

    if llm_input.get("new_message"):
        content += llm_input.get("new_message") + "\n"
    else:
        content += fileProcessor.default_prompt + "\n"

    # running in multi-agent mode
    if os.getenv("MODE") == "multi-agent":
        MA = Multi_Agent(
            ChatWithOpenai=ChatWithOpenai, client=azure_connector_surf.client
        )
        graph = MA.initialize_graph()
        llm_output = MA.generate_multi_agent_answer(llm_input, graph)
    # running in single-agent mode
    elif os.getenv("MODE") == "single-agent":
        if llm_input.get("new_message") and check_relevance(
            azure_connector_surf, llm_input.get("new_message")
        ):
            llm_output["response"] = (
                "I'm here to help with trading and financial market queries. If you think your ask relates to trading and isn't addressed, please report a bug using the bottom right panel."
            )
            return llm_output
        SA = Single_Agent(azure_connector_surf.client)
        results_string, results_json, function_name = SA.generate_answer(
            llm_input=llm_input, content=content
        )
        llm_output["response"] = results_string
        llm_output["chart_info"] = results_json
        llm_output["function_call"] = function_name

    if not DEBUG:
        llm_output["file"] = fileProcessor.text_to_speech(llm_output["response"])

    return llm_output
