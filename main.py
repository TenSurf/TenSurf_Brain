import function_calling.function_calling as function_calling
import file_processor.file_processor as file_processor
import config
from openai import OpenAI, AzureOpenAI

# llm_input_json_keys = ["symbol", "start_datetime", "end_datetime", "timeframe", "timezone", "user_id", "history_message", "new_message", "file"]

# llm_output_json_keys = ["response", "chart_info", "file_path", "function_call"]

class AzureConnectorSurf:
    def __init__(self) -> None:
        self.api_endpoint = config.azure_api_endpoint
        self.api_version = config.azure_api_version
        self.api_key = config.azure_api_key
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_endpoint,
        )
        self.tts_api_version = config.tts_api_version
        self.tts_model1 = config.tts_model1
        self.GPT_MODEL = config.azure_GPT_MODEL_3
        self.whisper_model = config.azure_whisper_model
        self.voice_name = config.voice_name
        self.tts_model = config.tts_model2

class OpenaiConnectorSurf:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=config.openai_api_key)
        self.GPT_MODEL = config.openai_GPT_MODEL_3
        self.whisper_model = config.whisper_model_openai

def llm_surf(llm_input_json_keys: dict) -> str:
    symbol = llm_input_json_keys["symbol"]
    start_datetime = llm_input_json_keys["start_datetime"]
    end_datetime = llm_input_json_keys["end_datetime"]
    timeframe = llm_input_json_keys["timeframe"]
    timezone = llm_input_json_keys["timezone"]
    user_id = llm_input_json_keys["user_id"]
    history_message = llm_input_json_keys["history_message"]
    new_message = llm_input_json_keys["new_message"]
    file = llm_input_json_keys["file"]  # as type string or file binary

    azure_connector_surf = AzureConnectorSurf()
    content = ""
    if llm_input_json_keys["file"]:
        fileProcessor = file_processor.FileProcessor(azure_connector_surf)
        file_content = fileProcessor.get_file_content(file)
        if file_content:
            content += file_content + "\n"

    if new_message:
        content += new_message + "\n"
    else:
        content += fileProcessor.default_prompt + "\n"

    functionCalling = function_calling.FunctionCalling()
    return functionCalling.generate_answer(llm_input=llm_input_json_keys)
