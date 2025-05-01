"""
config.py

Application configuration data
"""
import prompts

"""
For each model use those parameters to make possible to use Llama
from llama_cpp to generate the model, see the example
MODEL_HUB = 
MODEL_NAME =
MODEL_FILE =

Examples:

model_path = Path(MODEL_HUB, MODEL_NAME, MODEL_FILE)

llm = LLama(
	model=model_path,
	...
)
"""

QA_AGENT = (
    "C:/Users/efthimiou/AppData/Roaming/Jan/data/models/huggingface.co/TheBloke/",	# model_hub 
	"zephyr-7B-beta-GGUF",							# model_name 
    "zephyr-7b-beta.Q2_K.gguf",						# model_file
)

QA_CRITIQUE = {
    "groundedness": prompts.question_groundedness_critique,
    "relevance": prompts.question_relevance_critique,
    "standalone": prompts.question_standalone_critique,
}

QA_LLM_NAME = "Mixtral-8x7B-v0.1"

QA_LLM_MODEL = {
    "Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1"
}

CRITIQUE_LLM_NAME = "Mixtral-8x7B-v0.1"

CRITIQUE_LLM_MODEL = {
    "Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1"
}

JUDGE_LLM_NAME = "GPT4"
QA_CRITIQUE = {
    "groundedness": prompts.question_groundedness_critique,
    "relevance": prompts.question_relevance_critique,
    "standalone": prompts.question_standalone_critique,
}

QA_LLM_NAME = "Mixtral-8x7B-v0.1"

QA_LLM_MODEL = {
    "Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1"
}

CRITIQUE_LLM_NAME = "Mixtral-8x7B-v0.1"

CRITIQUE_LLM_MODEL = {
    "Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1"
}

JUDGE_LLM_NAME = "GPT4"

JUDGE_LLM_MODEL = {
    "GPT4": "gpt-4-1106-preview",
}


READER_LLM_NAME = "zephyr-7b-beta"

READER_LLM_MODEL = {
    "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
}