import json
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from human_eval.data import read_problems
from openai import OpenAI

# pylint: disable=redefined-outer-name, line-too-long, missing-module-docstring, invalid-name, import-outside-toplevel
# ruff: noqa: E501

load_dotenv("keys.env")
OPENAI_KEY = os.getenv("OPENAI_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_KEY")
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")


def clear_output_interactive():
    """Clear output in an interactive environment."""
    from IPython.display import clear_output

    clear_output(wait=True)


def clear_output_non_interactive():
    """Clear output in a non-interactive environment."""
    if sys.platform == "win32":  # For Windows
        os.system("cls")
    else:  # For macOS and Linux
        os.system("clear")


def clear_output_robust():
    """Clear output irrespective of OS, and whether running interactively or non-interactively."""
    if "ipykernel" in sys.modules:
        clear_output_interactive()
    else:
        clear_output_non_interactive()


def sanitize_answer(raw_answer):
    """Sanitize the answer to remove unwanted parts of the code, like comments."""
    code_started = True
    answer = []
    for line in raw_answer.split("\n"):
        # valid start of line that we want to include
        if line.lstrip(" ").startswith("def") or line.lstrip(" ").startswith("import"):
            code_started = True
            answer.append(line)
        # marks start of code, but we don't want to include it
        elif line.startswith("```python"):
            code_started = True
        # don't include anything after the code block
        elif line.startswith("```"):
            break
        elif not line.startswith(" ") and not line.startswith("\t") and line != "" and line != "\n":
            code_started = False
        elif code_started:
            answer.append(line)
    return "\n".join(answer)


def parse_completion_stream(completion_stream, task_id, end_after_n_codeblocks=None, framework="ai"):
    """Parse a completion stream and return the response."""
    response = []
    finished = False
    while not finished:
        try:
            text = next(completion_stream).choices[0].delta.content if framework == "ai" else next(completion_stream)
            if text:
                response.append(text)
                clear_output_robust()
                if task_id:
                    print(f"{task_id}\n")
                print(prompt, flush=True)
                print("".join(response), flush=True)
                if end_after_n_codeblocks:
                    num_code_blocks = 0
                    for line in "".join(response).splitlines():
                        if line.lstrip(" ").startswith("```"):
                            num_code_blocks += 1
                            if num_code_blocks == end_after_n_codeblocks:
                                finished = True
        except StopIteration:
            finished = True
    return "".join(response)


def hf(prompt, model, temperature=0.8, task_id=None, end_after_n_codeblocks=None):
    client = InferenceClient(model=model, token=HUGGINGFACE_KEY)
    completion_stream = client.text_generation(prompt=prompt, stream=True, max_new_tokens=1_000, temperature=temperature)
    return parse_completion_stream(completion_stream, task_id, end_after_n_codeblocks, framework="hf")


def ai(prompt, system=None, url="http://127.0.0.1:8080/v1", model="llama!", key="na", temperature=0.8, presence_penalty=None, task_id=None):
    client = OpenAI(base_url=url, api_key=key)
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    kwargs = {
        "temperature": temperature,
    }
    if presence_penalty:
        kwargs["presence_penalty"] = presence_penalty
    completion_stream = client.chat.completions.create(model=model, messages=messages, stream=True, max_tokens=1_000, **kwargs)
    return parse_completion_stream(completion_stream, task_id)


problems = read_problems()
keys = list(problems.keys())
subset = {key: problems[key] for key in keys}
for task_id in subset:
    raw_prompt = problems[task_id]["prompt"]
    # deepseek (131/164=0.799)
    # running deepseek-coder-33b-instruct.Q4_K_M.gguf in a local llama.cpp server with ./build/bin/server -ngl 63 -m ./models/deepseek-coder-33b-instruct.Q4_K_M.gguf -c 2048
    # temperature = 0.0
    # model = "deepseek"
    # system = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function. After the function, include an EOS token.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt, system=system, temperature=temperature, task_id=task_id)

    # gpt4 (104/164=0.634) instruction-style
    # url = "https://api.openai.com/v1/"
    # model = "gpt-4"
    # key = OPENAI_KEY
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "### Instruction: Please continue to complete the function. After the function, include an EOS token.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,url=url,model=model,key=key,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # gpt4 (138/164=0.841) completion-style
    # url = "https://api.openai.com/v1/"
    # model = "gpt-4"
    # key = OPENAI_KEY
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "Please continue to complete the function.\n```python\n"
    # postamble = ""
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,url=url,model=model,key=key,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # mixtral8x7b (75/164=0.457)
    # mixtral does better in a completion-style prompt than instruction-style (by 2 questions, +1.2%)
    # running mixtral-8x7b-v0.1.Q5_K_M.gguf in a local llama.cpp server with ./build/bin/server -ngl 20 -m ./models/mixtral-8x7b-v0.1.Q5_K_M.gguf -c 2048
    # model = "mixtral"
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "```python\n"
    # postamble = ""
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # mistral-medium (102/164=0.622)
    # url = "https://api.mistral.ai/v1/"
    # model = "mistral-medium"
    # key = MISTRAL_KEY
    # temperature = 0.2
    # presence_penalty = 0
    # preamble = "Please continue to complete the function.\n```python\n"
    # postamble = ""
    # prompt = preamble + raw_prompt + postamble
    # raw_answer = ai(prompt=prompt,url=url,model=model,key=key,temperature=temperature,presence_penalty=presence_penalty,task_id=task_id)

    # llama2 (69/164=0.421) nice
    # temperature = 0.2
    # model = "meta-llama/Llama-2-70b-chat-hf"  # awful <30%
    # model = "codellama/CodeLlama-34b-Instruct-hf"  # slightly less awful
    # system = "### System Prompt\nYou are an intelligent programming assistant.\n"
    # system += "\n### Instruction:\n{prompt}\n### Response:\n"
    # preamble = "### Instruction: Please continue to complete the function.\n```python\n"
    # postamble = "```\n\n### Response:\n"
    # prompt = system + preamble + raw_prompt + postamble
    # raw_answer = hf(prompt=prompt,model=model,temperature=temperature,task_id=task_id,end_after_n_codeblocks=2)

    # mistral-large (120/164=0.732)
    url = "https://api.mistral.ai/v1/"
    model = "mistral-large-2402"
    key = MISTRAL_KEY
    temperature = 0.2
    presence_penalty = 0
    preamble = "Please continue to complete the function.\n```python\n"
    postamble = ""
    prompt = preamble + raw_prompt + postamble
    raw_answer = ai(prompt=prompt,url=url,model=model,key=key,temperature=temperature,task_id=task_id)

    # sanitize answer, and append it to the jsonl file
    with open(f"{model.split('/', maxsplit=1)[0]}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(task_id=task_id, completion=sanitize_answer(raw_answer))))
        f.write("\n")
    