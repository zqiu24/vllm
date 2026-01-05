from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


if __name__=='__main__':

    llm = LLM(model="/lustre/fast/fast/zqiu/hf_models/Qwen2.5-3B-Instruct", enable_lora=True, enforce_eager=True)

    sql_lora_path = snapshot_download(repo_id="zqiu/Qwen2.5-3B-Instruct-LoRA-Test")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    prompts = [
        "[user] What is the capital city of France? [/user] [assistant]",
        "[user] How many people live in China? [/user] [assistant]",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    )

    print('First output:')
    print(outputs[0].outputs[0].text)
    print('-'*100)
    print('Second output:')
    print(outputs[1].outputs[0].text)
    print('-'*100)
    exit()