from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.oft.request import OFTRequest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if __name__=='__main__':

    # model_id = "/lustre/fast/fast/zqiu/hf_models/Qwen2.5-3B-Instruct"
    # adapter_id = "zqiu/Qwen2.5-3B-Instruct-OFT-Test"

    # # 1. Load Base Model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     torch_dtype="auto",
    #     trust_remote_code=True
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    # # 2. Load Adapter (OFT) using PeftModel
    # # Note: Ensure your installed peft version supports OFT (it's a relatively new feature in PEFT)
    # model = PeftModel.from_pretrained(model, adapter_id)

    # # 3. Verify
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    llm = LLM(
        model="/lustre/fast/fast/zqiu/hf_models/Qwen2.5-3B-Instruct", 
        enable_oft=True, 
        enforce_eager=True,
        max_oft_block_size=32,  # <--- Essential!
    )
    
    sql_oft_path = snapshot_download(repo_id="zqiu/Qwen2.5-3B-Instruct-OFT-Test")

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
        oft_request=OFTRequest("sql_adapter", 1, sql_oft_path),
    )

    print('First output:')
    print(outputs[0].outputs[0].text)
    print('-'*100)
    print('Second output:')
    print(outputs[1].outputs[0].text)
    print('-'*100)
    exit()