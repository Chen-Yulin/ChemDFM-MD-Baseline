import torch
from transformers import AutoTokenizer,LlamaTokenizer, LlamaForCausalLM, GenerationConfig
model_name_or_id = "OpenDFM/ChemDFM-v1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_id, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")

input_text = "Can you please give detailed descriptions of the molecule below?\nCl.O=C1c2c(O)cccc2-c2nn(CCNCCO)c3ccc(NCCNCCO)c1c23"
input_text = f"[Round 0]\nHuman: {input_text}\nAssistant:"

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
generation_config = GenerationConfig(
    do_sample=True,
    top_k=20,
    top_p=0.9,
    temperature=0.9,
    max_new_tokens=1024,
    repetition_penalty=1.05,
    eos_token_id=tokenizer.eos_token_id
)

outputs = model.generate(**inputs, generation_config=generation_config)
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
print(generated_text.strip())
