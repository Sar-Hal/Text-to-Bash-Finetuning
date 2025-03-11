import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Hugging Face token from environment variable
HUGGINGFACETOKEN = os.environ.get("HUGGINGFACETOKEN")

def format_prompt(nl, bash=None):
    prompt = f"### Instruction: Convert the following English description to a Bash command:\n\n{nl}\n\n### Response:"
    if bash:
        return prompt + f" {bash}"
    return prompt

def generate_bash_command(model, tokenizer, nl_text, max_length=100):
    """Generate a bash command from natural language"""
    prompt = format_prompt(nl_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=0.1,
            top_p=0.75,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bash_command = response.split("### Response:")[-1].strip()
    return bash_command

def main():
    model_path = "/kaggle/working/mistral-nl2bash/final_model"
    model = AutoModelForCausalLM.from_pretrained(model_path,  trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    #Example prompt , replace with user input later on
    nl_text = "List all files in the current directory"
    try:
        bash_command = generate_bash_command(model, tokenizer, nl_text)
        print(f"Natural Language: {nl_text}")
        print(f"Generated Bash Command: {bash_command}")
    except Exception as e:
        print(f"Error generating command: {e}")

if __name__ == "__main__":
    main()
