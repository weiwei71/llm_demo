import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set MPS (Metal GPU) as the default device
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_default_device(device)

# Load a small model for testing (change to a larger one later)
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("💬 Enter your prompt (type 'exit' to quit):")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("👋 Exiting...")
        break

    inputs= tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=200)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n🤖 AI: {response}")

