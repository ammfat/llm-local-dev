import time
import torch

from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()


# Set-up the model
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "nickypro/tinyllama-110M"

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model_pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16,
    device=device
)


# Call the model

start_time = time.time()

result = model_pipe(
    "Toktok, who is there?", 
    max_new_tokens=128,
    temperature=0.9,
    top_p=0.95,
    top_k=40,
)

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
print(result)
