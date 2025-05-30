import time
import torch

from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()


def get_model_pipe(model_name: str) -> pipeline:
    """Get the model pipeline

    Args:
        model_name (str): The name of the model to get the pipeline for

    Returns:
        pipeline: The model pipeline
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device=device
    )


def inference(
    model_pipe: pipeline,
    prompt: str, 
    max_new_tokens: int = 128,
    **kwargs
) -> dict:
    """Run inference on the model

    Args:
        model_pipe (pipeline): The model pipeline
        prompt (str): The prompt to run inference on
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 128.
        **kwargs: Additional keyword arguments to pass to the model pipeline

    Returns:
        dict: The result of the inference
            - outputs: The outputs of the model
            - time_taken: The time taken to run the inference
    """
    start_time = time.time()

    try:
        outputs = model_pipe(prompt, max_new_tokens=max_new_tokens, **kwargs)
    except KeyboardInterrupt:
        outputs = None

    end_time = time.time()
    
    return {
        "outputs": outputs,
        "time_taken": end_time - start_time,
    }


if __name__ == "__main__":
    model_name = "nickypro/tinyllama-110M"
    model_pipe = get_model_pipe(model_name)

    prompt = "Toktok, who is there?"
    result = inference(
        model_pipe, 
        prompt, 
        max_new_tokens=128,
        temperature=0.9,
    )

    print(result)
