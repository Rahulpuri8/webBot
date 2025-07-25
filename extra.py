from llama_cpp import Llama

# # llm = Llama(
# #       model_path="TheBloke/Llama-2-7B-GGUF",
# #       # n_gpu_layers=-1, # Uncomment to use GPU acceleration
# #       # seed=1337, # Uncomment to set a specific seed
# #       # n_ctx=2048, # Uncomment to increase the context window
# # )

llm = Llama.from_pretrained(
    repo_id="TheBloke/Llama-2-7B-GGUF",
    filename="*Q2_K.gguf",
    verbose=False
)



output = llm(
      "Q: Name the planets in the solar system?", # Prompt
      max_tokens=232, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["\n", "Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output["choices"][0]["text"].strip())




# from llama_cpp import Llama

# llm = Llama(
#     model_path="/home/saadhassan/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-GGUF/snapshots/b4e04e128f421c93a5f1e34ac4d7ca9b0af47b80/./llama-2-7b.Q2_K.gguf",
#     n_ctx=2048,
#     n_threads=4,
# )

# def ask_llm(prompt: str) -> str:
#     output = llm(prompt, max_tokens=512, stop=["</s>"])
#     return output["choices"][0]["text"].strip()

# result = ask_llm("Name the planets in the solar system?")
# print(result)

