# import openai

# # Configure OpenAI client to use Ollama
# client = openai.OpenAI(
#     base_url="http://localhost:11434/v1",  # Ollama API URL
#     api_key="ollama",  # Ollama does not require authentication, so use a dummy key
# )

# # Call the model
# response = client.completions.create(
#     model="llama3.2:3b",  # Adjust model name if necessary
#     prompt="What is reinforcement learning?",
#     max_tokens=100
# )

# # Print the output
# print(response.choices[0].text.strip())

## text embedding
# text = "Reinforcement learning is a branch of machine learning."

# # Call the embedding model
# response = client.embeddings.create(
#     model="nomic-embed-text",  # Ensure the model is available in Ollama
#     input=text
# )

# # Extract and print embeddings
# embedding = response.data[0].embedding
# print("Embedding:", embedding)

if __name__ == "__main__":
    # #for tiktoken local cache
    # import tiktoken_ext.openai_public
    # import inspect

    # print(dir(tiktoken_ext.openai_public))
    # # The encoder we want is cl100k_base, we see this as a possible function

    # print(inspect.getsource(tiktoken_ext.openai_public.cl100k_base))
    # import hashlib

    # blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    # cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
    # print('*'*10)
    # print(cache_key)
    # import os

    # tiktoken_cache_dir = "/home/jx0800/.cache/tiktoken"
    # os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

    # # validate
    # assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key))