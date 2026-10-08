# AirLLM experiments

Two interactive CLIs for running large language models locally.

- `airllm-openai-server-client.py` installs, launches, and chats with an
  [airllm-openai-server](https://github.com/mkamranr/airllm-openai-server) through its
  OpenAI-compatible API. It runs the server natively with MLX acceleration on macOS and in
  Docker elsewhere.
- `airllm-lib-usage.py` runs the [airllm](https://github.com/lyogavin/airllm) library in-process
  (MLX on macOS, torch elsewhere), with interactive model selection and a chat session.

## airllm-openai-server-client.py

On macOS the tool runs the server natively with the MLX backend; on other platforms it builds and
runs the Docker image. Either way it opens a streaming chat session against the server, and on
later Docker runs it reuses the existing container.

### Requirements

- Python 3.7+
- [Docker](https://docs.docker.com/get-docker/) running locally (non-macOS only)
- `git` (only needed when the server source must be cloned)
- Python package `requests`

```sh
python3 -m pip install requests
```

On macOS, Apple silicon is required. The tool installs `airllm-openai-server[inference,mlx]`
(including `airllm` and `mlx`) into the current Python environment on first run, so no Docker is
needed.

### Usage

```sh
python3 airllm-openai-server-client.py
```

#### First run

On non-macOS the server is not installed, so the tool prompts for the installation settings.
Press Enter to accept each default:

| Setting          | Default                                              |
| ---------------- | ---------------------------------------------------- |
| Repository URL   | `https://github.com/mkamranr/airllm-openai-server.git` |
| Source directory | `airllm-openai-server`                               |
| Dockerfile path  | `docker/Dockerfile`                                  |
| Image tag        | `airllm-server:cpu`                                  |
| Container name   | `airllm-server`                                      |
| Model name       | `Qwen/Qwen2.5-0.5B-Instruct`                         |
| Host port        | `8000`                                               |
| Container port   | `8000`                                               |
| Cache volume     | `airllm-cache`                                       |
| HuggingFace token| none                                                 |

It then clones the source (if missing), builds the CPU image, starts the container with
the equivalent of:

```sh
docker build -f docker/Dockerfile -t airllm-server:cpu .
docker run -d -p 8000:8000 -v airllm-cache:/cache \
  -e AIRLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct airllm-server:cpu
```

The model is chosen from the shared catalog in `model_catalog.py`, the same list used by
`airllm-lib-usage.py`.

After the health check passes, the interactive chat starts.

#### macOS

macOS is detected automatically, so Docker is not used. The tool only prompts for the model,
the host port and (optionally) a HuggingFace token, then installs
`airllm-openai-server[inference,mlx]` if needed and starts the server with the MLX backend:

```sh
python3 -m airllm_server --host 127.0.0.1 --port 8000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --backend mlx
```

Only MLX-compatible (Llama-style) models are offered. The server runs as a child process and is
stopped when you exit the chat.

#### Later runs

A running container based on the `airllm-server:cpu` image is detected automatically, so
installation is skipped and the default configuration is reused. The native macOS server is
restarted on each run.

## airllm-lib-usage.py

Runs AirLLM directly in the current Python process, without Docker.

### Requirements

- Python 3.9+
- `pip` (missing dependencies are installed on demand)
- macOS: Apple silicon, using AirLLM's MLX runtime
- Other platforms: CUDA or CPU, using AirLLM's torch runtime
- A Hugging Face access token for gated models

### Usage

```sh
python3 airllm-lib-usage.py
```

The tool then:

1. Shows a numbered list of models (enter a number, a repository id, or an empty line for the
   default).
2. Installs `airllm` (and `mlx` on macOS) if they are missing.
3. Loads the selected model and opens a chat session.

On the first load AirLLM downloads the model and splits it into layer shards, which can take
a long time.

#### Gated models

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) before starting when using gated models:

```sh
export HF_TOKEN=hf_...
python3 airllm-lib-usage.py
```

#### macOS

AirLLM's macOS runtime uses its MLX Llama implementation, which only supports Llama-style
architectures (no attention bias, untied embeddings). Qwen and similar models are therefore
only offered on non-macOS platforms.

## Chat commands

Both tools share the same chat commands:

| Command  | Action                                                    |
| -------- | --------------------------------------------------------- |
| `/help`  | Show available commands                                   |
| `/clear` | Clear the conversation                                    |
| `/exit`  | Leave the chat (or `/quit`)                               |
| `Ctrl+C` | Interrupt a response; press again at the prompt to exit   |

`Ctrl+D` also exits.

## Notes

- The first reply can take minutes: AirLLM downloads the model and splits it into layer
  shards before generating. It reads every layer from disk per token, so throughput is
  seconds per token by design.
- `airllm-lib-usage.py` stores the Hugging Face download and the layer shards in the standard
  Hugging Face cache (`~/.cache/huggingface`). `airllm-openai-server-client.py` keeps them in
  the `airllm-cache` Docker volume on non-macOS and in the standard Hugging Face cache when run
  natively on macOS. Keep these to avoid re-downloading the model on restart.
- Gated models (for example `meta-llama/*`) require accepting their terms on Hugging Face and
  providing an access token.
