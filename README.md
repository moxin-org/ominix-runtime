
openvla.cpp is an open-source project based on llama.cpp. It currently supports the deployment and inference of multiple vision-language-action models on llama.cpp, including OpenVLA and OpenVote.
## 1. Features

1.	Based on ggml, it does not rely on other third-party libraries and is committed to edge deployment.
2.  Support Q3, Q4, Q5, Q6, Q8 quantization.

### 1.1 Backend Support

1.	Support more backends. In theory, ggml supports the following backends, and future adaptations will be gradually made. Contributions are welcome.

| Backend                                   | Device               | Supported    |
|--------------------------------------|----------------------|--------------|
| CPU                                  | All                  | ✅            |
| [Metal](./docs/build.md#metal-build) | Apple Silicon        | ✅            |   
| [BLAS](./docs/build.md#blas-build)   | All                  | ✅            |
| [CUDA](./docs/build.md#cuda)         | Nvidia GPU           | ✅            |
| [Vulkan](./docs/build.md#vulkan)     | GPU                  | ✅            |
| [BLIS](./docs/backend/BLIS.md)       | All                  |              |
| [SYCL](./docs/backend/SYCL.md)       | Intel and Nvidia GPU |              |

## 2. Usage

### Download Model

```bash
git lfs install
# openvla gguf model
git clone https://huggingface.co/MoYoYoTech/openvla-gguf
# vote gguf model
git clone https://huggingface.co/MoYoYoTech/spatial-gguf
```

### Download Code

```bash
# 1. get src code
git clone --recursive https://huggingface.co/MoYoYoTech/openvla.cpp

# 2. unzip tokenizers-cpp
cd openvla.cpp
unzip vendor/tokenizers-cpp.zip -d vendor/

# 3. CMake
cmake --preset x64-linux-clang-release

# 4. build
cmake --build build
```

### Parameter Description

```bash
usage: ./build/bin/openvla --model_dir /mount/weights/vote_model/ --llm_model llm_fp16.gguf --action_head_model action_head.gguf  -t tokenizer.json -i /mount/weights/vote_model/2.png

OPTIONS:
  -h,     --help              Print this help message and exit 
  -m,     --model_dir TEXT    Base directory for models (default: /mount/weights/vote_model/) 
          --dinov2_model TEXT DINOv2 model filename in the model directory (default: 
                              dinov2.gguf) 
          --siglip_model TEXT Siglip model filename in the model directory (default: 
                              siglip.gguf) 
          --proj_model TEXT   Projection model filename in the model directory (default: 
                              proj.gguf) 
          --action_head_model TEXT 
                              Action head model filename in the model directory (default: 
                              action_head.gguf) 
          --llm_model TEXT    LLM model filename in the model directory (default: 
                              llm_q8_0.gguf) 
  -t,     --tokenizer TEXT    Path to the tokenizer (default: empty, use built-in tokenizer) 
  -i,     --img TEXT          Path to the input image 
  -p,     --prompt TEXT       Text prompt for the model 
  -d,     --device TEXT       Device name for computation (default: CUDA0) 
  -n,     --n_threads INT     Number of threads for computation (default: 4) 
  -c,     --n_ctx INT         Context size for LLM (default: 300)

```

## other

### python TEST

```bash
# install pybind11 
pip install pybind11
# Add the -DBUILD_PYTHON=ON flag during compilation, which will generate openvla.so in the build/bin directory. Copy openvla.so to the directory containing your Python script, or add its location to the Python path via environment variables (e.g., PYTHONPATH).

# run
python ominix/openvla/test_openvla.py
```

### convert openvla7b to gguf

```bash
python ominix/openvla/export_openvla7b.py
```

### convert to gguf

```bash
python convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-f16.gguf  --outtype f16
```

### quantize

```bash
./bin/llama-quantize ${model_bf16} ${model_q8_0} q8_0 $(nproc)
```
