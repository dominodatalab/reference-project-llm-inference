# OSS LLM Inference Template

## License
This template is licensed under Apache 2.0 and contains the following open source components: 
* Transformer [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)
* Falcon [Apache 2.0](https://huggingface.co/tiiuae/falcon-7b/blob/main/README.md)
* MLFlow [Apache 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
* NVIDIA [EULA](https://docs.nvidia.com/cuda/eula/index.html#license-grant)

## About this template
This project shows how to generate text output from a fine tuned LLM (Falcon-7b finetuned for summarization) using different inference frameworks. This project also has code that deploys the fine tuned LLM as a Model API and an app in Domino.Please note that the execution time to generate output will differ based on the hardware and model you are using, the notebooks in this project were run on 1 `V100` GPU that has 24GB of VRAM. 

In general, `ctranslate2` is a good choice to run LLMs on CPU, GPU accelerators and is highly performant while `vLLM` is most suited for use cases that require scale as it can be backed by Ray ; the native Huggingface option is good for prototyping and development and small scale use cases that leverage GPUs.  

Here are a list of important files in the project that you might need to edit in order to customize this further for your use case.

* [ft_falcon7b_8bit_lora.ipynb](https://github.com/dominodatalab/reference-project-llm-inference/blob/release-1.0.0/ft_falcon7b_8bit_lora.ipynb) : This notebook contains code to fine tune a LoRA adapter for the Falcon-7b model to perform summarization. The code also logs training metrics to `mlflow` and can be viewed in the `Experiments` section of the project.

* [convert_hf_ct.ipynb](https://github.com/dominodatalab/reference-project-llm-inference/blob/release-1.0.0/convert_hf_ct.ipynb) : : This notebook contains code to convert a Huggingface model to a `ctranslate2` model. `ctranslate2` does not support adapters out of the box so we merge it with the model and export it for subsequent use

* [bench_ct2.ipynb](https://github.com/dominodatalab/reference-project-llm-inference/blob/release-1.0.0/bench_ct2.ipynb) : This notebook contains code that loads a `ctranslate2` model and generates output from it.
  
* [bench_hf.ipynb](https://github.com/dominodatalab/reference-project-llm-inference/blob/release-1.0.0/bench_hf.ipynb) : This notebook contains code that loads a `Huggingface` model and generates output from it. 

* [bench_vllm.ipynb](https://github.com/dominodatalab/reference-project-llm-inference/blob/release-1.0.0/bench_vllm.ipynb) :  This notebook contains code that uses `vLLM` to generate output from a Huggingface model that has the summarization adapter attached to it
  
* [app.sh](https://github.com/dominodatalab/reference-project-llm-inference/blob/release-1.0.0/app.sh) : The shell script needed to run the chat app

* [app.py](https://github.com/dominodatalab/reference-project-llm-inference/blob/release-1.0.0/app.py) : Streamlit app code for the summarization app. This app uses the `ctranslate2` model to generate responses

* [model.py](https://github.com/dominodatalab/reference-project-llm-inference/blob/release-1.0.0/model.py) : This file has sample code that shows how to use the `ctranslate2` model as an API. Please ensure that the build pods have enough resources to build the API and the Model API has the right resource quota assigned to it
\
&nbsp;


<p float="left">
  <img src="https://github.com/dominodatalab/reference-project-llm-inference/blob/a83eae919f8cbc1bdb206ef243f088eda7ce9523/mlflow.png?raw=true" width="410" height="300" /> 
  <img src="https://github.com/dominodatalab/reference-project-llm-inference/blob/a83eae919f8cbc1bdb206ef243f088eda7ce9523/model_api.png?raw=true" width="410" height="300" />
</p>

<p float="center">
  <img src="https://github.com/dominodatalab/reference-project-llm-inference/blob/a83eae919f8cbc1bdb206ef243f088eda7ce9523/summarization_app.png?raw=true" width="860" height="350" />
</p>




## Setup instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present. Also please ensure the [‘Automatically make compatible with Domino’](https://docs.dominodatalab.com/en/latest/user_guide/a00d1b/automatic-adaptation-of-custom-images/#_pre_requisites_for_automatic_custom_image_compatibility_with_domino) checkbox is selected and create the environment afresh instead of duplicating an existing environment that does not use the same base image and dependencies.


### LLM Inference

#### Hardware Requirements
The notebooks in this project require 1 V100 GPU that has 24GB of VRAM.

#### Environment Requirements

`nvcr.io/nvidia/pytorch:22.12-py3`

**Dockerfile Instructions**

```
# System-level dependency injection runs as root
USER root:root

# Validate base image pre-requisites
# Complete requirements can be found at
# https://docs.dominodatalab.com/en/latest/user_guide/a00d1b/automatic-adaptation-of-custom-images/#_pre_requisites_for_automatic_custom_image_compatibility_with_domino
RUN /opt/domino/bin/pre-check.sh

# Configure /opt/domino to prepare for Domino executions
RUN /opt/domino/bin/init.sh

# Validate the environment
RUN /opt/domino/bin/validate.sh

RUN pip uninstall --yes torch torchvision torchaudio

RUN pip install torch  --index-url https://download.pytorch.org/whl/cu118

RUN pip uninstall -y protobuf
RUN pip install "protobuf==3.20.3" "mlflow==2.6.0"
RUN pip install -q -U bitsandbytes==0.39.1 "datasets>=2.10.0,<3" "ipywidgets" "ctranslate2==3.17.1"
RUN pip install -q -U py7zr einops tensorboardX transformers peft accelerate deepspeed
RUN pip install --no-cache-dir Flask Flask-Compress Flask-Cors jsonify uWSGI streamlit streamlit-chat "vllm==0.1.3"
RUN pip uninstall --yes transformer-engine

RUN pip uninstall -y apex
```
