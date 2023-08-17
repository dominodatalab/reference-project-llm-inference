*Disclaimer - Domino Reference Projects are starter kits built by Domino researchers. They are not officially supported by Domino. Once loaded, they are yours to use or modify as you see fit. We hope they will be a beneficial tool on your journey!

# OSS LLM Inference Reference Project
Project to show how to generate text output from LLMs using different inference frameworks

* [bench_ct2.ipynb](bench_ct2.ipynb) : This file loads a PDF,converts it to embeddings, stores the embeddings in Pinecone, runs the semantic search against the embeddings, constructs a prompt and calls OpenAI's models to get a response. You will need your OpenAPI and Pinecone keys to be set in the environment for this example.

* [bench_ct2.ipynb](bench_ct2.ipynb) : This file loads a PDF, converts it to embeddings, stores the embeddings locally using a FAISS index, runs the semantic search against the embeddings, constructs a prompt and calls OpenAI's models to get a response. You will need your OpenAPI key to be set in the environment for this example.

* [bench_vllm.ipynb](bench_vllm.ipynb) : This file loads a documents from a Domino Dataset, converts it to embeddings, stores the embeddings locally using a FAISS index (there is code you can uncomment if you want to use Pinecone), runs the semantic search against the embeddings, constructs a prompt and calls OpenAI's models to get a response. You will need your OpenAPI key to be set in the environment for this example.

* [faiss_ddl_doc_store.pkl](faiss_store.pkl) : This file contains the FAISS embeddings of Domino's documentation . You can use this if you don't want to (re)compute embeddings of Select_Global_Value_Fund.pdf again

* [app.sh](app.sh) : The shell script needed to run the chat app

* [app.py](app.py) : Streamlit app code for the Q&A chatbot. This app uses ```faiss_ddl_doc_store.pkl``` for the embeddings

* [Select_Global_Value_Fund.pdf](Select_Global_Value_Fund.pdf) : A report that can be used as an example for the flow that has been described above in case you want to compute embeddings on a fresh document

* [Solution_Overview.pdf](Solution_Overview.pdf) : A diagram that depicts the different components and the flow of information between them

## Setup instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present:

### PromptEngineering
**Environment Base** 

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
