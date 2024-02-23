import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler, LlamaChatCompletionHandler
from PIL import Image
from io import BytesIO
from typing import Any
import logging
import sys
import streamlit as st


# Load model and projection matrix
@st.cache_resource
def load_model():
    """
    Loads the LLaMA model, alongside its projection matrix.

    Decorators:
        @st.cache_resource - avoid reloading the model whenever the user presses a “Refresh” button in the browser
    """
    file_model = 'llava-v1.5-7b-Q4_K.gguf'
    file_mmproj_model = 'llava-v1.5-7b-mmproj-f16.gguf'

    chatter = Llava15ChatHandler(clip_model_path=file_mmproj_model)
    # chatter = LlamaChatCompletionHandler()

    model = Llama(
        model_path=file_model,
        chat_handler=chatter,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=True,
        logits_all=True
    )

    return model


def image_b64_encode(img: Image) -> str:
    """Converts image to a base64 format.

    Args:
        img (str): The image to be converted
    """
    buffered = BytesIO()

    img.save(buffered, format='JPEG')

    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def inference(model: Any, query: str, img: Image) -> str:
    """Prompts the model with a query.

    Args:
        model (Any): The assistant used to answer the query.
        query (str): The request asked by the user.
        img (PIL.Image): The image to be described.
    """
    img_b64 = image_b64_encode(img)

    out_stream = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a very smart assistant, always looking to help the user and can perfectly "
                           "describe images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
        ],
        stream=True,
        temperature=0.35
    )

    out = ""

    for response in out_stream:
        data = response['choices'][0]['delta']
        if 'content' in data:
            print(data['content'], end="")
            logging.info(data['content'])

            sys.stdout.flush()
            out += data['content']

    return out


# Just for debug
# model = load_model()
# img = Image.open('car.jpg')
# inference(model, 'Describe this image', img)
