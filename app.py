from __future__ import annotations

import gradio as gr
import spaces
from PIL import Image
import torch

from my_run import run as run_model


DESCRIPTION = '''# Turbo Edit
'''

@spaces.GPU
def main_pipeline(
        input_image: str,
        src_prompt: str,
        tgt_prompt: str,
        seed: int,
        w1: float,
        # w2: float,
        ):

        w2 = 1.0
        res_image = run_model(input_image, src_prompt, tgt_prompt, seed, w1, w2)

        return res_image


with gr.Blocks(css='app/style.css') as demo:
    gr.Markdown(DESCRIPTION)

    gr.HTML(
        '''<a href="https://huggingface.co/spaces/garibida/ReNoise-Inversion?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to run privately without waiting in queue''')

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input image",
                type="filepath",
                height=512,
                width=512
            )
            src_prompt = gr.Text(
                label='Source Prompt',
                max_lines=1,
                placeholder='Source Prompt',
            )
            tgt_prompt = gr.Text(
                label='Target Prompt',
                max_lines=1,
                placeholder='Target Prompt',
            )
            with gr.Accordion("Advanced Options", open=False):
                seed = gr.Slider(
                    label='seed',
                    minimum=0,
                    maximum=16*1024,
                    value=7865,
                    step=1
                )
                w1 = gr.Slider(
                    label='w',
                    minimum=1.0,
                    maximum=3.0,
                    value=1.5,
                    step=0.05
                )
                # w2 = gr.Slider(
                #     label='w2',
                #     minimum=1.0,
                #     maximum=3.0,
                #     value=1.0,
                #     step=0.05
                # )

            run_button = gr.Button('Edit')
        with gr.Column():
            # result = gr.Gallery(label='Result')
            result = gr.Image(
                label="Result",
                type="pil",
                height=512,
                width=512
            )

            examples = [
                [
                    "demo_im/WhatsApp Image 2024-05-17 at 17.32.53.jpeg", #input_image
                    "a painting of a white cat sleeping on a lotus flower", #src_prompt
                    "a painting of a white cat sleeping on a lotus flower", #tgt_prompt
                    4759, #seed
                    1.0, #w1
                    # 1.1, #w2
                ],
                [
                    "demo_im/pexels-pixabay-458976.less.png", #input_image
                    "a squirrel standing in the grass", #src_prompt
                    "a squirrel standing in the grass", #tgt_prompt
                    6128, #seed
                    1.25, #w1
                    # 1.1, #w2
                ],
            ]

            gr.Examples(examples=examples,
                        inputs=[
                            input_image,
                            src_prompt,
                            tgt_prompt,
                            seed,
                            w1,
                            # w2,
                        ],
                        outputs=[
                            result
                        ],
                        fn=main_pipeline,
                        cache_examples=True)


    inputs = [
        input_image,
        src_prompt,
        tgt_prompt,
        seed,
        w1,
        # w2,
    ]
    outputs = [
        result
    ]
    run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)

demo.queue(max_size=50).launch(share=True, max_threads=100)