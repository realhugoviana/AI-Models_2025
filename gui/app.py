import gradio as gr
from gui.pipeline import predict_pipeline

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 Celebrity Recognition")

    with gr.Row():
        image = gr.Image(type="pil")
        output = gr.JSON()
        celeb_img = gr.Image()

    with gr.Row():
        name_cb = gr.Checkbox(label="Predict name", value=True)
        sex_cb = gr.Checkbox(label="Predict sex", value=True)

    gr.Button("Predict").click(
        predict_pipeline,
        [image, name_cb, sex_cb],
        [output, celeb_img]
    )

demo.launch()
