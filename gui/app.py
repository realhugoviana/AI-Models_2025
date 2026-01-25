import gradio as gr
from gui.pipeline import predict_pipeline

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 Celebrity Recognition")

    with gr.Row():
        image = gr.Image(type="pil", label="Input image")
        output = gr.Textbox(label="Result", lines=8)
        celeb_img = gr.Image(label="Closest celebrity")

    with gr.Row():
        name_opt = gr.Dropdown(
            choices=["Yes", "No"],
            value="Yes",
            label="Predict name"
        )
        sex_opt = gr.Dropdown(
            choices=["Yes", "No"],
            value="Yes",
            label="Predict sex"
        )

    gr.Button("Predict").click(
        predict_pipeline,
        [image, name_opt, sex_opt],
        [output, celeb_img]
    )

demo.launch(show_api=False)