from pathlib import Path

import gradio as gr
import pandas as pd

STATE = dict()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            imshow = gr.Image()
            filename = gr.Textbox()

            # link imshow to filename
            @filename.change(inputs=[filename], outputs=[imshow])
            def _(filename):
                return STATE["img_dir"] / filename

            with gr.Row():

                @ (gr.Button("⬅️")).click(inputs=[filename], outputs=[filename])
                def _(filename):
                    names = STATE["filenames"]
                    idx = names.index(filename)
                    return names[(idx - 1) % len(names)]

                @ (gr.Button("➡️")).click(inputs=[filename], outputs=[filename])
                def _(filename):
                    names = STATE["filenames"]
                    idx = names.index(filename)
                    return names[(idx + 1) % len(names)]

        with gr.Column():
            img_dir = gr.Textbox(label="Input image folder")

            @ (gr.Button("Load images")).click(inputs=[img_dir], outputs=[filename])
            def _(img_dir):
                img_dir = Path(img_dir)

                STATE["img_dir"] = img_dir
                STATE["filenames"] = [str(x.relative_to(img_dir)) for x in img_dir.glob("**/*.jpg")]
                STATE["filenames"].sort()
                STATE["database"] = {}

                gr.Info(f"Found {len(STATE['filenames'])} images in {img_dir}")
                return STATE["filenames"][0]

            caption = gr.Textbox(label="Caption")

            @ (gr.Button("Submit caption", variant="primary")).click(inputs=[filename, caption])
            def _(filename, caption):
                STATE["database"][filename] = caption
                gr.Info("Save caption successfully")

            export_path = gr.Textbox(label="Export path")

            @ (gr.Button("Export data")).click(inputs=[export_path])
            def _(export_path):
                Path(export_path).parent.mkdir(exist_ok=True, parents=True)
                df = pd.DataFrame(list(STATE["database"].items()), columns=["filename", "prompt"])
                df.to_csv(export_path, index=False)
                gr.Info(f"Successfully exported data to {export_path}")


demo.launch()
