from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

STATE = dict()

shortcut_js = """
<script>
document.addEventListener(
    'keydown',
    (e) => {
        if (["input", "textarea"].includes(event.target.tagName.toLowerCase())) {
            return;
        }
        switch (event.key) {
            case "ArrowLeft":
                document.getElementById("prev_btn").click();
                break;
            case "ArrowRight":
                document.getElementById("next_btn").click();
                break;
        }
    }
);
</script>
"""

with gr.Blocks(head=shortcut_js) as demo:
    with gr.Row():
        with gr.Column():
            imshow = gr.Image()
            img_idx = gr.Number(label="Image index", value=-1)
            filename = gr.Textbox(label="Filename")

            with gr.Row():

                @ (gr.Button("⬅️", elem_id="prev_btn")).click(inputs=[img_idx], outputs=[img_idx])
                def _(img_idx):
                    return (int(img_idx) - 1) % len(STATE["filenames"])

                @ (gr.Button("➡️", elem_id="next_btn")).click(inputs=[img_idx], outputs=[img_idx])
                def _(img_idx):
                    return (int(img_idx) + 1) % len(STATE["filenames"])

            reload_btn = gr.Button("Reload")

        with gr.Column():
            img_dir = gr.Textbox(label="Input image folder")

            @img_dir.submit(inputs=[img_dir])
            def _(img_dir):
                img_dir = Path(img_dir)

                STATE["img_dir"] = img_dir
                STATE["filenames"] = []
                for ext in ("jpg", "webp"):
                    STATE["filenames"].extend([str(x.relative_to(img_dir)) for x in img_dir.glob(f"**/*.{ext}")])
                STATE["filenames"].sort()
                STATE["database"] = {}

                gr.Info(f"Found {len(STATE['filenames'])} images in {img_dir}")

            taglist_path = gr.Textbox(label="Path to taglist")

            @gr.render(inputs=[taglist_path], triggers=[taglist_path.submit])
            def _(taglist_path):
                data = yaml.safe_load(open(taglist_path))

                def add_tag(caption, tag):
                    return f"{caption}, {tag}" if caption else tag

                with gr.Column():
                    for key, value in data.items():
                        assert isinstance(value, list)

                        with gr.Group(), gr.Row():
                            for tag in value:
                                tag_btn = gr.Button(tag, min_width=100)
                                tag_btn.click(add_tag, inputs=[caption, tag_btn], outputs=[caption])

            caption = gr.Textbox(label="Caption")

            @ (gr.Button("Save caption", variant="primary", elem_id="submit_btn")).click(inputs=[filename, caption])
            def _(filename, caption):
                STATE["database"][filename] = caption
                gr.Info("Save caption successfully")

            db_path = gr.Textbox(label="Database path")

            @ (gr.Button("Import data")).click(inputs=[db_path])
            def _(db_path):
                df = pd.read_csv(db_path)
                for filename, prompt in zip(df["filename"], df["prompt"]):
                    assert filename in STATE["filenames"]
                    STATE["database"][filename] = prompt
                gr.Info(f"Successfully imported data from {db_path}")

            @ (gr.Button("Export data")).click(inputs=[db_path])
            def _(db_path):
                Path(db_path).parent.mkdir(exist_ok=True, parents=True)
                df = pd.DataFrame(list(STATE["database"].items()), columns=["filename", "prompt"])
                df.sort_values("filename").to_csv(db_path, index=False)
                gr.Info(f"Successfully exported data to {db_path}")

        # link img_idx to other stuff
        @gr.on(
            triggers=[img_idx.change, reload_btn.click],
            inputs=[img_idx],
            outputs=[filename, imshow, caption],
        )
        def _(img_idx):
            filename = STATE["filenames"][int(img_idx)]
            return filename, STATE["img_dir"] / filename, STATE["database"].get(filename, "")


demo.launch()
