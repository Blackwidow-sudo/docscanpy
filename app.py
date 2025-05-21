import gradio as gr
import numpy as np

from scanner.scan import Scanner


def on_img_change(image: np.ndarray):
    return gr.Button(interactive=True if image is not None else False)


def on_img_clear(image: np.ndarray, orig_img: np.ndarray):
    return None, None, None, gr.Checkbox(value=False, interactive=False)


def find_contours(image: np.ndarray):
    if image is None:
        return None, None, image, gr.Checkbox(interactive=False)
    pts = Scanner.find_contour(image)
    if pts is None:
        return image, None, image, gr.Checkbox(interactive=False)
    transformed = Scanner.transform_perspective(image, pts)
    cpy = image.copy()
    preview = Scanner.draw_contour(cpy, pts)
    return preview, transformed, image, gr.Checkbox(interactive=True)


def apply_filter(checkbox: bool, image: np.ndarray):
    preview, transformed, _, _ = find_contours(image)
    if checkbox and transformed is not None:
        return preview, Scanner.document_filter(transformed)
    return preview, transformed


with gr.Blocks() as demo:
    orig_img = gr.State(None)
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label='Input Image',
                type='numpy'
            )
            scan_btn = gr.Button(
                'Scan Document',
                variant='primary',
                interactive=False
            )
        with gr.Column():
            output_img = gr.Image(label='Output Image', type='numpy')
            filter_chkbx = gr.Checkbox(
                label='Apply black/white filter',
                interactive=False
            )

    input_img.change(
        fn=on_img_change,
        inputs=input_img,
        outputs=scan_btn,
    )

    input_img.clear(
        fn=on_img_clear,
        outputs=[input_img, output_img, orig_img, filter_chkbx]
    )

    scan_btn.click(
        fn=find_contours,
        inputs=input_img,
        outputs=[input_img, output_img, orig_img, filter_chkbx]
    )

    filter_chkbx.change(
        fn=apply_filter,
        inputs=[filter_chkbx, orig_img],
        outputs=[input_img, output_img]
    )

if __name__ == "__main__":
    demo.launch()
