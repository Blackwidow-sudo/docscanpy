import gradio as gr
import numpy as np
import os

from scanner.scan import Scanner

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')


def on_img_change(image: np.ndarray):
    return gr.Button(interactive=True if image is not None else False)


def on_img_clear():
    return [
        None,
        None,
        None,
        gr.Checkbox(value=False, interactive=False),
        gr.DownloadButton(value=None, interactive=False)
    ]

def find_contours(image: np.ndarray):
    if image is None:
        return None, None, image, gr.Checkbox(interactive=False)
    pts = Scanner.find_contour(image)
    if pts is None:
        return image, None, image, gr.Checkbox(interactive=False)
    transformed = Scanner.transform_perspective(image, pts)
    preview = Scanner.draw_contour(image.copy(), pts)
    document = Scanner.to_pdf(transformed, os.path.join(UPLOAD_DIR, 'document.pdf'))
    return [
        preview,
        transformed,
        image,
        gr.Checkbox(interactive=True),
        gr.DownloadButton(value=document, interactive=True)
    ]


def apply_filter(checkbox: bool, image: np.ndarray):
    preview, transformed, _, _, document = find_contours(image)
    if checkbox and transformed is not None:
        filtered = Scanner.document_filter(transformed)
        document = Scanner.to_pdf(filtered, os.path.join(UPLOAD_DIR, 'document.pdf'))
        return preview, filtered, gr.DownloadButton(value=document)
    return preview, transformed, document


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
            download_btn = gr.DownloadButton('Download', interactive=False)

    input_img.change(
        fn=on_img_change,
        inputs=input_img,
        outputs=scan_btn,
    )

    input_img.clear(
        fn=on_img_clear,
        outputs=[input_img, output_img, orig_img, filter_chkbx, download_btn]
    )

    scan_btn.click(
        fn=find_contours,
        inputs=input_img,
        outputs=[input_img, output_img, orig_img, filter_chkbx, download_btn]
    )

    filter_chkbx.change(
        fn=apply_filter,
        inputs=[filter_chkbx, orig_img],
        outputs=[input_img, output_img, download_btn]
    )

if __name__ == "__main__":
    demo.launch()
