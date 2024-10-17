# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import argparse
from aiohttp import web, WSCloseCode
import logging
import weakref
import cv2
import time
import PIL.Image
import matplotlib.pyplot as plt
from typing import List
from nanoowl.owl_predictor import OwlPredictor
from nanosam.utils.predictor import Predictor
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl_image_encoder", type=str, default="../../data/owl_image_encoder_patch32.engine")
    parser.add_argument("--sam_image_encoder", type=str, default="../../data/resnet18_image_encoder.engine")
    parser.add_argument("--sam_mask_decoder", type=str, default="../../data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    CAMERA_DEVICE = args.camera
    IMAGE_QUALITY = args.image_quality

    owl_predictor = OwlPredictor(
        image_encoder_engine=args.owl_image_encoder
    )

    sam_predictor = Predictor(
        args.sam_image_encoder, args.sam_mask_decoder
    )

    prompt_data = None

    def get_colors(count: int):
        cmap = plt.cm.get_cmap("rainbow", count)
        colors = []
        for i in range(count):
            color = cmap(i)
            color = [int(255 * value) for value in color]
            colors.append(tuple(color))
        return colors


    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)


    async def handle_index_get(request: web.Request):
        logging.info("handle_index_get")
        return web.FileResponse("./index.html")


    async def websocket_handler(request):

        global prompt_data

        ws = web.WebSocketResponse()

        await ws.prepare(request)

        logging.info("Websocket connected.")

        request.app['websockets'].add(ws)

        try:
            async for msg in ws:
                logging.info(f"Received message from websocket.")
                if "prompt" in msg.data:
                    header, prompt = msg.data.split(":")
                    logging.info("Received prompt: " + prompt)
                    try:
                        prompt_data = prompt
                        logging.info("Set prompt: " + prompt)
                    except Exception as e:
                        print(e)
        finally:
            request.app['websockets'].discard(ws)

        return ws


    async def on_shutdown(app: web.Application):
        for ws in set(app['websockets']):
            await ws.close(code=WSCloseCode.GOING_AWAY,
                        message='Server shutdown')


    async def detection_loop(app: web.Application):

        loop = asyncio.get_running_loop()

        logging.info("Opening camera.")

        camera = cv2.VideoCapture(CAMERA_DEVICE)

        logging.info("Loading predictor.")

        def _read_and_encode_image():

            re, image = camera.read()

            if not re:
                return re, None

            image_pil = cv2_to_pil(image)
            num_detections = 0
            if prompt_data is not None:
                prompt_data_local = prompt_data.strip("][()")
                prompt_data_local = prompt_data_local.split(",")
                t0 = time.perf_counter_ns()
                text_encodings = owl_predictor.encode_text(prompt_data_local)
                detections = owl_predictor.predict(
                    image=image_pil,
                    text=prompt_data_local,
                    text_encodings=text_encodings,
                    pad_square=False
                )
                t1 = time.perf_counter_ns()
                dt1 = (t1 - t0) / 1e6
                # logging.info(f"OWL time: {dt1:.3f}ms")
                num_detections = len(detections.labels)
                # logging.info(f"num of detections:{num_detections}")
            if num_detections != 0:
                # logging.info(f"scores: {detections.scores}")
                # TODO: use multiple bboxes as prompts
                max_score_idx = detections.scores.max(dim=0)[1].item()
                bbox = detections.boxes[max_score_idx].cpu() # or choose max score one
                # bbox = detections.boxes[0].cpu() # or choose first one
                # TODO: or choose largest one
                points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                point_labels = np.array([2, 3])
                t2 = time.perf_counter_ns()
                sam_predictor.set_image(image_pil)
                mask, _, _ = sam_predictor.predict(points, point_labels)
                t3 = time.perf_counter_ns()
                dt2 = (t3 - t2) / 1e6
                # logging.info(f"SAM time: {dt2:.3f}ms")
                mask_overlay = (mask[0, 0].detach().cpu().numpy() > 0).astype(np.uint8) * 255
                mask_overlay = cv2.applyColorMap(mask_overlay, cv2.COLORMAP_PINK)
                image = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)

            image_jpeg = bytes(
                cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])[1]
            )

            return re, image_jpeg

        while True:

            re, image = await loop.run_in_executor(None, _read_and_encode_image)

            if not re:
                break

            for ws in app["websockets"]:
                await ws.send_bytes(image)

        camera.release()


    async def run_detection_loop(app):
        try:
            task = asyncio.create_task(detection_loop(app))
            yield
            task.cancel()
        except asyncio.CancelledError:
            pass
        finally:
            await task


    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app['websockets'] = weakref.WeakSet()
    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)
    web.run_app(app, host=args.host, port=args.port)