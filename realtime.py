import cv2
import depthai as dai
import numpy as np
import time
import sys

import geometry_utils as geom_utils
sys.modules.setdefault("src.geometry_utils", geom_utils)

try:
    import onnxruntime as ort
except Exception:
    ort = None

from inference import ONNXTemporalRunner
from geometry_utils import parallelogram_from_triangle, tiny_filter_on_dets
from evaluation_utils import decode_predictions

IMG_H, IMG_W = 864, 1536
STRIDES = [8.0, 16.0, 32.0]
CONF_TH = {0: 0.50}
NMS_IOU = 0.2
CONTAIN_THR = 0.85
TOPK = 50
SCORE_MODE = "obj*cls"
MODEL_PATH = "yolo11m_2_5d_real_coshow_v5.onnx"

FONT = cv2.FONT_HERSHEY_SIMPLEX

def preprocess_frame(frame_bgr):
    if frame_bgr.shape[0] != IMG_H or frame_bgr.shape[1] != IMG_W:
        frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return frame_bgr, np.expand_dims(img_np, 0)

def draw_detections(frame_bgr, dets):
    for det in dets:
        tri = np.asarray(det.get("tri"), dtype=np.float32)
        if tri.shape != (3, 2):
            continue
        poly4 = parallelogram_from_triangle(tri[0], tri[1], tri[2]).astype(np.int32)
        cv2.polylines(frame_bgr, [poly4], True, (0, 255, 0), 2, cv2.LINE_AA)
        p0 = poly4[0]
        cv2.putText(
            frame_bgr,
            f"{det.get('score', 0.0):.2f}",
            (int(p0[0]), max(16, int(p0[1]) - 4)),
            FONT,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

def make_runner():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if ort is None or ort.get_device().upper() != "GPU":
        providers = ["CPUExecutionProvider"]
    return ONNXTemporalRunner(
        MODEL_PATH,
        providers=providers,
        state_stride_hint=32,
        default_hidden_ch=256,
    )

def main():
    runner = make_runner()

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        video_queue = cam.requestOutput((IMG_W, IMG_H)).createOutputQueue()
        benchmark_in = pipeline.create(dai.node.BenchmarkIn)
        output = cam.requestFullResolutionOutput()
        output.link(benchmark_in.input)

        pipeline.start()
        last_log = time.perf_counter()

        while pipeline.isRunning():
            msg = video_queue.get()
            if msg is None:
                continue
            frame_bgr = msg.getCvFrame()

            total_start = time.perf_counter()
            frame_proc, input_tensor = preprocess_frame(frame_bgr)

            infer_start = time.perf_counter()
            outs = runner.forward(input_tensor)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            dets_batch = decode_predictions(
                0,
                outs,
                STRIDES,
                clip_cells=None,
                conf_th=CONF_TH,
                nms_iou=NMS_IOU,
                topk=TOPK,
                contain_thr=CONTAIN_THR,
                score_mode=SCORE_MODE,
                use_gpu_nms=True,
            )
            dets = tiny_filter_on_dets(dets_batch[0], min_area=20.0, min_edge=3.0)

            draw_detections(frame_proc, dets)

            total_ms = (time.perf_counter() - total_start) * 1000.0
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0

            cv2.putText(frame_proc, f"Infer: {infer_ms:.1f} ms", (10, 24), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame_proc,
                f"Total: {total_ms:.1f} ms ({fps:.1f} FPS)",
                (10, 48),
                FONT,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLO11 realtime", frame_proc)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            now = time.perf_counter()
            if now - last_log >= 1.0:
                print(
                    f"Inference {infer_ms:.1f} ms | Total {total_ms:.1f} ms | FPS {fps:.1f} | dets {len(dets)}"
                )
                last_log = now

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
