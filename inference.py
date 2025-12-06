# inference_onnx.py
# YOLO11 2.5D temporal inference with ONNX (ConvLSTM/GRU hidden-state carry + seq reset + BEV)
import os
import cv2
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
import onnxruntime as ort
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ====== 기존 유틸 그대로 재사용 ======
from geometry_utils import parallelogram_from_triangle, tiny_filter_on_dets
from evaluation_utils import (
    decode_predictions,
    evaluate_single_image,
    compute_detection_metrics,
    orientation_error_deg,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MplPolygon = None
    Line2D = None
    _MATPLOTLIB_AVAILABLE = False

# =========================
# I/O helpers (라벨/BEV/시각화) - pth infer와 동일
# =========================
def load_gt_triangles(label_path: str) -> np.ndarray:
    if not os.path.isfile(label_path):
        return np.zeros((0,3,2), dtype=np.float32)
    tris = []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 7: continue
            _, p0x, p0y, p1x, p1y, p2x, p2y = p
            tris.append([[float(p0x), float(p0y)],
                         [float(p1x), float(p1y)],
                         [float(p2x), float(p2y)]])
    if len(tris) == 0:
        return np.zeros((0,3,2), dtype=np.float32)
    return np.asarray(tris, dtype=np.float32)

def poly_from_tri(tri: np.ndarray) -> np.ndarray:
    p0, p1, p2 = tri[0], tri[1], tri[2]
    return parallelogram_from_triangle(p0, p1, p2).astype(np.float32)

def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:,0], poly[:,1]
    return float(abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))) * 0.5)

def iou_polygon(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    try:
        pa = poly_a.astype(np.float32)
        pb = poly_b.astype(np.float32)
        inter_area, _ = cv2.intersectConvexConvex(pa, pb)
        if inter_area <= 0:
            return 0.0
        ua = polygon_area(pa)
        ub = polygon_area(pb)
        union = ua + ub - inter_area
        return float(inter_area / max(union, 1e-9))
    except Exception:
        xa1, ya1 = poly_a[:,0].min(), poly_a[:,1].min()
        xa2, ya2 = poly_a[:,0].max(), poly_a[:,1].max()
        xb1, yb1 = poly_b[:,0].min(), poly_b[:,1].min()
        xb2, yb2 = poly_b[:,0].max(), poly_b[:,1].max()
        inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
        inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
        inter = inter_w * inter_h
        ua = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        ub = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = ua + ub - inter
        return float(inter / max(union, 1e-9))

def draw_pred_only(image_bgr, dets, save_path_img, save_path_txt, W, H, W0, H0):
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_txt), exist_ok=True)
    img = image_bgr.copy()
    sx, sy = float(W0) / float(W), float(H0) / float(H)

    lines = []
    tri_orig_list: List[np.ndarray] = []
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float32)
        score = float(d["score"])
        poly4 = parallelogram_from_triangle(tri[0], tri[1], tri[2]).astype(np.int32)
        cv2.polylines(img, [poly4], isClosed=True, color=(0,255,0), thickness=2)
        cx, cy = int(tri[0][0]), int(tri[0][1])
        cv2.putText(img, f"{score:.2f}", (cx, max(0, cy-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        tri_orig = tri.copy()
        tri_orig[:, 0] *= sx
        tri_orig[:, 1] *= sy
        tri_orig_list.append(tri_orig.copy())
        p0o, p1o, p2o = tri_orig[0], tri_orig[1], tri_orig[2]
        lines.append(f"0 {p0o[0]:.2f} {p0o[1]:.2f} {p1o[0]:.2f} {p1o[1]:.2f} {p2o[0]:.2f} {p2o[1]:.2f} {score:.4f}")

    cv2.imwrite(save_path_img, img)
    with open(save_path_txt, "w") as f:
        f.write("\n".join(lines))
    return tri_orig_list

def draw_pred_with_gt(image_bgr_resized, dets, gt_tris_resized, save_path_img_mix, iou_thr=0.5):
    os.makedirs(os.path.dirname(save_path_img_mix), exist_ok=True)
    img = image_bgr_resized.copy()
    for g in gt_tris_resized:
        poly_g = poly_from_tri(g).astype(np.int32)
        cv2.polylines(img, [poly_g], True, (0,255,0), 2)
        for k in range(3):
            x = int(round(float(g[k,0]))); y = int(round(float(g[k,1])))
            cv2.circle(img, (x, y), 3, (0,255,0), -1)
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float32)
        score = float(d["score"])
        poly4 = poly_from_tri(tri).astype(np.int32)
        cv2.polylines(img, [poly4], True, (0,0,255), 2)
        p0 = tri[0].astype(int)
        cv2.putText(img, f"{score:.2f}", (int(p0[0]), max(0, int(p0[1])-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    if gt_tris_resized.shape[0] > 0 and len(dets) > 0:
        det_idx_sorted = np.argsort([-float(d["score"]) for d in dets])
        matched = np.zeros((gt_tris_resized.shape[0],), dtype=bool)
        for di in det_idx_sorted:
            d = dets[di]
            tri_d = np.asarray(d["tri"], dtype=np.float32)
            poly_d = poly_from_tri(tri_d)
            best_j, best_iou = -1, 0.0
            for j, gtri in enumerate(gt_tris_resized):
                if matched[j]: continue
                poly_g = poly_from_tri(gtri)
                iou = iou_polygon(poly_d, poly_g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                matched[best_j] = True
                p0 = tri_d[0].astype(int)
                cv2.putText(img, f"IoU {best_iou:.2f}",
                            (int(p0[0]), int(p0[1]) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.imwrite(save_path_img_mix, img)

def normalize_angle_deg(angle: float) -> float:
    while angle <= -180.0: angle += 360.0
    while angle > 180.0: angle -= 360.0
    return angle

def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0: return pts.copy()
    flat = pts.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([flat, ones], axis=1)
    proj = homog @ H.T
    denom = proj[:, 2:3]
    result = np.full_like(proj[:, :2], np.nan, dtype=np.float64)
    valid = np.abs(denom[:, 0]) > 1e-9
    if np.any(valid):
        result[valid] = proj[valid, :2] / denom[valid]
    return result.reshape(pts.shape)

def _read_h_matrix(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix.lower() == ".npy":
            data = np.load(path)
        else:
            data = np.loadtxt(path)
    except Exception:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 9: arr = arr.reshape(3, 3)
    if arr.shape != (3, 3): return None
    return arr

def load_homography(calib_dir: str, image_name: str, cache: dict, invert: bool = False) -> Optional[np.ndarray]:
    base = os.path.splitext(os.path.basename(image_name))[0]
    if base in cache: return cache[base]
    search_order = [base + ext for ext in (".txt", ".npy", ".csv")]
    H = None
    for candidate in search_order:
        c_path = Path(calib_dir) / candidate
        if c_path.is_file():
            H = _read_h_matrix(c_path); break
    if H is None:
        matches = sorted(Path(calib_dir).glob(base + ".*"))
        for c_path in matches:
            H = _read_h_matrix(c_path)
            if H is not None: break
    if H is not None and invert:
        try: H = np.linalg.inv(H)
        except np.linalg.LinAlgError: H = None
    cache[base] = H
    return H

def compute_bev_properties(tri):
    p0, p1, p2 = np.asarray(tri, dtype=np.float64)
    if not np.all(np.isfinite(tri)):
        return None
    front_center = (p1 + p2) / 2.0
    front_vec = front_center - p0
    yaw = math.degrees(math.atan2(front_vec[1], front_vec[0]))
    yaw = (yaw + 180) % 360 - 180
    poly = parallelogram_from_triangle(p0, p1, p2)
    edges = [np.linalg.norm(poly[(i+1)%4]-poly[i]) for i in range(4)]
    length = max(edges); width = min(edges); center = poly.mean(axis=0)
    front_edge = (p1, p2)
    return (float(center[0]), float(center[1])), float(length), float(width), float(yaw), front_edge

def write_bev_labels(save_path: str, bev_dets: List[dict]):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    for det in bev_dets:
        cx, cy = det["center"]; length = det["length"]; width = det["width"]; yaw = det["yaw"]
        lines.append(f"0 {cx:.4f} {cy:.4f} {length:.4f} {width:.4f} {yaw:.2f}")
    with open(save_path, "w") as f:
        f.write("\n".join(lines))

def _prepare_bev_canvas(polygons: List[np.ndarray], padding: float = 1.0, target: float = 800.0):
    if not polygons: return None
    pts = np.concatenate(polygons, axis=0)
    if pts.size == 0: return None
    min_x = float(np.nanmin(pts[:, 0]) - padding)
    max_x = float(np.nanmax(pts[:, 0]) + padding)
    min_y = float(np.nanmin(pts[:, 1]) - padding)
    max_y = float(np.nanmax(pts[:, 1]) + padding)
    range_x = max(max_x - min_x, 1e-3)
    range_y = max(max_y - min_y, 1e-3)
    scale = target / max(range_x, range_y)
    width = int(max(range_x * scale, 300))
    height = int(max(range_y * scale, 300))
    return {"min_x":min_x,"max_x":max_x,"min_y":min_y,"max_y":max_y,"scale":scale,"width":width,"height":height}

def _to_canvas(points: np.ndarray, params: dict) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xs = (pts[:, 0] - params["min_x"]) * params["scale"]
    ys = (params["max_y"] - pts[:, 1]) * params["scale"]
    return np.stack([xs, ys], axis=1).astype(np.int32)

def draw_bev_visualization(preds_bev: List[dict], gt_tris_bev: Optional[np.ndarray], save_path_img: str, title: str):
    pred_polys = [det["poly"] for det in preds_bev]
    gt_polys = [poly_from_tri(tri) for tri in gt_tris_bev] if gt_tris_bev is not None else []
    polygons = pred_polys + gt_polys
    params = _prepare_bev_canvas(polygons)
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)

    if params is None:
        canvas = np.full((360, 360, 3), 240, dtype=np.uint8)
        cv2.putText(canvas, "No BEV data", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(save_path_img, canvas); return

    if _MATPLOTLIB_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=220)
            ax.set_facecolor("#f7f7f7"); ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.set_aspect("equal", adjustable="box"); ax.set_title(title, fontsize=11, pad=10)
            ax.set_xlabel("X (m)", fontsize=10); ax.set_ylabel("Y (m)", fontsize=10)
            ax.set_xlim(params["min_x"], params["max_x"]); ax.set_ylim(params["max_y"], params["min_y"])
            if params["min_x"] <= 0.0 <= params["max_x"]: ax.axvline(0.0, color="#999999", linewidth=0.9, linestyle="--", alpha=0.8)
            if params["min_y"] <= 0.0 <= params["max_y"]: ax.axhline(0.0, color="#999999", linewidth=0.9, linestyle="--", alpha=0.8)
            legend_handles = []
            if gt_polys:
                for poly in gt_polys:
                    if not np.all(np.isfinite(poly)): continue
                    patch = MplPolygon(poly, closed=True, fill=False, edgecolor="#27ae60", linewidth=1.8)
                    ax.add_patch(patch); ax.scatter(poly[:, 0], poly[:, 1], s=8, color="#27ae60", alpha=0.9)
                legend_handles.append(Line2D([0], [0], color="#27ae60", lw=2, label="GT"))
            if preds_bev:
                dy = max(params["max_y"] - params["min_y"], 1e-3)
                text_offset = 0.015 * dy
                for det in preds_bev:
                    poly = det["poly"]
                    if not np.all(np.isfinite(poly)): continue
                    patch = MplPolygon(poly, closed=True, fill=False, edgecolor="#e74c3c", linewidth=1.8)
                    ax.add_patch(patch); center = det["center"]
                    ax.scatter(center[0], center[1], s=20, color="#e74c3c", alpha=0.9)
                    f1, f2 = det.get("front_edge", (None, None))
                    if f1 is not None and f2 is not None and np.all(np.isfinite(f1)) and np.all(np.isfinite(f2)):
                        ax.plot([f1[0], f2[0]], [f1[1], f2[1]], linewidth=2.2, color="#1f77b4")
                        front_center = (np.asarray(f1) + np.asarray(f2)) / 2.0
                        ax.annotate("", xy=(front_center[0], front_center[1]), xytext=(center[0], center[1]),
                                    arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#1f77b4"))
                    label = f"{det['score']:.2f} / {det['yaw']:.1f}°"
                    ax.text(center[0], center[1] + text_offset, label, fontsize=7.5, color="#e74c3c",
                            ha="center", va="bottom", bbox=dict(facecolor="#ffffff", alpha=0.6, edgecolor="none", pad=1.5))
                legend_handles.append(Line2D([0], [0], color="#e74c3c", lw=2, label="Pred"))
                legend_handles.append(Line2D([0], [0], color="#1f77b4", lw=3, label="Front edge"))
            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.75, fontsize=8)
            fig.tight_layout(pad=0.6); fig.savefig(save_path_img, dpi=220); plt.close(fig); return
        except Exception:
            plt.close("all")

    # OpenCV fallback
    canvas = np.full((params["height"], params["width"], 3), 255, dtype=np.uint8)
    axis_color = (120, 120, 120); axis_thickness = 1
    if params["min_x"] <= 0.0 <= params["max_x"]:
        axis_x = np.array([[0.0, params["min_y"]], [0.0, params["max_y"]]], dtype=np.float64)
        axis_x_px = _to_canvas(axis_x, params)
        cv2.line(canvas, tuple(axis_x_px[0]), tuple(axis_x_px[1]), axis_color, axis_thickness, cv2.LINE_AA)
    if params["min_y"] <= 0.0 <= params["max_y"]:
        axis_y = np.array([[params["min_x"], 0.0], [params["max_x"], 0.0]], dtype=np.float64)
        axis_y_px = _to_canvas(axis_y, params)
        cv2.line(canvas, tuple(axis_y_px[0]), tuple(axis_y_px[1]), axis_color, axis_thickness, cv2.LINE_AA)

    for det in preds_bev:
        poly = det["poly"]
        if not np.all(np.isfinite(poly)): continue
        poly_px = _to_canvas(poly, params)
        cv2.polylines(canvas, [poly_px], True, (0, 0, 255), 2)
        center_px = _to_canvas(np.asarray(det["center"]).reshape(1, 2), params)[0]
        cv2.circle(canvas, tuple(center_px), 4, (0, 0, 255), -1)
        f1, f2 = det.get("front_edge", (None, None))
        if f1 is not None and f2 is not None and np.all(np.isfinite(f1)) and np.all(np.isfinite(f2)):
            fpx = _to_canvas(np.vstack([f1, f2]), params)
            cv2.line(canvas, tuple(fpx[0]), tuple(fpx[1]), (255, 100, 0), 3, cv2.LINE_AA)
            front_center = (np.asarray(f1) + np.asarray(f2)) / 2.0
            fcp = _to_canvas(front_center.reshape(1,2), params)[0]
            cv2.arrowedLine(canvas, tuple(center_px), tuple(fcp), (255, 100, 0), 2, tipLength=0.18)

    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    coord_text = f"x:[{params['min_x']:.2f},{params['max_x']:.2f}]  y:[{params['min_y']:.2f},{params['max_y']:.2f}]"
    cv2.putText(canvas, coord_text, (10, params["height"] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.imwrite(save_path_img, canvas)

def evaluate_single_image_bev(preds_bev: List[dict], gt_tris_bev: np.ndarray, iou_thr=0.5):
    gt_arr = np.asarray(gt_tris_bev)
    num_gt = gt_arr.shape[0] if gt_arr.ndim == 3 else 0
    preds_sorted = sorted(preds_bev, key=lambda d: d["score"], reverse=True)
    if num_gt == 0:
        records = [(det["score"], 0, 0.0, None) for det in preds_sorted]
        return records, 0
    gt_polys = [poly_from_tri(tri) for tri in gt_arr]
    matched = np.zeros((num_gt,), dtype=bool)
    records = []
    for det in preds_sorted:
        poly_d = det["poly"]
        if not np.all(np.isfinite(poly_d)):
            records.append((det["score"], 0, 0.0, None)); continue
        best_iou, best_idx = 0.0, -1
        for idx, poly_g in enumerate(gt_polys):
            if matched[idx] or not np.all(np.isfinite(poly_g)): continue
            iou = iou_polygon(poly_d, poly_g)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0 and best_iou >= iou_thr:
            matched[best_idx] = True
            orient_err = orientation_error_deg(det["tri"], gt_arr[best_idx])
            records.append((det["score"], 1, best_iou, orient_err))
        else:
            records.append((det["score"], 0, best_iou, None))
    return records, int(matched.sum())

# =========================
# ONNX temporal runner (고쳐진 버전: 입력 메타에서 zero-state 생성)
# =========================
class ONNXTemporalRunner:
    """
    ConvLSTM/GRU가 들어간 ONNX 모델을 실행.
    - 입력: images (1,3,H,W)  + (선택) h_in[, c_in]
    - 출력: 스케일별 reg/obj/cls + (선택) h_out[, c_out]
    입출력 이름은 세션에서 자동으로 탐색, state shape은 '입력 메타'에서 바로 생성(부트스트랩 실행 없음).
    """
    def __init__(self, onnx_path, providers=("CUDAExecutionProvider","CPUExecutionProvider"),
                 state_stride_hint: int = 32, default_hidden_ch: int = 256):
        self.sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.inputs  = {i.name:i for i in self.sess.get_inputs()}
        self.outs    = [o.name for o in self.sess.get_outputs()]

        # 입력 이름 추론
        cand_x = [n for n in self.inputs if n.lower() in ("images","image","input")]
        self.x_name = cand_x[0] if cand_x else list(self.inputs.keys())[0]
        self.h_name = next((n for n in self.inputs if "h_in" in n.lower()), None)
        self.c_name = next((n for n in self.inputs if "c_in" in n.lower()), None)

        # 출력 이름 그룹
        self.ho_name = next((n for n in self.outs if "h_out" in n.lower()), None)
        self.co_name = next((n for n in self.outs if "c_out" in n.lower()), None)
        self.reg_names = [n for n in self.outs if "reg" in n.lower()]
        self.obj_names = [n for n in self.outs if "obj" in n.lower()]
        self.cls_names = [n for n in self.outs if "cls" in n.lower()]

        def _sort_key(s):
            toks = []
            acc = ""
            for ch in s:
                if ch.isdigit():
                    acc += ch
                else:
                    if acc:
                        toks.append(int(acc)); acc = ""
                    toks.append(ch)
            if acc: toks.append(int(acc))
            return tuple(toks)

        self.reg_names.sort(key=_sort_key)
        self.obj_names.sort(key=_sort_key)
        self.cls_names.sort(key=_sort_key)

        # 상태 버퍼
        self.h_buf = None
        self.c_buf = None

        # 상태 shape 메타
        self.state_stride_hint = int(state_stride_hint)
        self.default_hidden_ch = int(default_hidden_ch)
        self.h_shape_meta = self._shape_from_input_meta(self.h_name)
        self.c_shape_meta = self._shape_from_input_meta(self.c_name)

    def _shape_from_input_meta(self, name):
        if name is None: return None
        meta = self.inputs[name].shape  # [N,C,Hs,Ws] with possible None/str
        def _to_int(val, default):
            return int(val) if isinstance(val, (int, np.integer)) else default
        N = _to_int(meta[0], 1)
        C = _to_int(meta[1], self.default_hidden_ch)
        Hs= _to_int(meta[2], 0)
        Ws= _to_int(meta[3], 0)
        return [N, C, Hs, Ws]

    def reset(self):
        self.h_buf = None
        self.c_buf = None

    def _ensure_state(self, img_numpy_chw: np.ndarray):
        # img: (1,3,H,W)
        _, _, H, W = img_numpy_chw.shape
        if self.h_name and self.h_buf is None:
            N, C, Hs, Ws = self.h_shape_meta
            if Hs == 0 or Ws == 0:
                Hs = max(1, H // self.state_stride_hint)
                Ws = max(1, W // self.state_stride_hint)
            self.h_buf = np.zeros((N, C, Hs, Ws), dtype=np.float32)
        if self.c_name and self.c_buf is None:
            N, C, Hs, Ws = self.c_shape_meta
            if Hs == 0 or Ws == 0:
                Hs = max(1, H // self.state_stride_hint)
                Ws = max(1, W // self.state_stride_hint)
            self.c_buf = np.zeros((N, C, Hs, Ws), dtype=np.float32)

    def forward(self, img_numpy_chw):
        """
        img_numpy_chw: (1,3,H,W) float32 [0..1]
        returns: list of (reg,obj,cls) as torch.Tensors (1,C,Hs,Ws)
        """
        self._ensure_state(img_numpy_chw)

        feeds = { self.x_name: img_numpy_chw }
        if self.h_name is not None and self.h_buf is not None:
            feeds[self.h_name] = self.h_buf
        if self.c_name is not None and self.c_buf is not None:
            feeds[self.c_name] = self.c_buf

        outs = self.sess.run(self.outs, feeds)
        out_map = {n:v for n,v in zip(self.outs, outs)}

        # 상태 갱신
        if self.ho_name: self.h_buf = out_map[self.ho_name]
        if self.co_name: self.c_buf = out_map[self.co_name]

        # PyTorch 디코더와 호환되는 포맷(list of (reg,obj,cls) torch.Tensor)
        pred_list = []
        for rn, on, cn in zip(self.reg_names, self.obj_names, self.cls_names):
            pr = torch.from_numpy(out_map[rn])
            po = torch.from_numpy(out_map[on])
            pc = torch.from_numpy(out_map[cn])
            pred_list.append((pr, po, pc))
        return pred_list

# =========================
# 시퀀스 키
# =========================
def seq_key(file_path: str, mode: str) -> str:
    p = Path(file_path)
    if mode == "by_subdir":
        return p.parent.name
    stem = p.stem
    if "_" in stem: return stem.split("_")[0]
    if "-" in stem: return stem.split("-")[0]
    return "ALL"

# =========================
# 메인
# =========================
def main():
    ap = argparse.ArgumentParser("YOLO11 2.5D ONNX Temporal Inference (+GT & BEV)")
    ap.add_argument("--input-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default="./inference_results_onnx")
    ap.add_argument("--weights", type=str, required=True, help="ONNX 파일 경로")
    ap.add_argument("--img-size", type=str, default="864,1536")
    ap.add_argument("--score-mode", type=str, default="obj*cls", choices=["obj","cls","obj*cls"])
    ap.add_argument("--conf", type=float, default=0.80)
    ap.add_argument("--nms-iou", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--contain-thr", type=float, default=0.85)
    ap.add_argument("--clip-cells", type=float, default=None)
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png")

    # 디코더에 필요한 stride (ONNX엔 저장 안돼 있을 수 있으므로 인자로 받음)
    ap.add_argument("--strides", type=str, default="8,16,32")

    # temporal & sequence
    ap.add_argument("--temporal", type=str, default="lstm", choices=["none","gru","lstm"])
    ap.add_argument("--seq-mode", type=str, default="by_prefix", choices=["by_prefix","by_subdir"])
    ap.add_argument("--reset-per-seq", action="store_true", default=True)

    # state feature map 추정용 힌트(마지막 스케일 stride 추정값)
    ap.add_argument("--state-stride-hint", type=int, default=32)
    ap.add_argument("--default-hidden-ch", type=int, default=256)

    # BEV & Eval
    ap.add_argument("--gt-label-dir", type=str, default=None)
    ap.add_argument("--eval-iou-thr", type=float, default=0.5)
    ap.add_argument("--labels-are-original-size", action="store_true", default=True)
    ap.add_argument("--calib-dir", type=str, default=None)
    ap.add_argument("--invert-calib", action="store_true")
    ap.add_argument("--bev-scale", type=float, default=1.0)

    # onnxruntime providers
    ap.add_argument("--no-cuda", action="store_true", help="CUDA EP 비활성화")

    args = ap.parse_args()
    H, W = map(int, args.img_size.split(","))
    strides = [float(s) for s in args.strides.split(",")]

    providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    if args.no_cuda or ort.get_device().upper() != "GPU":
        providers = ["CPUExecutionProvider"]

    runner = ONNXTemporalRunner(
        args.weights, providers=providers,
        state_stride_hint=args.state_stride_hint,
        default_hidden_ch=args.default_hidden_ch
    )

    out_img_dir = os.path.join(args.output_dir, "images")
    out_lab_dir = os.path.join(args.output_dir, "labels")
    out_mix_dir = os.path.join(args.output_dir, "images_with_gt")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)
    os.makedirs(out_mix_dir, exist_ok=True)

    use_bev = args.calib_dir is not None and os.path.isdir(args.calib_dir)
    if use_bev:
        out_bev_img_dir = os.path.join(args.output_dir, "bev_images")
        out_bev_lab_dir = os.path.join(args.output_dir, "bev_labels")
        out_bev_mix_dir = os.path.join(args.output_dir, "bev_images_with_gt")
        os.makedirs(out_bev_img_dir, exist_ok=True)
        os.makedirs(out_bev_lab_dir, exist_ok=True)
        os.makedirs(out_bev_mix_dir, exist_ok=True)
        homography_cache = {}
        missing_h_names = set()
    else:
        out_bev_img_dir = out_bev_lab_dir = out_bev_mix_dir = None
        homography_cache = {}
        missing_h_names = set()

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    names = [f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)]
    names.sort()

    do_eval_2d = args.gt_label_dir is not None and os.path.isdir(args.gt_label_dir)

    metric_records = []
    total_gt = 0
    metric_records_bev = []
    total_gt_bev = 0

    print(f"[Infer-ONNX] imgs={len(names)}, temporal={args.temporal}, seq={args.seq_mode}, reset_per_seq={args.reset_per_seq}, eval2D={do_eval_2d}, evalBEV={use_bev}")

    prev_key = None
    for name in tqdm(names, desc="[Infer-ONNX]"):
        path = os.path.join(args.input_dir, name)

        # 시퀀스 경계 판단 → reset
        k = seq_key(path, args.seq_mode)
        if args.reset_per_seq and k != prev_key:
            runner.reset()
        prev_key = k

        img_bgr0 = cv2.imread(path)
        if img_bgr0 is None:
            continue

        H0, W0 = img_bgr0.shape[:2]
        img_bgr = cv2.resize(img_bgr0, (W, H), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_np = img_rgb.transpose(2,0,1).astype(np.float32) / 255.0
        img_np = np.expand_dims(img_np, 0)  # (1,3,H,W)

        scale_resize_x = W / float(W0)
        scale_resize_y = H / float(H0)
        scale_to_orig_x = float(W0) / float(W)
        scale_to_orig_y = float(H0) / float(H)

        outs = runner.forward(img_np)

        # decode (GPU NMS 미사용)
        dets = decode_predictions(
            outs, strides,
            clip_cells=args.clip_cells,
            conf_th=args.conf,
            nms_iou=args.nms_iou,
            topk=args.topk,
            contain_thr=args.contain_thr,
            score_mode=args.score_mode,
            use_gpu_nms=True
        )[0]

        dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)

        save_img = os.path.join(out_img_dir, name)
        save_txt = os.path.join(out_lab_dir, os.path.splitext(name)[0] + ".txt")
        pred_tris_orig = draw_pred_only(img_bgr, dets, save_img, save_txt, W, H, W0, H0)

        gt_for_eval = np.zeros((0, 3, 2), dtype=np.float32)
        gt_tri_orig_for_bev = np.zeros((0, 3, 2), dtype=np.float32)

        if do_eval_2d:
            lab_path = os.path.join(args.gt_label_dir, os.path.splitext(name)[0] + ".txt")
            gt_tri_raw = load_gt_triangles(lab_path)
            if gt_tri_raw.shape[0] > 0:
                if args.labels_are_original_size:
                    gt_tri_orig_for_bev = gt_tri_raw.astype(np.float32)
                    gt_for_eval = gt_tri_raw.copy()
                    gt_for_eval[:, :, 0] *= scale_resize_x
                    gt_for_eval[:, :, 1] *= scale_resize_y
                else:
                    gt_for_eval = gt_tri_raw.astype(np.float32)
                    gt_tri_orig_for_bev = gt_tri_raw.copy()
                    gt_tri_orig_for_bev[:, :, 0] *= scale_to_orig_x
                    gt_tri_orig_for_bev[:, :, 1] *= scale_to_orig_y

            save_img_mix = os.path.join(out_mix_dir, name)
            draw_pred_with_gt(img_bgr, dets, gt_for_eval, save_img_mix, iou_thr=args.eval_iou_thr)

            if gt_for_eval.shape[0] > 0:
                records, _ = evaluate_single_image(dets, gt_for_eval, iou_thr=args.eval_iou_thr)
                metric_records.extend(records)
                total_gt += gt_for_eval.shape[0]

        # ---- BEV (옵션) ----
        if use_bev:
            H_img2ground = load_homography(args.calib_dir, name, homography_cache, invert=args.invert_calib)
            if H_img2ground is None:
                missing_h_names.add(os.path.splitext(name)[0])
            else:
                bev_dets = []
                if pred_tris_orig:
                    pred_stack_orig = np.asarray(pred_tris_orig, dtype=np.float64)
                    pred_tris_bev = apply_homography(pred_stack_orig, H_img2ground)
                    pred_tris_bev = pred_tris_bev * float(args.bev_scale)
                    for det, tri_bev in zip(dets, pred_tris_bev):
                        if not np.all(np.isfinite(tri_bev)): continue
                        poly_bev = poly_from_tri(tri_bev)
                        props = compute_bev_properties(tri_bev)
                        if props is None: continue
                        center, length, width, yaw, front_edge = props
                        bev_dets.append({
                            "score": float(det["score"]),
                            "tri": tri_bev,
                            "poly": poly_bev,
                            "center": center,
                            "length": length,
                            "width": width,
                            "yaw": yaw,
                            "front_edge": front_edge,
                        })

                if gt_tri_orig_for_bev.size > 0:
                    gt_tris_bev = apply_homography(gt_tri_orig_for_bev.astype(np.float64), H_img2ground)
                    gt_tris_bev = gt_tris_bev * float(args.bev_scale)
                else:
                    gt_tris_bev = np.zeros((0, 3, 2), dtype=np.float64)

                bev_img_path = os.path.join(out_bev_img_dir, name)
                draw_bev_visualization(bev_dets, None, bev_img_path, f"{name} | Pred BEV")

                bev_mix_path = os.path.join(out_bev_mix_dir, name)
                draw_bev_visualization(bev_dets, gt_tris_bev, bev_mix_path, f"{name} | Pred & GT BEV")

                bev_label_path = os.path.join(out_bev_lab_dir, os.path.splitext(name)[0] + ".txt")
                write_bev_labels(bev_label_path, bev_dets)

                if gt_tris_bev.size > 0:
                    records_bev, _ = evaluate_single_image_bev(bev_dets, gt_tris_bev, iou_thr=args.eval_iou_thr)
                    metric_records_bev.extend(records_bev)
                    total_gt_bev += gt_tris_bev.shape[0]

    # ---- 전체 메트릭 출력 ----
    if do_eval_2d:
        metrics = compute_detection_metrics(metric_records, total_gt)
        print("== 2D Eval (dataset-wide) ==")
        print("Precision:  {:.4f}".format(metrics["precision"]))
        print("Recall:     {:.4f}".format(metrics["recall"]))
        print("mAP@50:     {:.4f}".format(metrics["map50"]))
        print("mAOE(deg):  {:.2f}".format(metrics["mAOE_deg"]))

    if use_bev:
        metrics_bev = compute_detection_metrics(metric_records_bev, total_gt_bev)
        if total_gt_bev > 0 or metric_records_bev:
            print("== BEV Eval (dataset-wide) ==")
            print("Precision:  {:.4f}".format(metrics_bev["precision"]))
            print("Recall:     {:.4f}".format(metrics_bev["recall"]))
            print("APbev@50:   {:.4f}".format(metrics_bev["map50"]))
            print("mAOE_bev:   {:.2f}".format(metrics_bev["mAOE_deg"]))
        else:
            print("[Info] BEV 평가는 GT 또는 유효한 H 행렬이 부족해 계산하지 않았습니다.")

    print("Done.")

if __name__ == "__main__":
    main()
"""
python ./src/inference_lstm_onnx.py \
  --input-dir ./dataset_example/images \
  --output-dir ./inference_results_onnx \
  --weights ./onnx/yolo11m_2_5d_epoch_005.onnx \
  --temporal lstm \
  --seq-mode by_prefix --reset-per-seq \
  --conf 0.8 --nms-iou 0.2 --topk 50 \
  --gt-label-dir ./dataset_example/labels \
  --calib-dir ./dataset_example/calib
"""