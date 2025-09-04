"""
Video denoising inference script with selectable evaluator.
 - Set EVALUATOR to 1,2,3 or 4 in config.py to choose which evaluator runs.
 - The list of input files is taken from config.FILES and they are searched
inside md.PATH_TO_INFER.
 - Outputs are written into md.PATH_INFERED with per-file suffixes indicating
the evaluator used.
"""

import os, sys

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

# To import local modules.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import metadata as md
import config as conf


##############################################################################
# Evaluator utils.

def read_all_frames(path):
    """Read all frames from an mp4 and return list of RGB uint8 numpy arrays
    and fps.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames, fps

def frames_to_batch_tensor(frames_rgb, device):
    """Convert a list of T frames (H,W,3) uint8 RGB to a torch tensor of shape
    (1,3,T,H,W), float32 normalized to [0,1], on the given device.
    """
    arr = np.stack(frames_rgb, axis=0)             # (T, H, W, 3)
    arr = arr.astype(np.float32) / 255.0           # (T, H, W, 3) float32
    arr = np.transpose(arr, (3, 0, 1, 2))          # (3, T, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0)    # (1, 3, T, H, W)
    return tensor.to(device)

def batch_tensor_to_rgb_uint8(batch_tensor):
    """Convert tensor (1,3,T,H,W) or (3,T,H,W) or (T,3,H,W) with values
    assumed in [0,1] to a list of RGB uint8 numpy arrays (length T).
    """
    t = batch_tensor.detach().cpu()
    if t.dim() == 5:
        t = t.squeeze(0) # -> (3,T,H,W)
    if t.dim() == 4:
        t = t.permute(1, 2, 3, 0).numpy() # (3,T,H,W) -> (T, H, W, 3)
    elif t.dim() == 3:
        t = np.transpose(t.numpy(), (0, 1, 2))
    else:
        raise ValueError("Unexpected tensor dims")
    t = np.clip(t, 0.0, 1.0)
    t_uint8 = (t * 255.0).astype(np.uint8) # (T, H, W, 3) RGB
    return [t_uint8[i] for i in range(t_uint8.shape[0])]

def write_rgb_frames_to_mp4(frames_rgb, fps, out_path):
    """Write a list of RGB uint8 numpy arrays to an mp4 file (BGR writer).
    """
    if len(frames_rgb) == 0:
        raise ValueError("No frames to write.")
    H, W, _ = frames_rgb[0].shape
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for f_rgb in frames_rgb:
        f_bgr = cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR)
        writer.write(f_bgr)
    writer.release()

def infer_window(model, window_frames_rgb, device):
    """Run the model on a window (list of exactly 4 RGB frames).
    Returns a list of 4 RGB uint8 numpy arrays (model output), values assumed
    in [0,1] before conversion.
    """
    # Ensure window length is 4 by padding with the last frame if necessary.
    if len(window_frames_rgb) < 4:
        last = window_frames_rgb[-1]
        pad = 4 - len(window_frames_rgb)
        window_frames_rgb = window_frames_rgb + [last] * pad
    batch = frames_to_batch_tensor(window_frames_rgb, device) # (1,3,4,H,W)
    with torch.no_grad():
        out = model(batch)    # expect (1,3,4,H,W)
    out = torch.clamp(out, 0.0, 1.0)
    out_frames = batch_tensor_to_rgb_uint8(out)
    return out_frames


##############################################################################
# Evaluators.

def block_evaluator(model, frames, device):
    """Evaluator 1: Non-overlapping blocks of 4.
    If remaining frames < 4 at the end:
      - evaluate the last 4 frames (window starting at N-4),
      - append only the missing tail frames to the output sequence.
    Returns a list of RGB uint8 frames of length equal to original
    number of frames.
    """
    N = len(frames)
    if N == 0:
        return []
    out_seq = [None] * N

    # Process full non-overlapping blocks
    i = 0
    while i + 4 <= N:
        window = frames[i:i+4]
        out_frames = infer_window(model, window, device)
        for t in range(4):
            out_seq[i + t] = out_frames[t]
        i += 4

    # If there are leftover frames at the end (i < N), evaluate last 4 frames
    # and append only missing ones.
    if i < N:
        start = max(0, N - 4)
        last_window = frames[start:start+4]
        out_last = infer_window(model, last_window, device)
        for idx in range(i, N):
            rel = idx - start
            out_seq[idx] = out_last[rel]
    
    return out_seq

def sliding_center_evaluator(model, frames, device):
    """Evaluator 2: sliding windows with stride=2. For each window we keep
    only the two central output frames. Mapping rule: for a window starting
    at s, the model outputs frames out[0..3]; we map:
        out[1] -> index s
        out[2] -> index s + 1
    The first frame of the video is guaranteed to be produced by the very
    first inference:
      - if a valid window starting at 0 exists, it is used;
      - otherwise a padded window starting at 0 is run first (so out[1]
      supplies frame 0).
    Tail frames are fixed by running an inference on the last 4 frames (if any
    positions remain unset).
    """
    N = len(frames)
    out_seq = [None] * N

    starts = list(range(0, N, 2))
    valid_starts = [s for s in starts if s + 4 <= N]

    processed_starts = set()

    # Ensure we always process a window starting at 0 first (padded if needed)
    # so frame 0 comes from first inference.
    if 0 not in valid_starts:
        win0 = frames[0:4]
        out0 = infer_window(model, win0, device)
        if 0 < N:
            out_seq[0] = out0[1]  # out[1] -> index 0
        if 1 < N:
            out_seq[1] = out0[2]  # out[2] -> index 1
        processed_starts.add(0)

    # Process remaining valid starts (skip 0 if we already processed it).
    for s in tqdm(valid_starts, desc="sliding_center eval", leave=False):
        if s in processed_starts:
            continue
        window = frames[s:s+4]
        out_frames = infer_window(model, window, device)
        if s < N:
            out_seq[s] = out_frames[1]
        if (s + 1) < N:
            out_seq[s + 1] = out_frames[2]

    # Fix tail: if some frames at the end are still None, evaluate last 4
    # frames and fill missing tail frames.
    first_missing = None
    for idx in range(N):
        if out_seq[idx] is None:
            first_missing = idx
            break
    if first_missing is not None:
        start = max(0, N - 4)
        out_last = infer_window(model, frames[start:start+4], device)
        for idx in range(first_missing, N):
            rel = idx - start
            out_seq[idx] = out_last[rel]

    return out_seq

def sliding_mean_stride2_evaluator(model, frames, device):
    """Evaluator 3: sliding windows with stride=2. For each window starting
    at s, map output frame t to original index s+t. Accumulate sums and counts
    for each original frame index and average at the end. If there are indexes
    never predicted (due to edge rounding), evaluate last 4 frames and include
    them.
    """
    N = len(frames)
    if N == 0:
        return []
    acc_sum = [None] * N
    acc_count = [0] * N

    starts = list(range(0, N, 2))
    valid_starts = [s for s in starts if s + 4 <= N]

    for s in tqdm(
        valid_starts, desc="sliding_mean_stride2 eval", leave=False
    ):
        out_frames = infer_window(model, frames[s:s+4], device)
        for t in range(4):
            idx = s + t
            if idx >= N:
                continue
            arr = out_frames[t].astype(np.float32) / 255.0
            if acc_sum[idx] is None:
                acc_sum[idx] = arr
            else:
                acc_sum[idx] = acc_sum[idx] + arr
            acc_count[idx] += 1

    # Tail handling: if some indices have zero count, run last-4 inference
    # and add them.
    start = max(0, N - 4)
    out_last = infer_window(model, frames[start:start+4], device)
    for t in range(4):
        idx = start + t
        if idx >= N:
            continue
        arr = out_last[t].astype(np.float32) / 255.0
        if acc_sum[idx] is None:
            acc_sum[idx] = arr
        else:
            acc_sum[idx] = acc_sum[idx] + arr
        acc_count[idx] += 1

    # Build averaged frames. If for some reason count==0,
    # fallback to original frame.
    out_seq = []
    for i in range(N):
        if acc_count[i] == 0 or acc_sum[i] is None:
            out_seq.append(frames[i])
        else:
            avg = acc_sum[i] / float(acc_count[i])
            avg_uint8 = np.clip(avg * 255.0, 0, 255).astype(np.uint8)
            out_seq.append(avg_uint8)
    
    return out_seq

def sliding_mean_stride1_evaluator(model, frames, device):
    """Evaluator 4: sliding windows with stride=1. For every window starting
    at s (0..N-4) evaluate and map output t->s+t, accumulate and average all
    available predictions per original frame.
    """
    N = len(frames)
    if N == 0:
        return []
    acc_sum = [None] * N
    acc_count = [0] * N

    starts = list(range(0, max(1, N)))
    valid_starts = [s for s in starts if s + 4 <= N]

    for s in tqdm(
        valid_starts, desc="sliding_mean_stride1 eval", leave=False
    ):
        out_frames = infer_window(model, frames[s:s+4], device)
        for t in range(4):
            idx = s + t
            if idx >= N:
                continue
            arr = out_frames[t].astype(np.float32) / 255.0
            if acc_sum[idx] is None:
                acc_sum[idx] = arr
            else:
                acc_sum[idx] = acc_sum[idx] + arr
            acc_count[idx] += 1

    # Edge case: if N < 4 or some frames uncovered, run last-4 and include.
    if any(c == 0 for c in acc_count):
        start = max(0, N - 4)
        out_last = infer_window(model, frames[start:start+4], device)
        for t in range(4):
            idx = start + t
            if idx >= N:
                continue
            arr = out_last[t].astype(np.float32) / 255.0
            if acc_sum[idx] is None:
                acc_sum[idx] = arr
            else:
                acc_sum[idx] = acc_sum[idx] + arr
            acc_count[idx] += 1

    # Average and fallback.
    out_seq = []
    for i in range(N):
        if acc_count[i] == 0 or acc_sum[i] is None:
            out_seq.append(frames[i])
        else:
            avg = acc_sum[i] / float(acc_count[i])
            avg_uint8 = np.clip(avg * 255.0, 0, 255).astype(np.uint8)
            out_seq.append(avg_uint8)
    
    return out_seq

# Map evaluator id to function and suffix.
EVALUATOR_MAP = {
    1: (block_evaluator, "eval1_block"),
    2: (sliding_center_evaluator, "eval2_sliding_center"),
    3: (sliding_mean_stride2_evaluator, "eval3_sliding_mean_s2"),
    4: (sliding_mean_stride1_evaluator, "eval4_sliding_mean_s1"),
}


if __name__ == "__main__":
    device = md.DEVICE
    model = conf.MODEL(conf.BASE_CHANNELS).to(device)
    ckpt = torch.load(
        os.path.join(md.PATH_MODELS, conf.NAME_INF + ".pth"),
        map_location=device
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    evaluator_fn, suffix = EVALUATOR_MAP[conf.EVALUATOR]

    # Ensure output directory exists.
    out_dir = md.PATH_INFERED
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over files listed in conf.FILES.
    files = getattr(conf, "FILES", [])

    for fname in tqdm(files, desc="files", leave=True):
        input_path = os.path.join(md.PATH_TO_INFER, fname)
        if not os.path.isfile(input_path):
            print(f"Skipping missing file: {input_path}")
            continue

        print(f"Processing file: {fname}")
        frames, fps = read_all_frames(input_path)
        N = len(frames)
        print(f"Read {N} frames at {fps:.2f} fps. Device: {device}")

        # Run selected evaluator.
        out_frames = evaluator_fn(model, frames, device)

        # Compose output path and save.
        base = os.path.splitext(os.path.basename(fname))[0]
        out_path = os.path.join(out_dir, f"{base}_{suffix}.mp4")
        write_rgb_frames_to_mp4(out_frames, fps, out_path)
        print(f"Saved output to: {out_path}")

    print("Inference is over.")
