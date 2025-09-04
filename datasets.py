"""
Script with the torch dataset classes used to load PVDD data and create
correct data structures for training and evaluating.
"""

import os
import glob
import re
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class Train3DDataset(Dataset):
    """
    Dataset class to train the 3D models using PVDD which takes 30//"clip_len"
    non-overlapping clips of length "clip_len" for each 30-frame block and
    loads them on demand.
    """
    def __init__(
        self, root_dir: str, clip_len: int, transform=None
    ):
        self.clip_len  = clip_len
        self.transform = transform or T.ToTensor()

        # Collect all clean/noisy pairs' paths.
        ext = "png"
        clean_files = glob.glob(
            os.path.join(root_dir, 'clean', f'clean_*.{ext}')
        )
        noisy_files = glob.glob(
            os.path.join(root_dir, 'noisy', f'noisy_*.{ext}')
        )

        # Group by video and frame_group.
        seqs = {}
        for p in clean_files:
            fn = os.path.basename(p)
            _, vid, grp, idx_ext = fn.split('_')
            idx = int(idx_ext.split('.')[0])
            seqs.setdefault(
                (vid, grp), {'clean':[], 'noisy':[]}
            )['clean'].append((idx,p))
        for p in noisy_files:
            fn = os.path.basename(p)
            _, vid, grp, idx_ext = fn.split('_')
            idx = int(idx_ext.split('.')[0])
            key = (vid, grp)
            if key in seqs:
                seqs[key]['noisy'].append((idx,p))

        # Filter and sort all 30-frame valid sequences.
        self.samples = []
        for key, d in seqs.items():
            clean_list = sorted(d['clean'], key=lambda x: x[0])
            noisy_list = sorted(d['noisy'], key=lambda x: x[0])
            if len(clean_list)==30 and len(noisy_list)==30:
                clean_paths = [p for _,p in clean_list]
                noisy_paths = [p for _,p in noisy_list]
                self.samples.append((noisy_paths, clean_paths))
        assert self.samples, "There are no valid 30-frame sequences."

        # Non-overlapping offsets.
        self.offsets = list(range(0, 30 - clip_len + 1, clip_len))

    def __len__(self):
        return len(self.samples) * len(self.offsets)

    # Edited __getitem__ method in order to load frames on demand instead of
    # loading all of them in memory, which is impossible.
    def __getitem__(self, idx):
        clips_per_seq = len(self.offsets)
        seq_idx    = idx // clips_per_seq
        offset_idx = idx %  clips_per_seq
        off        = self.offsets[offset_idx]

        noisy_paths, clean_paths = self.samples[seq_idx]

        # Select clip.
        noisy_clip = noisy_paths[off: off + self.clip_len]
        clean_clip = clean_paths[off: off + self.clip_len]

        # Load on demand, transform and stack.
        noisy_imgs = [self.transform(Image.open(p)) for p in noisy_clip]
        clean_imgs = [self.transform(Image.open(p)) for p in clean_clip]
        noisy = torch.stack(noisy_imgs, dim=1)
        clean = torch.stack(clean_imgs, dim=1)

        return noisy, clean

class Test3DDataset(Dataset):
    """
    Dataset class to test the 3D models using PVDD's synthetic noise data,
    which takes all possible sequences with "clip_len" <=5 length given a
    noise level and loads them on demand.
    """
    def __init__(
        self, root_dir, level, clip_len, transform=None
    ):
        assert level in ('S','M','L')
        self.clip_len = clip_len
        self.transform = transform or T.ToTensor()

        noisy_dir = os.path.join(root_dir, 'noisy')
        clean_dir = os.path.join(root_dir, 'clean')
        pat = re.compile(r'^noisy_(T\d+)_(frame\d+)_(\d+)_([SML])\.png$')

        # Group by sequence, frame group and level.
        seqs = {}
        for fn in sorted(os.listdir(noisy_dir)):
            m = pat.match(fn)
            if not m: 
                continue
            vid, grp, idx_str, lvl = m.groups()
            if lvl != level:
                continue
            idx = int(idx_str)
            key = (vid, grp, lvl)
            seqs.setdefault(key, []).append((idx, fn))

        # For each valid sequence of "clip_len" length, create a clip.
        self.clips = []
        for (vid, grp, lvl), frames in seqs.items():
            if len(frames) != 5:
                continue
            frames = sorted(frames, key=lambda x: x[0])
            noisy_paths = [os.path.join(noisy_dir, fn) for _, fn in frames]
            clean_paths = [
                os.path.join(clean_dir, fn.replace('noisy_','clean_'))
                for _, fn in frames
            ]
            # Verify existence.
            for p in clean_paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(
                        f"Clean file: {p} does not exist."
                    )
            # Slide window.
            for off in range(5 - clip_len + 1):
                self.clips.append(
                    (
                        noisy_paths[off:off+clip_len],
                        clean_paths[off:off+clip_len]
                    )
                )

        # Check if there are any valid clips.
        assert self.clips, "No valid clips found."

    def __len__(self):
        return len(self.clips)

    # Edited __getitem__ method in order to load frames on demand instead of
    # loading all of them in memory, which is impossible.
    def __getitem__(self, idx):
        noisy_paths, clean_paths = self.clips[idx]
        noisy = [
            self.transform(Image.open(p).convert('RGB')) for p in noisy_paths
        ]
        clean = [
            self.transform(Image.open(p).convert('RGB')) for p in clean_paths
        ]
        noisy = torch.stack(noisy, dim=1)  # (C, clip_len, H, W)
        clean = torch.stack(clean, dim=1)
        return noisy, clean


class Train2DDataset(Dataset):
    """
    Dataset class to train the 2D models using PVDD which takes 30//"clip_len"
    non-overlapping noisy clips of length "clip_len" (it has to be odd) and
    its central clean frame for each 30-frame block and loads them on demand.
    """
    def __init__(self, root_dir: str, clip_len: int = 5, transform=None):
        super().__init__()
        assert clip_len % 2 == 1, "clip_len must be odd."
        self.clip_len = clip_len
        self.half = clip_len // 2
        self.transform = transform or T.ToTensor()

        # Get clean and noisy file lists.
        ext = 'png'
        clean_files = glob.glob(
            os.path.join(root_dir, 'clean', f'clean_*.{ext}')
        )
        noisy_files = glob.glob(
            os.path.join(root_dir, 'noisy', f'noisy_*.{ext}')
        )

        # Group by 30-frame sequences.
        seqs = {}
        for p in clean_files:
            fn = os.path.basename(p)
            _, vid, grp, idx_ext = fn.split('_')
            idx = int(idx_ext.split('.')[0])
            seqs.setdefault((vid, grp), {})['clean'] = seqs.get(
                (vid, grp), {}
            ).get('clean', []) + [(idx, p)]
        for p in noisy_files:
            fn = os.path.basename(p)
            _, vid, grp, idx_ext = fn.split('_')
            idx = int(idx_ext.split('.')[0])
            seqs.setdefault((vid, grp), {})['noisy'] = seqs.get(
                (vid, grp), {}
            ).get('noisy', []) + [(idx, p)]

        # Filter 30-frame sequences and generate non-overlapping offsets.
        self.samples = []
        offsets = list(range(0, 30 - clip_len + 1, clip_len))
        for key, data in seqs.items():
            clean_list = sorted(data.get('clean', []), key=lambda x: x[0])
            noisy_list = sorted(data.get('noisy', []), key=lambda x: x[0])
            if len(clean_list) == 30 and len(noisy_list) == 30:
                clean_paths = [p for _, p in clean_list]
                noisy_paths = [p for _, p in noisy_list]
                for off in offsets:
                    clip_noisy = noisy_paths[off:off + clip_len]
                    clip_clean = clean_paths[off:off + clip_len]
                    self.samples.append((clip_noisy, clip_clean))

    def __len__(self):
        return len(self.samples)

    # Edited __getitem__ method in order to load frames on demand instead of
    # loading all of them in memory, which is impossible.
    def __getitem__(self, idx):
        noisy_paths, clean_paths = self.samples[idx]
        # Load and transform.
        noisy_imgs = [self.transform(Image.open(p).convert('RGB')) for p in noisy_paths]
        clean_imgs = [self.transform(Image.open(p).convert('RGB')) for p in clean_paths]

        # Stack as (C, T, H, W).
        noisy = torch.stack(noisy_imgs, dim=1)
        clean = torch.stack(clean_imgs, dim=1)

        # Reshape noisy clips: (C*T, H, W).
        C, T, H, W = noisy.shape
        noisy = noisy.view(C * T, H, W)

        # Central clean frame: (C, H, W).
        central = clean[:, self.half, :, :]

        return noisy, central


class Test2DDataset(Dataset):
    """
    Dataset class to test the 2D models using PVDD's synthetic noise data,
    which takes all possible sequences with "clip_len" <=5 length given a
    noise level and loads them on demand.
    """
    def __init__(self, root_dir: str, level: str, clip_len: int = 5, transform=None):
        super().__init__()
        assert level in ('S', 'M', 'L'), "level must be 'S', 'M' o 'L'."
        assert clip_len % 2 == 1, "clip_len must be odd."
        self.clip_len = clip_len
        self.half = clip_len // 2
        self.transform = transform or T.ToTensor()

        noisy_dir = os.path.join(root_dir, 'noisy')
        clean_dir = os.path.join(root_dir, 'clean')
        pat = re.compile(r'^noisy_(T\d+)_(frame\d+)_(\d+)_([SML])\.png$')

        seqs = {}
        for fn in sorted(os.listdir(noisy_dir)):
            m = pat.match(fn)
            if m and m.group(4) == level:
                vid, grp, idx_str, _ = m.groups()
                idx = int(idx_str)
                seqs.setdefault((vid, grp), []).append((idx, fn))

        self.samples = []
        for key, frames in seqs.items():
            if len(frames) != 5:
                continue
            frames = sorted(frames, key=lambda x: x[0])
            noisy_paths = [os.path.join(noisy_dir, fn) for _, fn in frames]
            clean_paths = [os.path.join(clean_dir, fn.replace('noisy_', 'clean_')) for _, fn in frames]
            for off in range(5 - clip_len + 1):
                clip_noisy = noisy_paths[off:off + clip_len]
                clip_clean = clean_paths[off:off + clip_len]
                self.samples.append((clip_noisy, clip_clean))

    def __len__(self):
        return len(self.samples)

    # Edited __getitem__ method in order to load frames on demand instead of
    # loading all of them in memory, which is impossible.
    def __getitem__(self, idx):
        noisy_paths, clean_paths = self.samples[idx]
        noisy_imgs = [self.transform(Image.open(p).convert('RGB')) for p in noisy_paths]
        clean_imgs = [self.transform(Image.open(p).convert('RGB')) for p in clean_paths]

        noisy = torch.stack(noisy_imgs, dim=1)
        clean = torch.stack(clean_imgs, dim=1)

        C, T, H, W = noisy.shape
        noisy = noisy.view(C * T, H, W)
        central = clean[:, self.half, :, :]
        return noisy, central
