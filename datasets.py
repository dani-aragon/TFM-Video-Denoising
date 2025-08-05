import os
import glob
import re
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class Train3DDataset(Dataset):
    """
    Dataset PVDD que toma 30//clip_len clips NO solapados de longitud clip_len
    por cada bloque de 30 frames, cargando bajo demanda.
    """
    def __init__(
        self, root_dir: str, clip_len: int, transform=None
    ):
        self.clip_len  = clip_len
        self.transform = transform or T.ToTensor()

        # 1) recoge todos los paths clean/noisy
        ext = "png"
        clean_files = glob.glob(
            os.path.join(root_dir, 'clean', f'clean_*.{ext}')
        )
        noisy_files = glob.glob(
            os.path.join(root_dir, 'noisy', f'noisy_*.{ext}')
        )

        # 2) agrupa por (video, frame_group)
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

        # 3) filtra y ordena secuencias válidas de 30 frames
        self.samples = []
        for key, d in seqs.items():
            clean_list = sorted(d['clean'], key=lambda x: x[0])
            noisy_list = sorted(d['noisy'], key=lambda x: x[0])
            if len(clean_list)==30 and len(noisy_list)==30:
                clean_paths = [p for _,p in clean_list]
                noisy_paths = [p for _,p in noisy_list]
                self.samples.append((noisy_paths, clean_paths))
        assert self.samples, "No hay secuencias de 30 frames válidas"

        # 4) offsets NO solapados
        self.offsets = list(range(0, 30 - clip_len + 1, clip_len))

    def __len__(self):
        # cada secuencia aporta len(self.offsets) clips
        return len(self.samples) * len(self.offsets)

    def __getitem__(self, idx):
        clips_per_seq = len(self.offsets)
        seq_idx    = idx // clips_per_seq
        offset_idx = idx %  clips_per_seq
        off        = self.offsets[offset_idx]

        noisy_paths, clean_paths = self.samples[seq_idx]

        # selecciona clip de longitud clip_len
        noisy_clip = noisy_paths[off: off + self.clip_len]
        clean_clip = clean_paths[off: off + self.clip_len]

        # carga bajo demanda y transforma
        noisy_imgs = [self.transform(Image.open(p)) for p in noisy_clip]
        clean_imgs = [self.transform(Image.open(p)) for p in clean_clip]

        # apila en (C, T, H, W)
        noisy = torch.stack(noisy_imgs, dim=1)
        clean = torch.stack(clean_imgs, dim=1)
        return noisy, clean

class Test3DDataset(Dataset):
    """
    Test dataset PVDD synNoiseData: 
    - agrupa secuencias de 5 frames (índices 0..4) con sufijo _[SML]
    - genera clips deslizando ventana de tamaño clip_len (p.ej. 4)
    - empareja noisy_<base>_[SML].png <-> clean_<base>_[SML].png
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

        # Paso 1: agrupa por (vid, frameGroup, level)
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

        # Paso 2: para cada secuencia válida de 4 frames, crea clips
        self.clips = []
        for (vid, grp, lvl), frames in seqs.items():
            if len(frames) != 5:
                continue
            # ordena por idx 0..4
            frames = sorted(frames, key=lambda x: x[0])
            noisy_paths = [os.path.join(noisy_dir, fn) for _, fn in frames]
            clean_paths = [
                os.path.join(clean_dir, fn.replace('noisy_','clean_'))
                for _, fn in frames
            ]
            # verifica existencia
            for p in clean_paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(
                        f"No existe el archivo clean: {p}"
                    )
            # sliding window
            for off in range(5 - clip_len + 1):
                self.clips.append((
                    noisy_paths[off:off+clip_len],
                    clean_paths[off:off+clip_len]
                ))

        assert self.clips, "No se han encontrado secuencias/clips válidos"

    def __len__(self):
        return len(self.clips)

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
    Dataset para entrenar FastDVDnet de forma 2D (sin noise_map).
    Devuelve clips ruidosos apilados en canales y el frame limpio central.
    """
    def __init__(self, root_dir: str, clip_len: int = 5, transform=None):
        super().__init__()
        assert clip_len % 2 == 1, "clip_len debe ser impar para tener un frame central"
        self.clip_len = clip_len
        self.half = clip_len // 2
        self.transform = transform or T.ToTensor()

        # Obtener listas de archivos clean y noisy
        ext = 'png'
        clean_files = glob.glob(os.path.join(root_dir, 'clean', f'clean_*.{ext}'))
        noisy_files = glob.glob(os.path.join(root_dir, 'noisy', f'noisy_*.{ext}'))

        # Agrupar por secuencia completa de 30 frames
        seqs = {}
        for p in clean_files:
            fn = os.path.basename(p)
            _, vid, grp, idx_ext = fn.split('_')
            idx = int(idx_ext.split('.')[0])
            seqs.setdefault((vid, grp), {})['clean'] = seqs.get((vid, grp), {}).get('clean', []) + [(idx, p)]
        for p in noisy_files:
            fn = os.path.basename(p)
            _, vid, grp, idx_ext = fn.split('_')
            idx = int(idx_ext.split('.')[0])
            seqs.setdefault((vid, grp), {})['noisy'] = seqs.get((vid, grp), {}).get('noisy', []) + [(idx, p)]

        # Filtrar secuencias completas de 30 frames y generar offsets no solapados
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

    def __getitem__(self, idx):
        noisy_paths, clean_paths = self.samples[idx]
        # Cargar y transformar
        noisy_imgs = [self.transform(Image.open(p).convert('RGB')) for p in noisy_paths]
        clean_imgs = [self.transform(Image.open(p).convert('RGB')) for p in clean_paths]

        # Apilar como (C, T, H, W)
        noisy = torch.stack(noisy_imgs, dim=1)
        clean = torch.stack(clean_imgs, dim=1)

        # Aplanar clips ruidosos: (C*T, H, W)
        C, T, H, W = noisy.shape
        noisy = noisy.view(C * T, H, W)

        # Frame central limpio: (C, H, W)
        central = clean[:, self.half, :, :]
        return noisy, central


class Test2DDataset(Dataset):
    """
    Dataset de prueba para FastDVDnet de forma 2D.
    Desliza ventana de tamaño clip_len sobre secuencias de 5 frames.
    """
    def __init__(self, root_dir: str, level: str, clip_len: int = 5, transform=None):
        super().__init__()
        assert level in ('S', 'M', 'L'), "level debe ser 'S', 'M' o 'L'"
        assert clip_len % 2 == 1, "clip_len debe ser impar"
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
