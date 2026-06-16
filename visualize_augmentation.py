"""
Visualisasi Augmentation Pipeline untuk deep-text-recognition-benchmark.

Menampilkan perbandingan sample gambar sebelum dan sesudah berbagai augmentasi
yang digunakan saat training dengan flag --data_augmentation.

Augmentasi yang divisualisasikan (cocok dengan AlignCollate di dataset.py):
  1. Original         - Gambar asli tanpa augmentasi
  2. ColorJitter       - Brightness/Contrast/Saturation/Hue acak
  3. RandomAffine      - Rotasi + Translasi + Scaling ringan
  4. GaussianBlur      - Blurring (kernel 3x3)
  5. Combined (PIL)    - ColorJitter + RandomAffine + GaussianBlur (stage 1)
  6. Final (Tensor)    - Combined + Resize + Normalize + RandomErasing (stage 1 + 2)

Penggunaan:
  # CLI
  python visualize_augmentation.py --image sample.jpg
  python visualize_augmentation.py --image_dir ./clean_data/train --num_samples 5
  python visualize_augmentation.py --lmdb ./data_lmdb/train --num_samples 4 --output viz.png

  # Notebook
  from visualize_augmentation import visualize_augmentations
  visualize_augmentations(image_path='sample.jpg')
  visualize_augmentations(lmdb_path='./data_lmdb/train', num_samples=4)
"""

import argparse
import os
import random
import math
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend agar aman dipanggil dari notebook
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ──────────────────────────────────────────────
#  Augmentation pipeline (sama persis dengan dataset.py)
# ──────────────────────────────────────────────

def get_augmentation_transforms(force_visualize=True):
    """
    Mengembalikan dictionary berisi semua transform augmentation.

    Args:
        force_visualize: Jika True, GaussianBlur dan RandomErasing dipaksa aktif (p=1.0)
                         agar efeknya selalu terlihat. Jika False, menggunakan probabilitas
                         asli dari training (p=0.2 dan p=0.4).
    """
    blur_p = 1.0 if force_visualize else 0.2
    erase_p = 1.0 if force_visualize else 0.4

    # --- Individual augmentations (masing-masing berdiri sendiri) ---
    color_jitter = transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
    )

    random_affine = transforms.RandomAffine(
        degrees=5, translate=(0.02, 0.02), scale=(0.9, 1.1)
    )

    gaussian_blur = transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=blur_p
    )

    # --- Combined PIL augmentation (Stage 1: persis dengan AlignCollate) ---
    aug_transforms_pil = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.9, 1.1)),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=blur_p
        ),
    ])

    # --- Resize + Normalize (sama persis dengan ResizeNormalize di dataset.py) ---
    resize_normalize = transforms.Compose([
        transforms.Resize((32, 100), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # --- Random Erasing (Stage 2: diterapkan pada tensor) ---
    random_erasing = transforms.RandomErasing(
        p=erase_p, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'
    )

    # --- Final pipeline lengkap (Stage 1 + resize/normalize + Stage 2) ---
    final_pipeline = transforms.Compose([
        aug_transforms_pil,
        resize_normalize,
        random_erasing,
    ])

    return {
        'color_jitter': color_jitter,
        'random_affine': random_affine,
        'gaussian_blur': gaussian_blur,
        'aug_transforms_pil': aug_transforms_pil,
        'resize_normalize': resize_normalize,
        'random_erasing': random_erasing,
        'final_pipeline': final_pipeline,
    }


def apply_augmentations(img, transforms_dict, force_seed=None):
    """
    Terapkan semua augmentasi pada satu gambar PIL.

    Args:
        img: PIL Image (RGB atau L)
        transforms_dict: dictionary dari get_augmentation_transforms()
        force_seed: jika diberikan, set torch seed sebelum setiap augmentasi
                    untuk hasil yang deterministik

    Returns:
        dict: {'original': PIL, 'color_jitter': PIL, ..., 'final': tensor_visual}
    """
    if force_seed is not None:
        torch.manual_seed(force_seed)
        random.seed(force_seed)
        np.random.seed(force_seed)

    is_grayscale = img.mode == 'L'

    if is_grayscale:
        img_rgb = img.convert('RGB')
    else:
        img_rgb = img

    results = {}
    results['original'] = img

    torch.manual_seed(42)
    results['color_jitter'] = transforms_dict['color_jitter'](img_rgb)

    torch.manual_seed(42)
    results['random_affine'] = transforms_dict['random_affine'](img_rgb)

    torch.manual_seed(42)
    results['gaussian_blur'] = transforms_dict['gaussian_blur'](img_rgb)

    torch.manual_seed(42)
    combined_pil = transforms_dict['aug_transforms_pil'](img_rgb)
    results['combined_pil'] = combined_pil

    torch.manual_seed(42)
    final_tensor = transforms_dict['final_pipeline'](img_rgb)
    results['final'] = final_tensor

    return results


def tensor_to_display(tensor):
    """
    Konversi tensor (C,H,W) dengan normalisasi (x-0.5)/0.5 kembali ke
    numpy array uint8 [0,255] untuk display matplotlib.
    Sama seperti tensor2im() di dataset.py.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.cpu().float().numpy()
    # Denormalize: back from (x-0.5)/0.5 to [0,1]
    img_np = img_np * 0.5 + 0.5
    img_np = np.clip(img_np, 0, 1)
    if img_np.shape[0] == 1:
        img_np = np.tile(img_np, (3, 1, 1))
    img_np = np.transpose(img_np, (1, 2, 0))
    return (img_np * 255).astype(np.uint8)


def pil_to_display(pil_img, target_size=(100, 32)):
    """
    Konversi PIL Image ke numpy array uint8 untuk display.
    Resize ke target_size agar seragam.
    """
    if pil_img.mode == 'L':
        pil_img = pil_img.convert('RGB')
    resized = pil_img.resize(target_size, Image.BICUBIC)
    return np.array(resized)


# ──────────────────────────────────────────────
#  Image loaders
# ──────────────────────────────────────────────

def load_images_from_dir(dir_path, num_samples=None):
    exts = ('.jpg', '.jpeg', '.png')
    paths = sorted([
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if f.lower().endswith(exts)
    ])
    n = num_samples if num_samples is not None else 5
    if len(paths) <= n:
        selected = paths
    else:
        step = len(paths) / n
        selected = [paths[int(i * step)] for i in range(n)]
    return [Image.open(p) for p in selected], selected


def load_images_from_lmdb(lmdb_path, num_samples=4):
    """Load gambar dari LMDB dataset (format training)."""
    import lmdb
    import six

    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    images = []
    paths = []

    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))

        if num_samples >= n_samples:
            indices = list(range(1, n_samples + 1))
        else:
            step = n_samples / num_samples
            indices = [int(i * step) + 1 for i in range(num_samples)]

        for idx in indices:
            img_key = f'image-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()

            imgbuf = txn.get(img_key)
            label = txn.get(label_key).decode('utf-8') if txn.get(label_key) else ''

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            images.append(img)
            paths.append(f'[LMDB] idx={idx} label="{label}"')

    env.close()
    return images, paths


# ──────────────────────────────────────────────
#  Visualisasi utama
# ──────────────────────────────────────────────

def visualize_augmentations(
    image_path=None,
    image_dir=None,
    lmdb_path=None,
    images=None,
    num_samples=4,
    output_path=None,
    seed=42,
    figsize=None,
    dpi=150,
):
    """
    Fungsi utama visualisasi augmentasi.

    Args:
        image_path: Path ke satu file gambar
        image_dir: Path ke folder berisi gambar
        lmdb_path: Path ke folder LMDB dataset
        images: List PIL Image langsung (untuk dipanggil dari notebook)
        num_samples: Jumlah sample yang ditampilkan
        output_path: Jika diberikan, simpan plot ke file (default: tampilkan)
        seed: Random seed untuk reproducibility
        figsize: Ukuran figure (tuple width, height in inches)
        dpi: DPI output gambar
    """
    source_labels = []

    if images is not None:
        pil_images = images[:num_samples]
        source_labels = [f'Sample {i+1}' for i in range(len(pil_images))]
    elif image_path is not None:
        pil_images = [Image.open(image_path)]
        source_labels = [os.path.basename(image_path)]
    elif image_dir is not None:
        pil_images, paths = load_images_from_dir(image_dir, num_samples)
        source_labels = [os.path.basename(p) for p in paths]
    elif lmdb_path is not None:
        pil_images, labels = load_images_from_lmdb(lmdb_path, num_samples)
        source_labels = labels
    else:
        raise ValueError("Salah satu dari image_path, image_dir, lmdb_path, atau images harus diisi.")

    if len(pil_images) == 0:
        print("Tidak ada gambar ditemukan.")
        return

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    tforms = get_augmentation_transforms(force_visualize=True)

    col_labels = [
        ('Original', '#2E86AB'),
        ('ColorJitter\n(b=0.5, c=0.5, s=0.5, h=0.1)', '#A23B72'),
        ('RandomAffine\n(deg=5, trans=0.02, scale=0.9-1.1)', '#F18F01'),
        ('GaussianBlur\n(k=3, σ=0.1-2.0, p=1.0)', '#C73E1D'),
        ('Combined PIL\n(Stage 1)', '#6A4C93'),
        ('Final (Tensor)\n+ RandomErasing\n(Stage 1+2)', '#198A5A'),
    ]

    n_rows = len(pil_images)
    n_cols = len(col_labels)

    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 2.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    header_ax = fig.add_subplot(n_rows + 1, n_cols, 1)
    header_ax.axis('off')
    header_ax2 = fig.add_subplot(n_rows + 1, n_cols, n_cols + 1)
    header_ax2.axis('off')

    for row_idx in range(n_rows):
        img = pil_images[row_idx]

        if img.mode == 'L':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img.copy()

        results = apply_augmentations(img_rgb, tforms, force_seed=seed + row_idx)

        for col_idx, (label, color) in enumerate(col_labels):
            ax = axes[row_idx, col_idx]

            col_map = {0: 'original', 1: 'color_jitter', 2: 'random_affine',
                       3: 'gaussian_blur', 4: 'combined_pil', 5: 'final'}
            key = col_map[col_idx]

            if key == 'final':
                display_img = tensor_to_display(results[key])
            else:
                display_img = pil_to_display(results[key])

            ax.imshow(display_img)
            ax.axis('off')

            if row_idx == 0:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(2)
                ax.set_title(label, fontsize=9, fontweight='bold', color=color, pad=6)

        label_color = '#333333'
        fig.text(
            0.01,
            0.5 - (row_idx - n_rows / 2 + 0.5) * (0.85 / n_rows),
            source_labels[row_idx],
            fontsize=7,
            color=label_color,
            va='center',
            ha='left',
            style='italic',
        )

    plt.suptitle(
        'Data Augmentation Pipeline — Before vs After',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        print(f"Visualisasi disimpan ke: {output_path}")
    else:
        plt.show()

    return fig


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualisasi augmentasi data untuk deep-text-recognition-benchmark'
    )
    parser.add_argument('--image', type=str, default=None,
                        help='Path ke satu file gambar')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path ke folder berisi gambar (jpg/png)')
    parser.add_argument('--lmdb', type=str, default=None,
                        help='Path ke folder LMDB dataset')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Jumlah sample gambar yang ditampilkan (default: 4)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path output gambar (contoh: viz_aug.png)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed untuk reproducibility (default: 42)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI gambar output (default: 150)')

    args = parser.parse_args()

    if not any([args.image, args.image_dir, args.lmdb]):
        parser.error("Salah satu dari --image, --image_dir, atau --lmdb harus diisi.")
        return

    visualize_augmentations(
        image_path=args.image,
        image_dir=args.image_dir,
        lmdb_path=args.lmdb,
        num_samples=args.num_samples,
        output_path=args.output,
        seed=args.seed,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
