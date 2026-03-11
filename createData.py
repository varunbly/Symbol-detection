import os
import random
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# ---------------- CONFIG ----------------
PDF_PATH = "symbols.pdf"
DPI = 300

AUGS_PER_SYMBOL = 50
TRAIN_RATIO = 0.8

OUT_DIR = "dataset"

random.seed(42)
np.random.seed(42)

# ----------------------------------------

import math

def colorize_bw_symbol(
    img,
    sat_range=(40, 120),
    hue_range=(0, 179),
    value_preserve=True
):
    """
    Injects color into black/white symbols by increasing saturation
    and assigning a random hue.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    # Mask near-black and near-white pixels (symbol regions)
    # Adjust V threshold if needed
    mask = (hsv[..., 2] < 240)

    # Assign random hue
    hue = random.randint(*hue_range)
    hsv[..., 0][mask] = hue

    # Inject saturation
    sat = random.randint(*sat_range)
    hsv[..., 1][mask] = sat

    if not value_preserve:
        hsv[..., 2][mask] = np.clip(
            hsv[..., 2][mask] * random.uniform(0.9, 1.1),
            0, 255
        )

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_gaussian_noise(img, sigma_range=(5, 25)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(img, amount_range=(0.001, 0.01)):
    amount = random.uniform(*amount_range)
    noisy = img.copy()
    h, w, _ = noisy.shape
    num = int(amount * h * w)

    # Salt
    coords = (np.random.randint(0, h, num),
              np.random.randint(0, w, num))
    noisy[coords] = 255

    # Pepper
    coords = (np.random.randint(0, h, num),
              np.random.randint(0, w, num))
    noisy[coords] = 0

    return noisy

def add_jpeg_noise(img, quality_range=(30, 80)):
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def add_blur(img, k_range=(3, 5)):
    k = random.choice(range(k_range[0], k_range[1] + 1, 2))
    return cv2.GaussianBlur(img, (k, k), 0)

def rotate_3d(img, yaw, pitch, roll, fov=60, bg_color=(255, 255, 255)):
    """
    Apply 3D rotation with perspective projection.
    yaw, pitch, roll in degrees.
    """
    h, w = img.shape[:2]

    # Convert degrees to radians
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    # Camera parameters
    f = w / (2 * math.tan(math.radians(fov) / 2))
    cx, cy = w / 2, h / 2

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])

    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    # 3D points of image corners
    corners = np.array([
        [-w/2, -h/2, 0],
        [ w/2, -h/2, 0],
        [ w/2,  h/2, 0],
        [-w/2,  h/2, 0]
    ])

    rotated = corners @ R.T

    # Project to 2D
    projected = []
    for x, y, z in rotated:
        z += f
        projected.append([
            f * x / z + cx,
            f * y / z + cy
        ])

    projected = np.array(projected, dtype=np.float32)

    src = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, projected)

    warped = cv2.warpPerspective(
        img, H, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color
    )

    return warped


def get_class(page):
    if page == 1:
        return "logo"
    elif 2 <= page <= 16:
        return "fake_symbol"
    else:
        return "real_symbol"


def augment_image(img):
    """Apply brightness, saturation, rotation, scale"""
    h, w = img.shape[:2]

    # --- Colorize black & white symbols (NEW) ---
    if random.random() < 0.6:
        img = colorize_bw_symbol(img)

    # --- Brightness & Saturation ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(0.5, 1.5)  # saturation
    hsv[..., 2] *= random.uniform(0.5, 1.5)  # brightness
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- 3D Rotation (cube-like) ---
    yaw   = random.uniform(-70, 70)   # left-right
    pitch = random.uniform(-70, 70)   # up-down
    roll  = random.uniform(-10, 10)   # slight roll

    img = rotate_3d(img, yaw, pitch, roll)

    # --- Rotation & Scale ---
    angle = random.uniform(-180, 180)
    scale = random.uniform(0.50, 1.50)

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    img = cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    if random.random() < 0.8:
        img = add_gaussian_noise(img)

    if random.random() < 0.5:
        img = add_salt_pepper(img)

    if random.random() < 0.6:
        img = add_jpeg_noise(img)

    if random.random() < 0.4:
        img = add_blur(img)


    return img


def ensure_dirs():
    for split in ["train", "val"]:
        for cls in ["logo", "fake_symbol", "real_symbol"]:
            os.makedirs(f"{OUT_DIR}/{split}/{cls}", exist_ok=True)


# ---------------- PIPELINE ----------------

print("📄 Converting PDF to images...")
pages = convert_from_path(PDF_PATH, dpi=DPI)

ensure_dirs()

print("🎨 Generating augmented data...")
for page_idx, page in enumerate(pages, start=1):
    cls = get_class(page_idx)

    # Convert PIL → OpenCV
    page = np.array(page)
    page = cv2.cvtColor(page, cv2.COLOR_RGB2BGR)

    # Generate augmentations
    augmented = []
    for i in range(AUGS_PER_SYMBOL):
        aug = augment_image(page)
        augmented.append(aug)

    # Split within THIS symbol
    random.shuffle(augmented)
    split_idx = int(TRAIN_RATIO * len(augmented))
    train_imgs = augmented[:split_idx]
    val_imgs = augmented[split_idx:]

    # Save
    for i, img in enumerate(train_imgs):
        cv2.imwrite(
            f"{OUT_DIR}/train/{cls}/page{page_idx}_train_{i}.png",
            img
        )

    for i, img in enumerate(val_imgs):
        cv2.imwrite(
            f"{OUT_DIR}/val/{cls}/page{page_idx}_val_{i}.png",
            img
        )

print("✅ Dataset creation complete.")
