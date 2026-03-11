import cv2
import numpy as np
import random
import os
from glob import glob
import math

# ---------------- CONFIG ----------------

IMG_SIZE = 640
TRAIN_SPLIT = 0.8
IMAGES_PER_SYMBOL = 50
MAX_SYMBOLS_PER_IMAGE = 1

CLASS_MAP = {
    "logo": 0,
    "fake_symbol": 1,
    "real_symbol": 2
}

OUT_DIR = "dataset4"

# ----------------------------------------

def make_dirs():
    for s in ["train", "val"]:
        os.makedirs(f"{OUT_DIR}/images/{s}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{s}", exist_ok=True)

# --------- SYNTHETIC BACKGROUNDS ---------

def random_background(size):
    bg_images = glob("generic_background/*")
    if bg_images:
        bg_path = random.choice(bg_images)
        bg = cv2.imread(bg_path)
        if bg is not None:
            bg = cv2.resize(bg, (size, size))
            return bg
    
    # Fallback to synthetic background
    h, w = size, size
    bg_type = random.choice(["solid", "gradient", "noise"])

    if bg_type == "solid":
        color = [random.randint(50, 255) for _ in range(3)]
        bg = np.full((h, w, 3), color, dtype=np.uint8)

    elif bg_type == "gradient":
        start = [random.randint(50, 200) for _ in range(3)]
        end = [random.randint(100, 255) for _ in range(3)]
        grad = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            grad[:, :, c] = np.linspace(start[c], end[c], w).astype(np.uint8)
        bg = grad

    else:  # noise
        bg = np.random.randint(50, 255, (h, w, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (7, 7), 0)

    return bg

# -------- SYMBOL AUGMENTATIONS --------

def mask_symbol(img):
    mask = np.all(img != [233, 233, 233], axis=-1).astype(np.uint8)
    masked = np.full_like(img, 255, dtype=np.uint8)
    for c in range(3):
        masked[:, :, c] = np.where(mask == 1, img[:, :, c], 255)
    return masked


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

def rotate_3d(img, yaw, pitch, roll, fov=60, bg_color=(233, 233, 233)):
    h, w = img.shape[:2]
    yaw, pitch, roll = math.radians(yaw), math.radians(pitch), math.radians(roll)
    f = w / (2 * math.tan(math.radians(fov) / 2))
    cx, cy = w / 2, h / 2
    
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    corners = np.array([[-w/2, -h/2, 0], [w/2, -h/2, 0], [w/2, h/2, 0], [-w/2, h/2, 0]])
    rotated = corners @ R.T
    projected = np.array([[f * x / (z + f) + cx, f * y / (z + f) + cy] for x, y, z in rotated], dtype=np.float32)
    
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, projected)
    return cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)



def add_noise(img):
    if random.random() < 0.8:
        img = add_gaussian_noise(img)

    if random.random() < 0.5:
        img = add_salt_pepper(img)

    if random.random() < 0.6:
        img = add_jpeg_noise(img)

    if random.random() < 0.4:
        img = add_blur(img)

    return img

def rotate_symbol(img, bg_color=(233,233,233)):
    h, w = img.shape[:2]
    angle = random.uniform(-180, 180)
    scale = random.uniform(0.7, 1.2)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)

# --------- COMPOSITING ---------

def extract_mask(img, cls_id):
    if cls_id == 0:  # logo class
        return np.all(img != [124, 221, 243], axis=-1).astype(np.uint8)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return ((gray <= 35) | (gray == 255)).astype(np.uint8)

def paste_symbol(bg, symbol, x, y, cls_id):
    h, w = symbol.shape[:2]
    mask = extract_mask(symbol, cls_id)
    for c in range(3):
        bg[y:y+h, x:x+w, c] = np.where(
            mask == 1,
            symbol[..., c],
            bg[y:y+h, x:x+w, c]
        )
    return bg

def check_overlap(new_box, existing_boxes, min_distance=20):
    """Check if new box overlaps with existing boxes"""
    x1, y1, w1, h1 = new_box
    for x2, y2, w2, h2 in existing_boxes:
        if (x1 < x2 + w2 + min_distance and x1 + w1 + min_distance > x2 and
            y1 < y2 + h2 + min_distance and y1 + h1 + min_distance > y2):
            return True
    return False

# -------- DATASET GENERATION --------

def generate():
    make_dirs()
    symbols = []

    for cls in CLASS_MAP:
        for p in glob(f"symbols/{cls}/*"):
            symbols.append((p, CLASS_MAP[cls]))

    # random.shuffle(symbols)

    for idx, (path, cls_id) in enumerate(symbols):
        symbol_img = cv2.imread(path)
        if symbol_img is None:
            continue

        for i in range(IMAGES_PER_SYMBOL):
            split = "train" if i < int(IMAGES_PER_SYMBOL * TRAIN_SPLIT) else "val"
            bg = random_background(IMG_SIZE)
            labels = []
            placed_boxes = []

            num_objects = random.randint(1, MAX_SYMBOLS_PER_IMAGE)
            if random.random() < 0.2:
                num_objects = 0

            for _ in range(num_objects):
                sym = symbol_img.copy()
                if cls_id == 0:  # logo class
                    bg_color = (124, 221, 243)  # #F3DD7C
                    sym = rotate_symbol(sym, bg_color)
                    if random.random() < 0.8:
                        yaw = random.uniform(-65, 65)
                        pitch = random.uniform(-65, 65)
                        roll = random.uniform(-5, 5)
                        sym = rotate_3d(sym, yaw, pitch, roll, bg_color=bg_color)
                else:
                    sym = rotate_symbol(sym)
                    if random.random() < 0.8:
                        yaw = random.uniform(-65, 65)
                        pitch = random.uniform(-65, 65)
                        roll = random.uniform(-5, 5)
                        sym = rotate_3d(sym, yaw, pitch, roll)
                
                # sym = add_noise(sym)

                scale = random.uniform(0.05, 0.15)
                h, w = sym.shape[:2]
                sym = cv2.resize(sym, (int(w*scale), int(h*scale)))

                sh, sw = sym.shape[:2]
                # if sh >= IMG_SIZE or sw >= IMG_SIZE:
                #     continue

                # Try to find non-overlapping position
                attempts = 0
                while attempts < 30:
                    x = random.randint(0, IMG_SIZE - sw)
                    y = random.randint(0, IMG_SIZE - sh)
                    
                    if not check_overlap((x, y, sw, sh), placed_boxes, min_distance=10):
                        break
                    attempts += 1
                
                if attempts >= 30:
                    continue

                bg = paste_symbol(bg, sym, x, y, cls_id)
                # bg = colorize_bw_symbol(bg)
                # bg = add_noise(bg)
                placed_boxes.append((x, y, sw, sh))

                xc = (x + sw/2) / IMG_SIZE
                yc = (y + sh/2) / IMG_SIZE
                ww = sw / IMG_SIZE
                hh = sh / IMG_SIZE

                labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

            img_name = f"img_{idx}_{i}.jpg"
            lbl_name = f"img_{idx}_{i}.txt"

            cv2.imwrite(f"{OUT_DIR}/images/{split}/{img_name}", bg)

            with open(f"{OUT_DIR}/labels/{split}/{lbl_name}", "w") as f:
                f.write("\n".join(labels))

    # data.yaml
    with open(f"{OUT_DIR}/data.yaml", "w") as f:
        f.write(
            f"path: {OUT_DIR}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: logo\n"
            "  1: fake_symbol\n"
            "  2: real_symbol\n"
        )

    print("Dataset generation complete.")

# -------- RUN --------

if __name__ == "__main__":
    generate()
