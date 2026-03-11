import cv2
import numpy as np
import random
import os
from glob import glob
import math
from collections import defaultdict

# ---------------- CONFIG ----------------

TOTAL_IMAGES = 10000
IMG_SIZE = 640
TRAIN_SPLIT = 0.9
MIN_SYMBOLS_PER_IMAGE = 1
MAX_SYMBOLS_PER_IMAGE = 4
EMPTY_IMAGE_RATIO = 0.15

CLASS_MAP = {
    "logo": 0,
    "fake_symbol": 1,
    "real_symbol": 2
}

# Define weights for specific symbols (filename only).
# Default weight is 1.0 if not specified.
SYMBOL_WEIGHTS = {
    # Example: "symbol_01.png": 5.0,
    # "symbol_02.png": 0.5,
    "fake_page_8.png": 3,
    "real_page_30.png": 3
}

OUT_DIR = "dataset8"

# ----------------------------------------

def make_dirs():
    for s in ["train", "val"]:
        os.makedirs(f"{OUT_DIR}/images/{s}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{s}", exist_ok=True)

def random_background(size):
    bg_images = glob("generic_background/*")
    if bg_images:
        bg_path = random.choice(bg_images)
        bg = cv2.imread(bg_path)
        if bg is not None:
            bg = cv2.resize(bg, (size, size))
            return bg
    
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
    else:
        bg = np.random.randint(50, 255, (h, w, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (7, 7), 0)
    return bg


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

def add_hue_noise(img, hue_shift_range=(-90, 90), sat_shift_range=(50, 100), val_shift_range=(-20, 20)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Add random shift to hue
    hue_shift = random.randint(*hue_shift_range)
    h_new = h.astype(np.int16) + hue_shift
    h_new = np.where(h_new < 0, h_new + 180, h_new)
    h_new = np.where(h_new >= 180, h_new - 180, h_new)
    
    # Add random shift to saturation to colorize BW images
    # If the image is BW, S is low. We want to boost it.
    sat_shift = random.randint(*sat_shift_range)
    s_new = s.astype(np.int16) + sat_shift
    s_new = np.clip(s_new, 0, 255)
    
    # Optional: slight value shift to add more variety
    val_shift = random.randint(*val_shift_range)
    v_new = v.astype(np.int16) + val_shift
    v_new = np.clip(v_new, 0, 255)
    
    hsv_new = cv2.merge([h_new.astype(np.uint8), s_new.astype(np.uint8), v_new.astype(np.uint8)])
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

def add_noise(img):
    if random.random() < 0.8:
        img = add_gaussian_noise(img)

    if random.random() < 0.5:
        img = add_salt_pepper(img)

    if random.random() < 0.6:
        img = add_jpeg_noise(img)

    if random.random() < 0.4:
        img = add_blur(img)

    if random.random() < 0.5:
        img = add_hue_noise(img)

    return img

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

def rotate_symbol(img, bg_color=(233,233,233)):
    h, w = img.shape[:2]
    angle = random.uniform(-180, 180)
    scale = random.uniform(0.7, 1.2)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)

def extract_mask(img, cls_id):
    if cls_id == 0:
        return np.all(img != [124, 221, 243], axis=-1).astype(np.uint8)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return ((gray <= 35) | (gray == 255)).astype(np.uint8)

def paste_symbol(bg, symbol, x, y, cls_id):
    h, w = symbol.shape[:2]
    mask = extract_mask(symbol, cls_id)
    for c in range(3):
        bg[y:y+h, x:x+w, c] = np.where(mask == 1, symbol[..., c], bg[y:y+h, x:x+w, c])
    return bg

def check_overlap(new_box, existing_boxes, min_distance=20):
    x1, y1, w1, h1 = new_box
    for x2, y2, w2, h2 in existing_boxes:
        if (x1 < x2 + w2 + min_distance and x1 + w1 + min_distance > x2 and
            y1 < y2 + h2 + min_distance and y1 + h1 + min_distance > y2):
            return True
    return False

def choose_balanced_symbol(symbols_by_class):
    # Flatten all symbols to a single list to choose from, or choose class first?
    # The original logic chose a class to balance class counts, then a symbol to balance symbol counts.
    # The request is to "assign weightage to individual symbol and that is how it is chosen".
    # This implies we might want to choose purely based on symbol weights, which implicitly affects class distribution.
    
    all_symbols = []
    all_weights = []
    all_classes = []
    
    for cls_name, cls_id in CLASS_MAP.items():
        # symbols_by_class keys are class IDs
        for sym_path in symbols_by_class[cls_id]:
            sym_name = os.path.basename(sym_path)
            weight = SYMBOL_WEIGHTS.get(sym_name, 1.0)
            
            all_symbols.append(sym_path)
            all_weights.append(weight)
            all_classes.append(cls_id)
            
    # Weighted selection
    # random.choices returns a list, we just want 1 item
    chosen_path = random.choices(all_symbols, weights=all_weights, k=1)[0]
    
    # Find the class for the chosen path (we stored it to avoid searching again)
    # But random.choices only returned the path. 
    # Let's align them.
    idx = all_symbols.index(chosen_path)
    chosen_class = all_classes[idx]
    
    return chosen_path, chosen_class

def generate():
    make_dirs()
    symbols_by_class = defaultdict(list)
    
    for cls in CLASS_MAP:
        for p in glob(f"symbols/{cls}/*"):
            symbols_by_class[CLASS_MAP[cls]].append(p)
    
    class_usage_count = {cls_id: 0 for cls_id in CLASS_MAP.values()}
    symbol_usage_count = defaultdict(int)
    
    for i in range(TOTAL_IMAGES):
        split = "train" if i < int(TOTAL_IMAGES * TRAIN_SPLIT) else "val"
        bg = random_background(IMG_SIZE)
        labels = []
        placed_boxes = []
        
        if random.random() < EMPTY_IMAGE_RATIO:
            num_objects = 0
        else:
            num_objects = random.randint(MIN_SYMBOLS_PER_IMAGE, MAX_SYMBOLS_PER_IMAGE)
        
        for _ in range(num_objects):
            path, cls_id = choose_balanced_symbol(symbols_by_class)
            class_usage_count[cls_id] += 1
            symbol_usage_count[path] += 1
            
            symbol_img = cv2.imread(path)
            if symbol_img is None:
                continue
                
            sym = symbol_img.copy()
            
            if cls_id == 0:
                bg_color = (124, 221, 243)
                sym = rotate_symbol(sym, bg_color)
                if random.random() < 0.95:
                    yaw = random.uniform(-70, 70)
                    pitch = random.uniform(-70, 70)
                    roll = random.uniform(-5, 5)
                    sym = rotate_3d(sym, yaw, pitch, roll, bg_color=bg_color)
            else:
                sym = rotate_symbol(sym)
                if random.random() < 0.95:
                    yaw = random.uniform(-70, 70)
                    pitch = random.uniform(-70, 70)
                    roll = random.uniform(-5, 5)
                    sym = rotate_3d(sym, yaw, pitch, roll)
            
            scale = random.uniform(0.05, 0.15)
            h, w = sym.shape[:2]
            sym = cv2.resize(sym, (int(w*scale), int(h*scale)))
            
            sh, sw = sym.shape[:2]
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
            placed_boxes.append((x, y, sw, sh))
            
            xc = (x + sw/2) / IMG_SIZE
            yc = (y + sh/2) / IMG_SIZE
            ww = sw / IMG_SIZE
            hh = sh / IMG_SIZE
            
            labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        bg = add_noise(bg)
        
        img_name = f"img_{i}.jpg"
        lbl_name = f"img_{i}.txt"
        
        cv2.imwrite(f"{OUT_DIR}/images/{split}/{img_name}", bg)
        
        with open(f"{OUT_DIR}/labels/{split}/{lbl_name}", "w") as f:
            f.write("\n".join(labels))
    
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
    
    print(f"Dataset generation complete. Created {TOTAL_IMAGES} images.")
    print(f"Class usage: {class_usage_count}")
    
    # Print top 10 most used symbols for verification
    sorted_symbols = sorted(symbol_usage_count.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 symbols used:")
    for path, count in sorted_symbols[:10]:
        print(f"{os.path.basename(path)}: {count}")

if __name__ == "__main__":
    generate()