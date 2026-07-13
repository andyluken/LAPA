import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Match your configuration
FRAMES_DIR = Path("/home/andy/Dataset/nuscenes_laq_frames_improved")
PATCH_GRID = (8, 8) 
STATIONARY_THRESHOLD = 0.5

def compute_corrected_heatmap(frame_a: np.ndarray, frame_b: np.ndarray, patch_h: int, patch_w: int) -> tuple:
    """Computes the corrected heatmap and returns raw patch matrices for plotting."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    
    dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
    flow = dis.calc(gray_a, gray_b, None)
    
    fx_raw = flow[..., 0]
    fy_raw = flow[..., 1]
    
    # Area-average downsample FIRST
    fx_patch = cv2.resize(fx_raw, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
    fy_patch = cv2.resize(fy_raw, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
    
    mag_patch, _ = cv2.cartToPolar(fx_patch, fy_patch)
    patch_peak = float(mag_patch.max())
    
    if patch_peak < STATIONARY_THRESHOLD:
        return np.zeros((3, patch_h, patch_w), dtype=np.float32), fx_patch, fy_patch, mag_patch
    
    mag_norm = mag_patch / patch_peak
    fx_norm = fx_patch / patch_peak
    fy_norm = fy_patch / patch_peak
    
    heatmap = np.stack([mag_norm, fx_norm, fy_norm], axis=0).astype(np.float32)
    return heatmap, fx_patch, fy_patch, mag_patch

def save_diagnostic_plot(img_bgr: np.ndarray, heatmap: np.ndarray, fx_p: np.ndarray, fy_p: np.ndarray, out_path: Path):
    """Generates a high-utility 4-panel verification plot."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    # Panel 1: Original Image + Vector Field Flow Overlay
    axes[0].imshow(img_rgb)
    # Generate spatial grid matching the 8x8 patches
    x = np.linspace(w / (2 * PATCH_GRID[1]), w - w / (2 * PATCH_GRID[1]), PATCH_GRID[1])
    y = np.linspace(h / (2 * PATCH_GRID[0]), h - h / (2 * PATCH_GRID[0]), PATCH_GRID[0])
    X, Y = np.meshgrid(x, y)
    # Invert Y vector for matplotlib quiver (image coords vs cartesian coords)
    axes[0].quiver(X, Y, fx_p, -fy_p, color='lime', angles='xy', scale_units='xy', scale=1.0)
    axes[0].set_title("Ego-Motion Flow Field")
    axes[0].axis('off')
    
    # Panel 2: Magnitude Channel [0]
    im1 = axes[1].imshow(heatmap[0], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Ch[0]: Magnitude (0 to 1)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Panel 3: Horizontal Flow Channel [1]
    im2 = axes[2].imshow(heatmap[1], cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_title("Ch[1]: X-Flow (-1=Left, 1=Right)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Panel 4: Vertical Flow Channel [2]
    im3 = axes[3].imshow(heatmap[2], cmap='coolwarm', vmin=-1, vmax=1)
    axes[3].set_title("Ch[2]: Y-Flow (-1=Up, 1=Down)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches='tight', dpi=150)
    plt.close()

def main():
    scene_dirs = sorted(d for d in FRAMES_DIR.iterdir() if d.is_dir())
    print(f"Visualizing scenes from {FRAMES_DIR}...")
    
    for scene_dir in tqdm(scene_dirs[:3], desc="Processing sample scenes"):  # Limit to 3 scenes for verification
        files = sorted([f for f in scene_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        
        for f_a, f_b in zip(files[:-1], files[1:]):
            img_a = cv2.imread(str(f_a))
            img_b = cv2.imread(str(f_b))
            if img_a is None or img_b is None:
                continue
                
            heatmap, fx_p, fy_p, _ = compute_corrected_heatmap(img_a, img_b, PATCH_GRID[0], PATCH_GRID[1])
            
            # Save visual verification sibling file
            out_img_path = f_a.with_suffix('').with_suffix('.debug.png')
            save_diagnostic_plot(img_a, heatmap, fx_p, fy_p, out_img_path)

if __name__ == "__main__":
    main()