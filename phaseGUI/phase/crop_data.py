import numpy as np

def is_valid_square(rstart, cstart, size, mask):
    region = mask[rstart:rstart+size, cstart:cstart+size]

    # --- Check rows (top/bottom must have some content) ---
    if not np.any(region[0, :]):   # top row all black
        return False
    if not np.any(region[-1, :]):  # bottom row all black
        return False

    # --- Check columns (ignore fully black rows) ---
    valid_rows = np.any(region, axis=1)  # which rows have content
    if not np.all(region[valid_rows, 0]):   # left col must touch content
        return False
    if not np.all(region[valid_rows, -1]):  # right col must touch content
        return False

    return True

# def extract_strict_inner_square(img: np.ndarray):

def get_strict_inner_square_bbox(img: np.ndarray):
    """
    Find the bounding box (rstart, cstart, size) of the largest square of pure content (no outer padding).
    """
    mask = img > 0
    if not np.any(mask):
        return None

    # --- Step 1: find valid rows ---
    rows = np.any(mask, axis=1)
    r_indices = np.where(rows)[0]
    rmin, rmax = r_indices[[0, -1]]

    # --- Step 2: restrict columns to valid rows only ---
    restricted_mask = mask[rmin:rmax+1, :]
    cols = np.any(restricted_mask, axis=0)
    c_indices = np.where(cols)[0]
    cmin, cmax = c_indices[[0, -1]]

    h = rmax - rmin + 1
    w = cmax - cmin + 1
    max_size = min(h, w)

    # Try from largest possible square downward
    for size in range(max_size, 0, -1):
        for rstart in range(rmin, rmax - size + 2):
            for cstart in range(cmin, cmax - size + 2):
                if is_valid_square(rstart, cstart, size, mask):
                    return (rstart, cstart, size)
    return None

def extract_square_by_bbox(img: np.ndarray, bbox):
    """
    Extract the square region from img using bbox (rstart, cstart, size),
    and interpolate missing rows.
    """
    if bbox is None:
        return None
    rstart, cstart, size = bbox
    content = img[rstart:rstart+size, cstart:cstart+size]
    return interpolate_missing_rows(content)


def put_back(original: np.ndarray, processed: np.ndarray, bbox):
    """Put processed square back into its original position."""
    result = original.copy()
    rstart, cstart, size = bbox
    result[rstart:rstart+size, cstart:cstart+size] = processed
    return result

def interpolate_missing_rows(img):
    # Interpolate missing (all-zero) rows using nearest non-zero row values
    if img is None:
        return None
    img_interp = img.copy()
    rows = np.any(img_interp != 0, axis=1)
    if not np.any(rows):
        return img_interp  # all rows are zero

    nonzero_indices = np.where(rows)[0]
    for i in range(img_interp.shape[0]):
        if not rows[i]:
            # Find nearest nonzero row index
            nearest = nonzero_indices[np.argmin(np.abs(nonzero_indices - i))]
            img_interp[i, :] = img_interp[nearest, :]
    return img_interp


if __name__ == "__main__":

    def processing(x):
        from phaseModel import PhaseModel
        pm = PhaseModel()
        o, _ = pm.spiral_darkfield(x, 0, 256, -1)
        return o

    import matplotlib.pyplot as plt
    import cv2

    img1 = cv2.imread("C:/Users/julia/Documents/!SLO/AngioSLO_Measurement_2Deg_NFrames_150_2023-03-15_12-48-21-4080/channel1/calib/_calib_00035.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("C:/Users/julia/Documents/!SLO/AngioSLO_Measurement_2Deg_NFrames_150_2023-03-15_12-48-21-4080/channel2/calib/_calib_00035.png", cv2.IMREAD_GRAYSCALE)
    # img_ = cv2.imread("C:/Users/julia/Documents/!SLO/AngioSLO_Measurement_2Deg_NFrames_150_2023-03-15_12-48-21-4080/channel2/stab/_stab_00035.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("C:/Users/julia/Documents/!SLO/ED_remmide/Processed32/ch2/calib2_RegVid0106.png", cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread("C:/Users/julia/Documents/!SLO/ED_remmide/Processed32/ch1/calib1_RegVid0358.png", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread("C:/Users/julia/Documents/!SLO/ED_remmide/Processed32/ch2/calib2_RegVid0358.png", cv2.IMREAD_GRAYSCALE)

    # Step 1: get bbox from img1
    bbox = get_strict_inner_square_bbox(img1)
    print("Detected bbox:", bbox)

    # Step 2: extract content from both images using bbox
    content1 = extract_square_by_bbox(img1, bbox)
    content2 = extract_square_by_bbox(img2, bbox)

    # Step 3: processing (example: process both)
    processed1 = processing(content1)
    processed2 = processing(content2)

    # Step 4: put back
    final_img1 = put_back(img1, processed1, bbox)
    final_img2 = put_back(img2, processed2, bbox)

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0,0].imshow(img1, cmap="gray"); axes[0,0].set_title("Original 1")
    axes[0,1].imshow(content1, cmap="gray"); axes[0,1].set_title("Largest Square 1")
    axes[0,2].imshow(processed1, cmap="gray"); axes[0,2].set_title("Processed 1")
    axes[0,3].imshow(final_img1, cmap="gray"); axes[0,3].set_title("Final 1")
    axes[1,0].imshow(img2, cmap="gray"); axes[1,0].set_title("Original 2")
    axes[1,1].imshow(content2, cmap="gray"); axes[1,1].set_title("Largest Square 2")
    axes[1,2].imshow(processed2, cmap="gray"); axes[1,2].set_title("Processed 2")
    axes[1,3].imshow(final_img2, cmap="gray"); axes[1,3].set_title("Final 2")
    for row in axes:
        for ax in row:
            ax.axis("off")
    plt.tight_layout()
    plt.show()