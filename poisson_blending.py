import argparse

import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve

COLORS = 3


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    im_src = im_src.astype(np.float64)
    im_tgt = im_tgt.astype(np.float64)

    # ROI: Region of Interest
    roi_src, roi_mask = get_src_roi_crops(im_src, im_mask)
    x_min, x_max, y_min, y_max = get_tgt_roi_bounds(
        center, im_tgt.shape[:2], roi_src.shape[:2]
    )
    roi_tgt = im_tgt[y_min:y_max, x_min:x_max]

    roi_blend = blend_src_to_tgt(roi_src, roi_tgt, roi_mask)
    im_blend = im_tgt.copy()  # we need to return an "im_blend" variable
    im_blend[y_min:y_max, x_min:x_max] = roi_blend
    return im_blend.astype(np.uint8)


def get_src_roi_crops(im_src: np.ndarray, im_mask: np.ndarray) -> tuple:
    h, w = im_src.shape[:2]

    # Get the bbox of the mask with 1-pixel padding
    indices = np.nonzero(im_mask)
    x_min, y_min = np.min(indices, axis=1)
    x_max, y_max = np.max(indices, axis=1)
    x_min = max(0, x_min - 1)
    x_max = min(h, x_max + 2)
    y_min = max(0, y_min - 1)
    y_max = min(w, y_max + 2)

    # Extract the ROI from the source image and mask
    roi_src = im_src[x_min:x_max, y_min:y_max, :]
    roi_mask = im_mask[x_min:x_max, y_min:y_max]

    return roi_src, roi_mask


def get_tgt_roi_bounds(center: tuple, tgt_shp: tuple, roi_shp: tuple) -> tuple:
    x_min = max(0, center[0] - roi_shp[1] // 2)
    x_max = min(tgt_shp[1], x_min + roi_shp[1])
    y_min = max(0, center[1] - roi_shp[0] // 2)
    y_max = min(tgt_shp[0], y_min + roi_shp[0])
    return x_min, x_max, y_min, y_max


def blend_src_to_tgt(src: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = src.shape[:2]
    A = create_poisson_matrix(mask, h, w)
    roi_laplacian = get_image_laplacian(src, tgt, mask)
    roi_blend = np.zeros((h, w, COLORS), np.float64)
    for i in range(COLORS):
        roi_blend[:, :, i] = spsolve(A, roi_laplacian[:, :, i].flatten()).reshape(
            (h, w)
        )

    roi_blend = np.clip(roi_blend, 0, 255)
    return roi_blend


def create_poisson_matrix(mask: np.ndarray, h: int, w: int) -> scipy.sparse.lil_matrix:
    A = scipy.sparse.lil_matrix((h * w, h * w))

    for y in range(h):
        for x in range(w):
            i = y * w + x
            if mask[y, x] == 0:
                A[i, i] = 1
            else:
                A[i, i] = -4
                if y > 0 and mask[y - 1, x] != 0:
                    A[i, i - w] = 1
                if y < h - 1 and mask[y + 1, x] != 0:
                    A[i, i + w] = 1
                if x > 0 and mask[y, x - 1] != 0:
                    A[i, i - 1] = 1
                if x < w - 1 and mask[y, x + 1] != 0:
                    A[i, i + 1] = 1

    return A.tocsc()


def get_image_laplacian(
    src: np.ndarray, tgt: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    h, w = src.shape[:2]
    laplacian = np.copy(tgt)

    for y in range(h):
        for x in range(w):
            if mask[y, x] != 0:
                laplacian[y, x, :] = -4 * src[y, x, :]
                if y > 0:
                    laplacian[y, x, :] += src[y - 1, x, :]
                    if mask[y - 1, x] == 0:
                        laplacian[y, x, :] -= tgt[y - 1, x, :]

                if y < h - 1:
                    laplacian[y, x, :] += src[y + 1, x, :]
                    if mask[y + 1, x] == 0:
                        laplacian[y, x, :] -= tgt[y + 1, x, :]

                if x > 0:
                    laplacian[y, x, :] += src[y, x - 1, :]
                    if mask[y, x - 1] == 0:
                        laplacian[y, x, :] -= tgt[y, x - 1, :]

                if x < w - 1:
                    laplacian[y, x, :] += src[y, x + 1, :]
                    if mask[y, x + 1] == 0:
                        laplacian[y, x, :] -= tgt[y, x + 1, :]

    return laplacian


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        default="./data/imgs/banana1.jpg",
        help="image file path",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="./data/seg_GT/banana1.bmp",
        help="mask file path",
    )
    parser.add_argument(
        "--tgt_path", type=str, default="./data/bg/table.jpg", help="mask file path"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == "":
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow("Cloned image", im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
