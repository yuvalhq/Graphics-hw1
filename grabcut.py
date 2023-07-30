import argparse
import time

import cv2
import igraph as ig
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

EPSILON = np.finfo(np.float64).eps

previous_energy = None


def require_initialization(func):
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            raise ValueError("Instance not initialized")
        return func(self, *args, **kwargs)

    return wrapper


class Component:
    def __init__(self, data_points, total_points_count, mean=None):
        self.data_points = data_points
        self.mean = mean if mean is not None else np.mean(data_points, axis=0)
        self.covariance_matrix = np.cov(self.data_points, rowvar=False)
        self.weight = len(data_points) / total_points_count
        self._multivariate_normal = multivariate_normal(
            self.mean, self.covariance_matrix, allow_singular=True, seed=0
        )

    def calc_pdfs(self, X):
        return self._multivariate_normal.pdf(X)


class GaussianMixture:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self._kmeans = KMeans(n_components, n_init="auto", random_state=0)
        self._initialized = False
        self.components = None
        self.data_points = None

    def init(self, X):
        if self._initialized:
            raise ValueError("Already initialized")
        self._initialized = True
        self.data_points = X
        labels = self._kmeans.fit_predict(X)

        self.components = []
        for i in range(self.n_components):
            component = Component(
                X[labels == i], len(X), mean=self._kmeans.cluster_centers_[i]
            )
            self.components.append(component)

        return self

    @property
    @require_initialization
    def weights(self):
        return np.array([c.weight for c in self.components])

    @require_initialization
    def calc_probs(self, X):
        probs = np.array([c.calc_pdfs(X) for c in self.components]) + EPSILON
        return np.dot(self.weights, probs)

    @require_initialization
    def assign_points_to_components(self, X):
        probs = np.array([c.calc_pdfs(X) for c in self.components])
        return np.argmax(probs, axis=0)

    @require_initialization
    def update(self, X):
        labels = self.assign_points_to_components(X)
        for i in range(self.n_components):
            assigned_pixels = X[labels == i]
            if len(assigned_pixels) <= 1:
                self.n_components -= 1
                self.components.pop(i)
                return self.update(X)
            self.components[i] = Component(assigned_pixels, total_points_count=len(X))


class GrabcutGraph:
    GAMMA = 50
    K = GAMMA * 9

    _beta = None
    _n_link_edges = None
    _n_link_capacities = None

    def __init__(self, pixels_count, edges, capacities):
        self.n = pixels_count + 2
        self.src = pixels_count
        self.sink = pixels_count + 1
        self.edges = edges
        self.capacities = capacities
        self._graph = ig.Graph(
            n=self.n, edges=edges, edge_attrs=dict(capacity=capacities)
        )

    def mincut(self):
        mincut = self._graph.st_mincut(self.src, self.sink, capacity="capacity")
        mincut_sets = mincut.partition
        mincut_sets[0].remove(self.src)
        mincut_sets[1].remove(self.sink)
        return mincut_sets, mincut.value

    @classmethod
    def construct_graph(cls, img, mask, bgGMM, fgGMM):
        rows, cols, _ = img.shape
        n_link_edges, n_link_capacities = cls.n_link_edges_and_capacities(img)
        t_link_edges, t_link_capacities = cls.t_link_edges_and_capacities(
            img, mask, bgGMM, fgGMM
        )
        edges = np.concatenate([t_link_edges, n_link_edges])
        capacities = np.concatenate([t_link_capacities, n_link_capacities])
        return cls(rows * cols, edges, capacities)

    @classmethod
    def beta(cls, img):
        if cls._beta is not None:
            return cls._beta

        cls._beta = 0.0
        rows, cols, _ = img.shape
        for y in range(rows):
            for x in range(cols):
                color = img[y, x]
                if x > 0:
                    diff = color - img[y, x - 1]
                    cls._beta += diff.dot(diff)
                if y > 0 and x > 0:
                    diff = color - img[y - 1, x - 1]
                    cls._beta += diff.dot(diff)
                if y > 0:
                    diff = color - img[y - 1, x]
                    cls._beta += diff.dot(diff)
                if y > 0 and x < cols - 1:
                    diff = color - img[y - 1, x + 1]
                    cls._beta += diff.dot(diff)
        cls._beta = 1.0 / (2 * cls._beta / (4 * cols * rows - 3 * cols - 3 * rows + 2))
        return cls._beta

    @classmethod
    def n_link_edges_and_capacities(cls, img):
        if cls._n_link_edges is not None and cls._n_link_capacities is not None:
            return cls._n_link_edges, cls._n_link_capacities

        rows, cols, _ = img.shape
        img_idxs = np.arange(rows * cols, dtype=np.uint32).reshape(rows, cols)
        beta = cls.beta(img)

        left_edges = np.c_[img_idxs[:, 1:].reshape(-1), img_idxs[:, :-1].reshape(-1)]
        up_edges = np.c_[img_idxs[1:, :].reshape(-1), img_idxs[:-1, :].reshape(-1)]
        upleft_edges = np.c_[
            img_idxs[1:, 1:].reshape(-1), img_idxs[:-1, :-1].reshape(-1)
        ]
        upright_edges = np.c_[
            img_idxs[1:, :-1].reshape(-1), img_idxs[:-1, 1:].reshape(-1)
        ]

        straight_edges = np.concatenate([left_edges, up_edges])
        diagonal_edges = np.concatenate([upleft_edges, upright_edges])

        left_diffs = img[:, 1:] - img[:, :-1]
        up_diffs = img[1:, :] - img[:-1, :]
        upleft_diffs = img[1:, 1:] - img[:-1, :-1]
        upright_diffs = img[1:, :-1] - img[:-1, 1:]

        straight_diffs = np.concatenate(
            [
                np.linalg.norm(left_diffs, 2, axis=2).flatten(),
                np.linalg.norm(up_diffs, 2, axis=2).flatten(),
            ]
        )
        diagonal_diffs = np.concatenate(
            [
                np.linalg.norm(upleft_diffs, 2, axis=2).flatten(),
                np.linalg.norm(upright_diffs, 2, axis=2).flatten(),
            ]
        )

        straight_capacities = cls.GAMMA * np.exp(-beta * np.square(straight_diffs))
        diagonal_capacities = cls.GAMMA * np.exp(-beta * np.square(diagonal_diffs))

        cls._n_link_edges = np.concatenate([straight_edges, diagonal_edges])
        cls._n_link_capacities = np.concatenate(
            [straight_capacities, diagonal_capacities]
        )

        return cls._n_link_edges, cls._n_link_capacities

    @classmethod
    def t_link_edges_and_capacities(cls, img, mask, bgGMM, fgGMM):
        rows, cols, _ = img.shape
        src_vertex = rows * cols
        sink_vertex = src_vertex + 1

        flat_mask = mask.reshape(-1)
        flat_img = img.reshape(-1, 3)
        pr_pixels_idxs = np.where(
            np.logical_or(flat_mask == GC_PR_BGD, flat_mask == GC_PR_FGD)
        )[0]
        bgd_pixels_idxs = np.where(flat_mask == GC_BGD)[0]
        fgd_pixels_idxs = np.where(flat_mask == GC_FGD)[0]

        pr_pixels = flat_img[pr_pixels_idxs]

        pr_src_edges = np.c_[np.full(pr_pixels_idxs.size, src_vertex), pr_pixels_idxs]
        pr_sink_edges = np.c_[np.full(pr_pixels_idxs.size, sink_vertex), pr_pixels_idxs]
        bgd_src_edges = np.c_[
            np.full(bgd_pixels_idxs.size, src_vertex), bgd_pixels_idxs
        ]
        bgd_sink_edges = np.c_[
            np.full(bgd_pixels_idxs.size, sink_vertex), bgd_pixels_idxs
        ]
        fgd_src_edges = np.c_[
            np.full(fgd_pixels_idxs.size, src_vertex), fgd_pixels_idxs
        ]
        fgd_sink_edges = np.c_[
            np.full(fgd_pixels_idxs.size, sink_vertex), fgd_pixels_idxs
        ]

        t_link_edges = np.concatenate(
            [
                pr_src_edges,
                pr_sink_edges,
                bgd_src_edges,
                bgd_sink_edges,
                fgd_src_edges,
                fgd_sink_edges,
            ]
        )

        pr_src_capacities = -np.log(bgGMM.calc_probs(pr_pixels))
        pr_sink_capacities = -np.log(fgGMM.calc_probs(pr_pixels))
        bgd_src_capacities = np.full(bgd_pixels_idxs.size, 0)
        bgd_sink_capacities = np.full(bgd_pixels_idxs.size, cls.K)
        fgd_src_capacities = np.full(fgd_pixels_idxs.size, cls.K)
        fgd_sink_capacities = np.full(fgd_pixels_idxs.size, 0)

        t_link_capacities = np.concatenate(
            [
                pr_src_capacities,
                pr_sink_capacities,
                bgd_src_capacities,
                bgd_sink_capacities,
                fgd_src_capacities,
                fgd_sink_capacities,
            ]
        )

        return t_link_edges, t_link_capacities


def partition_pixels(img, mask):
    bg_pixels = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg_pixels = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]
    return bg_pixels, fg_pixels


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5, components=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Required patch
    # https://moodle.tau.ac.il/mod/forum/discuss.php?d=71803#p100498
    w -= x
    h -= y

    # Initalize the inner square to Foreground
    mask[y : y + h, x : x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    img = np.asarray(img, dtype=np.float64)
    bgGMM, fgGMM = initalize_GMMs(img, mask, components)

    for i in range(n_iter):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    img = img.astype(np.uint8)
    mask[mask == GC_PR_BGD] = GC_BGD
    mask[mask == GC_PR_FGD] = GC_FGD

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components: int = 5):
    # TODO: implement initalize_GMMs
    bg_pixels, fg_pixels = partition_pixels(img, mask)
    bgGMM = GaussianMixture(n_components).init(bg_pixels)
    fgGMM = GaussianMixture(n_components).init(fg_pixels)
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    bg_pixels, fg_pixels = partition_pixels(img, mask)
    bgGMM.update(bg_pixels)
    fgGMM.update(fg_pixels)
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    graph = GrabcutGraph.construct_graph(img, mask, bgGMM, fgGMM)
    return graph.mincut()


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    updated_mask = mask.copy().reshape(-1)
    updated_mask[mincut_sets[0]] = GC_PR_FGD
    updated_mask[mincut_sets[1]] = GC_BGD
    return updated_mask.reshape(mask.shape)


def check_convergence(energy):
    # TODO: implement convergence check
    print(f"energy={energy}")
    global previous_energy
    convergence = (
        False if previous_energy is None else (previous_energy - energy) < EPSILON
    )
    previous_energy = energy
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation
    accuracy = np.count_nonzero(predicted_mask == gt_mask) / predicted_mask.size

    intersection = np.logical_and(predicted_mask, gt_mask)
    union = np.logical_or(predicted_mask, gt_mask)
    jaccard = intersection.sum() / union.sum()

    return accuracy, jaccard


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_name",
        type=str,
        default="banana1",
        help="name of image from the course files",
    )
    parser.add_argument("--eval", type=int, default=1, help="calculate the metrics")
    parser.add_argument(
        "--input_img_path",
        type=str,
        default="",
        help="if you wish to use your own img_path",
    )
    parser.add_argument(
        "--use_file_rect", type=int, default=1, help="Read rect from course files"
    )
    parser.add_argument(
        "--rect",
        type=str,
        default="1,1,100,100",
        help="if you wish change the rect (x,y,w,h",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=5,
        help="if you wish to change the number of components",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="if you wish to save the output file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == "":
        input_path = f"data/imgs/{args.input_name}.jpg"
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(
            map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(" "))
        )
    else:
        rect = tuple(map(int, args.rect.split(",")))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    start_time = time.time()
    mask, bgGMM, fgGMM = grabcut(img, rect, components=args.components)
    elapsed = time.time() - start_time
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f"data/seg_GT/{args.input_name}.bmp", cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(
            f"Accuracy={acc:.3f}, Jaccard={jac:.3f}, elapsed={elapsed:.3f}, components={args.components}"
        )

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    if args.output_file:
        cv2.imwrite(args.output_file, img_cut)

    cv2.imshow("Original Image", img)
    cv2.imshow("GrabCut Mask", 255 * mask)
    cv2.imshow("GrabCut Result", img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
