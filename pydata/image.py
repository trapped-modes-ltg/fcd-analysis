from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from tkinter import filedialog
from pydata.roi import ROI
import numpy as np
import cv2


class Image:
    def __init__(self, image_path=None, roi=None):
        self._path = image_path if image_path else filedialog.askopenfilename(title="Seleccionar imagen")
        self._original = self._read_image()
        self._roi = ROI(self, roi)  # TODO: tal vez que si tipo roi=-1 o algo así automáticamente se abra la selección. Pasar ROIs a parte de (x, y, w, h).
        if roi == -1:
            self.select_roi()
        self._processed = self.cropped()  # TODO: No me convence tener dos veces guardada la imagen, pero no se me ocurre otra forma de acumular las trasnformaciones.

    def _read_image(self):  # TODO: tal vez conviene que eto sea staticmethod y que se pueda usar fuera también.
        if self._path is not None:
            flag = cv2.IMREAD_UNCHANGED
            image = cv2.imread(self._path, flag)  # TODO: falla si hay tildes en el path.
            return image
        else:
            raise ValueError("Image path is not valid.")

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, new_roi):
        if isinstance(new_roi, Image):
            self.roi.rect = new_roi.roi
        else:
            self.roi.rect = new_roi
        self.reset_changes()

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, new_path):
        self._path = new_path
        self._original = self._read_image()
        self.reset_changes()

    @property
    def original(self):
        return self._original

    @original.setter
    def original(self, new_original):
        self._original = new_original
        self.reset_changes()
        self._path = None

    @property
    def processed(self):
        return self._processed

    def reset_changes(self):
        self._processed = self.cropped()

    def select_roi(self, squared=True):
        self.roi = self.roi.update(self._original)
        if squared:
            self.roi = self.roi.squared()
        return self.roi.rect

    def cropped(self, image=None):  # Esto me gustaría que fuera @property, pero no funcionaría con rotated sino.
        image = self._original if image is None else image
        x, y, w, h = self.roi
        return image[y:y + h, x:x + w]

    def rotated(self, angle, center=None):
        if center is None:
            center = self.roi.get_center()
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(self._original, rot_mat, self._original.shape[1::-1], flags=cv2.INTER_LINEAR)
        self._processed = self.cropped(result)
        return self._processed

    def windowed(self, x_factor=0.02, y_factor=None):  # TODO: pasar funión ventana (tukey por defecto).
        y_factor = y_factor if y_factor else x_factor
        window_1d_x = np.abs(tukey(self.roi[2], x_factor))
        window_1d_y = np.abs(tukey(self.roi[3], y_factor))
        window_2d = np.sqrt(np.outer(window_1d_y, window_1d_x))
        self._processed = window_2d * self._processed
        return self._processed

    def masked(self, mask, background):
        mask = mask.copy() // np.max(mask)
        self._processed = self._processed * mask + background * (1 - mask)
        return self._processed

    def add_gaussian_noise(self, mean=0, std=5):
        noise = np.random.normal(mean, std, self._processed.shape).astype(np.uint8)
        self._processed = cv2.add(self._processed, noise)
        return self._processed

    def add_salt_and_pepper_noise(self, noise_ratio=0.02):
        h, w = self._processed.shape
        noisy_pixels = int(h * w * noise_ratio)

        for _ in range(noisy_pixels):
            row, col = np.random.randint(0, h), np.random.randint(0, w)
            if np.random.rand() < 0.5:
                self._processed[row, col] = 0
            else:
                self._processed[row, col] = 255
        return self._processed

    def correlate(self, template, image=None):
        if image is None:
            image = self.cropped()
        correlation_map = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation_map)
        return max_loc

    def track(self, template, search_roi=None, blur=0, factor=5):
        if search_roi:
            self.roi = search_roi
        self.roi = self.roi.expanded(factor)

        if blur:
            image_to_correlate = cv2.GaussianBlur(self.cropped, (blur, blur), 100.90)
            template_to_correlate = cv2.GaussianBlur(template, (blur, blur), 100.90)
            top_left_local = self.correlate(template_to_correlate, image_to_correlate)
        else:
            top_left_local = self.correlate(template)

        top_left = self.roi.local_to_absolute(top_left_local)
        self.roi = self.roi.new_roi_from_corner(top_left, template.shape[::-1])

    def make_circular_mask(self, center=None, radius=None):  # TODO: esto no pega mucho acá.
        h, w = self.roi[2], self.roi[3]
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        y_mesh, x_mesh = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_mesh - center[0]) ** 2 + (y_mesh - center[1]) ** 2)
        mask = dist_from_center <= radius
        return mask

    def make_blurred_mask(self, N=15, H=10, low=80, high=255, show_mask=False):
        blurred_image = cv2.GaussianBlur(self._processed, (N, N), H)

        _, binary = cv2.threshold(blurred_image, low, high, cv2.THRESH_BINARY)
        kernel = np.ones((N, N), np.uint8)
        mask_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel).astype(np.uint8)

        if show_mask:
            plt.imshow(mask_open)
            plt.show()
        return mask_open

    def edges(self, low, high):
        return cv2.Canny(self.make_blurred_mask(), low, high)

    def get_circle_center(self, low=20, high=40, show_center=False):
        circles = cv2.HoughCircles(self.edges(low, high), cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=40, param2=20,
                                   minRadius=40, maxRadius=80)
        center = (int(circles[0][0, 0]), int(circles[0][0, 1])) if circles is not None else None
        if show_center and center:
            plt.imshow(self._processed)
            plt.scatter([center[0]], [center[1]])
            plt.show()
        return center

    def center_toroid(self):
        new_center = self.get_circle_center()
        new_center = self.roi.local_to_absolute(new_center)
        self.roi = self.roi.new_roi_from_center(new_center)

    def show(self, axis=None, show_plot=True):
        if axis is None:
            fig, axis = plt.subplots()
        axis.imshow(self._processed)
        if show_plot:
            plt.show()
