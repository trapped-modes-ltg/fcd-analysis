import cv2


class ROI:
    def __init__(self, image, rect):
        self.image_shape = image.original.shape
        self._rect = None  # TODO: tal vez más "clean" tener funión que switchee llamada acá y por el setter.
        self.rect = rect

    def __getitem__(self, item):
        return self._rect[item]

    @property
    def rect(self):
        return self._rect

    @rect.setter
    def rect(self, new_roi):
        if isinstance(new_roi, ROI):
            self._rect = new_roi.rect  # TODO: tal vez cambiar el nombre del argumento.
        elif isinstance(new_roi, tuple):
            self._rect = new_roi  # TODO: hay que asegurarse que sean tuplas y no listas, sino problemas al copiar.
        else:
            self._rect = self.null_roi()

    def null_roi(self):
        return (0, 0, *self.image_shape)

    def local_to_absolute(self, local_coordinates):
        return [local_coordinates[0] + self._rect[0], local_coordinates[1] + self._rect[1]]

    def update_roi_center(self, new_center):
        x_old, y_old, w, h = self._rect
        cx, cy = new_center

        new_x = cx - w // 2
        new_y = cy - h // 2
        return (new_x, new_y, w, h)

    def new_roi_from_center(self, center, dimensions=None):
        if dimensions is None:
            dimensions = [self._rect[2], self._rect[3]]
        return (center[0] - dimensions[0] // 2, center[1] - dimensions[1] // 2, dimensions[0], dimensions[1])

    def new_roi_from_corner(self, corner, dimensions=None):
        if dimensions is None:
            dimensions = [self._rect[2], self._rect[3]]
        return (corner[0], corner[1], dimensions[0], dimensions[1])

    def get_center(self):
        return [self._rect[0] + self._rect[2] // 2, self._rect[1] + self._rect[3] // 2]

    def expanded(self, factor_x=5, factor_y=None):
        if factor_y is None:
            factor_y = factor_x
        new_x = max(self._rect[0] - factor_x, 0)
        new_y = max(self._rect[1] - factor_y, 0)
        new_w = min(self._rect[2] + 2 * factor_x, self.image_shape[1])
        new_h = min(self._rect[3] + 2 * factor_y, self.image_shape[0])
        return (new_x, new_y, new_w, new_h)

    def squared(self):
        x, y, w, h = self._rect
        size = min(w, h)
        x_adjusted = x + (w - size) // 2
        y_adjusted = y + (h - size) // 2
        return (x_adjusted, y_adjusted, size, size)

    def scaled(self, image, window_name="Select ROI", width=500, height=500):
        height_original, width_original = image.shape[:2]
        img_scaled = cv2.resize(image, (width, height))

        rect_scaled = cv2.selectROI(window_name, img_scaled)
        cv2.destroyWindow(window_name)

        x_scaled, y_scaled, w_scaled, h_scaled = rect_scaled
        scale_y = height_original / height
        scale_x = width_original / width

        x_original = int(x_scaled * scale_x)
        y_original = int(y_scaled * scale_y)
        w_original = int(w_scaled * scale_x)
        h_original = int(h_scaled * scale_y)
        return (x_original, y_original, w_original, h_original)

    def update(self, image, window_name="Select ROI", scale_kwargs={"width": 500, "height": 500}):
        if scale_kwargs is not None:
            rect = self.scaled(image, window_name, **scale_kwargs)
        else:
            rect = cv2.selectROI(window_name, image)  # TODO: Organizar esto un poco.
        return rect
