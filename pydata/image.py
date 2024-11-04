from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from tkinter import filedialog
from pydata.roi import ROI
import numpy as np
import cv2
from matplotlib.widgets import Slider, Button
import warnings


class Image:
    def __init__(self, image_path=None, roi=None, edge_configurations=None):
        self._path = image_path if image_path else filedialog.askopenfilename(title="Seleccionar imagen")
        self._original = self._read_image()
        self._roi = ROI(self, roi)  # TODO: tal vez que si tipo roi=-1 o algo así automáticamente se abra la selección. Pasar ROIs a parte de (x, y, w, h).
        if roi == -1:
            self.select_roi()
        self._processed = self.cropped()  # TODO: No me convence tener dos veces guardada la imagen, pero no se me ocurre otra forma de acumular las trasnformaciones.

        self.edge_configurations = edge_configurations or {
            'N': 11, 
            'H': 5, 
            'low_mask': 20, 
            'high_mask': 255, 
            'param1': 40, 
            'param2': 20, 
            'minRadius': 190, 
            'maxRadius': 210
        }
        
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
        window_1d_x = np.abs(tukey(self.roi[2], x_factor)) # TODO: revisar.
        window_1d_y = np.abs(tukey(self.roi[3], y_factor))
        window_2d = np.sqrt(np.outer(window_1d_y, window_1d_y))
        self._processed = window_2d * self._processed
        return self._processed

    def blurred(self, N=None, H=None): # TODO: no actualiza el self._processed, pero es que para hallar los centros no quiero que lo haga. Tal vez cambiar el nombre, para que no se confunda con el otro naming scheme. 
        N = N if N is not None else self.edge_configurations['N']
        H = H if H is not None else self.edge_configurations['H']
        return cv2.GaussianBlur(self._processed, (N, N), H)

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
        return max_loc # TODO: warning si pasan cosas raras acá. Si distancia con anterior menor a tanto (pero habría que guardar el anterior, tal vez pasar el centro como argumento optativo junto con un sigma para comparar.).

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

    def make_blurred_mask(self):  # TODO: No sé si vale la pena poner para que pasen opcionalmente otros N, H, low, high.
        _, binary = cv2.threshold(self.blurred(), self.edge_configurations['low_mask'], self.edge_configurations['high_mask'], cv2.THRESH_BINARY)
        kernel = np.ones((self.edge_configurations['N'], self.edge_configurations['N']), np.uint8)
        mask_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel).astype(np.uint8)        
        return mask_open

    def edges(self):
        return cv2.Canny(self.make_blurred_mask(), 100, 200)

    def get_circle_centers(self): 
        circles = cv2.HoughCircles(self.edges(), cv2.HOUGH_GRADIENT, dp=1, minDist=min(self.roi[2:]),
                                   param1=self.edge_configurations["param1"],
                                   param2=self.edge_configurations["param2"],
                                   minRadius=self.edge_configurations["minRadius"],
                                   maxRadius=self.edge_configurations["maxRadius"])
        return circles[0] if circles is not None else None # TODO: creo que acá se puede compactar este if.

    def center_toroid(self, use_sliders=False):
        if use_sliders: # TODO: no soy super fan de tener todo esto acá abajo. Probablemente lo encapsule en alguna otra función para que quede más prolijo (pero es puramente estético).
            fig, axes = plt.subplots(1, 3, figsize=(15, 8))
            plt.subplots_adjust(left=0.05, bottom=0.4, right=0.95, wspace=0.5)

            def update_edge_configurations():
                self.edge_configurations['N'] = int(slider_N.val)
                self.edge_configurations['H'] = int(slider_H.val)
                self.edge_configurations['low_mask'] = int(slider_low_mask.val)
                self.edge_configurations['high_mask'] = int(slider_high_mask.val)
                self.edge_configurations['param1'] = int(slider_param1.val)
                self.edge_configurations['param2'] = int(slider_param2.val)
                self.edge_configurations['minRadius'] = int(slider_minRadius.val)
                self.edge_configurations['maxRadius'] = int(slider_maxRadius.val)
            
            def update(val):
                update_edge_configurations()
                blurred_image = self.blurred()
                mask = self.make_blurred_mask()
                edges = self.edges()
                circles = self.get_circle_centers()
                
                axes[0].cla()
                axes[0].imshow(blurred_image, cmap='gray')
                axes[1].cla()
                axes[1].imshow(mask, cmap='gray')
                axes[2].cla()
                axes[2].imshow(edges, cmap='gray')

                if circles is not None:
                    for circle in circles:
                        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                        axes[0].add_patch(plt.Circle((x, y), r, color='red', fill=False))
                        axes[1].add_patch(plt.Circle((x, y), r, color='red', fill=False))
                        axes[2].add_patch(plt.Circle((x, y), r, color='red', fill=False))
                    
                for ax in axes:
                    ax.axis('off')
                fig.canvas.draw_idle()

            axcolor = 'lightgoldenrodyellow'
            
            ax_N = plt.axes([0.1, 0.25, 0.2, 0.03], facecolor=axcolor)
            ax_H = plt.axes([0.1, 0.2, 0.2, 0.03], facecolor=axcolor)
            slider_N = Slider(ax_N, 'N', 3, 31, valinit=self.edge_configurations['N'], valstep=2)
            slider_H = Slider(ax_H, 'H', 1, 10, valinit=self.edge_configurations['H'])

            ax_low_mask = plt.axes([0.4, 0.25, 0.2, 0.03], facecolor=axcolor)
            ax_high_mask = plt.axes([0.4, 0.2, 0.2, 0.03], facecolor=axcolor)
            slider_low_mask = Slider(ax_low_mask, 'Low Mask', 0, 255, valinit=self.edge_configurations['low_mask'])
            slider_high_mask = Slider(ax_high_mask, 'High Mask', 0, 255, valinit=self.edge_configurations['high_mask'])

            ax_param1 = plt.axes([0.7, 0.25, 0.2, 0.03], facecolor=axcolor)
            ax_param2 = plt.axes([0.7, 0.2, 0.2, 0.03], facecolor=axcolor)
            ax_minRadius = plt.axes([0.7, 0.15, 0.2, 0.03], facecolor=axcolor)
            ax_maxRadius = plt.axes([0.7, 0.1, 0.2, 0.03], facecolor=axcolor)

            slider_param1 = Slider(ax_param1, 'Param1', 0, 100, valinit=self.edge_configurations['param1'])
            slider_param2 = Slider(ax_param2, 'Param2', 0, 100, valinit=self.edge_configurations['param2'])
            slider_minRadius = Slider(ax_minRadius, 'Min Radius', 0, 300, valinit=self.edge_configurations['minRadius'])
            slider_maxRadius = Slider(ax_maxRadius, 'Max Radius', 0, 300, valinit=self.edge_configurations['maxRadius'])

            # TODO: for slider in sliders: slider.on_changed(update).
            slider_N.on_changed(update)
            slider_H.on_changed(update)
            slider_low_mask.on_changed(update)
            slider_high_mask.on_changed(update)
            slider_param1.on_changed(update)
            slider_param2.on_changed(update)
            slider_minRadius.on_changed(update)
            slider_maxRadius.on_changed(update)

            ax_button = plt.axes([0.85, 0.03, 0.1, 0.04])
            ax_button_circle = plt.axes([0.65, 0.03, 0.1, 0.04])
            button = Button(ax_button, 'Print Configurations', color=axcolor, hovercolor='0.975')
            button_circle = Button(ax_button_circle, 'Print Circle', color=axcolor, hovercolor='0.975') # o 

            def print_config(event):
                print("Current Configurations:", self.edge_configurations)
            button.on_clicked(print_config)

            def print_circle(event):
                circles = self.get_circle_centers()
                if circles is not None:
                    for circle in circles:
                        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                        print(f"Circle found in ({x}, {y}) with radius ({r} px).")    
            button_circle.on_clicked(print_circle)    

            update(None)
            plt.show()

        centers = self.get_circle_centers()
        if centers is not None:
            center = self.roi.local_to_absolute((int(centers[0, 0]), int(centers[0, 1])))
            self.roi = self.roi.new_roi_from_center(center)
        else:
            warnings.warn("No circle found.") # TODO: refinar esto (tipo el mensaje).
            default_center = (self.roi[2] // 2, self.roi[3] // 2)
            center = self.roi.local_to_absolute(default_center)
            self.roi = self.roi.new_roi_from_center(center) # TODO: esta combinación de local2absolute y from_center tal vez se podrían combinar en una tercera función que haga ambas.
    
    def show(self, axis=None, show_plot=True):
        if axis is None:
            fig, axis = plt.subplots()
        axis.imshow(self._processed)
        if show_plot:
            plt.show()
