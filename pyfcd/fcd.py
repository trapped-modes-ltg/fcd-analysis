from skimage.restoration import unwrap_phase
from scipy.fft import fft2, fftshift, ifft2
from matplotlib.colors import LogNorm
from pyfcd.height_map import HeightMap
from pyfcd.carriers import Carrier
import pyfcd.fourier_space as fs
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftn, ifftn, fftshift
from scipy.signal.windows import tukey


class FCD:
    def __init__(self, reference_image, layers=None, height=None, calibration_factor=None, square_size=None):
        self._reference_image = reference_image.astype(np.float32)
        self.calibration_factor = calibration_factor
        self._carriers = self._calculate_carriers(square_size)
        self.layers = layers  # From the pattern to the camera.
        if height is not None:
            if layers is None:
                self.height = height  # Effective height times alpha.
            else:
                self.height = height
                raise Warning("Provide either height or layers, not both.")
        else:
            if layers is None:
                self.height = 1
            else:
                self.height = self._height_from_layers()

    def _height_from_layers(self):  # TODO: No sé si hay vidrio por ejemplo entre el agua y la cámara cómo cambia esto.
        fluid = self.layers[-2]
        before_camera = self.layers[-1]
        alpha = 1 - before_camera.refractive_index / fluid.refractive_index

        height = 0
        for layer in self.layers[:-1]:
            height += layer.effective_height(fluid.refractive_index)

        return alpha * height

    @property
    def reference_image(self):
        return self._reference_image

    @property
    def carriers(self):
        return self._carriers

    # TODO: Esto no sé, porque habría que cambiar también el square_size y recalibrar todo.
    # @reference_image.setter  
    # def reference_image(self, new_reference_image):
    #     self._reference_image = new_reference_image
    #     self.calibration_factor = None
    #     self._carriers = self._calculate_carriers()

    def _calculate_carriers(self, square_size):
        peaks = fs.find_peaks(self._reference_image)
        peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2
        if self.calibration_factor is None:
            self.calibration_factor = self._find_calibration_factor(peaks, square_size)

        carriers = [Carrier(self._reference_image, self.calibration_factor, peak, peak_radius) for peak in peaks]

        return carriers

    def _find_calibration_factor(self, peaks, square_size):
        if square_size is None:
            calibration_factor = 1
            return calibration_factor
        else:
            checkerboard_frequencies_pixels = fs.pixel_to_wavenumber(self._reference_image.shape, peaks)
            checkerboard_wavelength_pixels = 2 * np.pi / np.mean(np.abs(checkerboard_frequencies_pixels))  # Assumed to be squares.
            checkerboard_wavelength_meters = 2 * square_size
            calibration_factor = checkerboard_wavelength_meters / checkerboard_wavelength_pixels
            return calibration_factor

    def _find_phases(self, displaced_image_fft, unwrap):  # TODO: En selecciones no cuadradas parece no funcionar muy bien el unwrap.
        phases = np.zeros((2, *displaced_image_fft.shape))
        for i, carrier in enumerate(self._carriers):
            angles = -np.angle(ifft2(displaced_image_fft * carrier.mask) * carrier.ccsgn)
            phases[i] = unwrap_phase(angles) if unwrap else angles
        return phases

    def _find_displacement_field(self, phases):
        det_a = self._carriers[0].frequencies[1] * self._carriers[1].frequencies[0] - \
                self._carriers[0].frequencies[0] * self._carriers[1].frequencies[1]
        u = (self._carriers[1].frequencies[0] * phases[0] - self._carriers[0].frequencies[0] * phases[1]) / det_a
        v = (self._carriers[0].frequencies[1] * phases[1] - self._carriers[1].frequencies[1] * phases[0]) / det_a
        return np.array([u, v])

    def analyze(self, displaced_image, unwrap=True, full_output=True, fixed_loc=None, fixed_val=0):
        displaced_image_fft = fft2(displaced_image.astype(np.float32))
        phases = self._find_phases(displaced_image_fft, unwrap)
        displacement_field = self._find_displacement_field(phases)
        height_gradient = -displacement_field / self.height

        height_map = fs.integrate_in_fourier(*height_gradient, self.calibration_factor)
        if fixed_loc is not None:
            height_map = height_map - height_map[fixed_loc[0], fixed_loc[1]] + fixed_val

        return HeightMap(height_map, phases, self.calibration_factor) if full_output else height_map

    def show_carriers(self):
        plt.imshow(self._reference_image, cmap='gray')

        module = 0.25 * self._reference_image.shape[0]

        for i, carrier in enumerate(self._carriers):  # TODO: Tal vez este bloque en Carrier().
            k = carrier.frequencies.copy()
            k /= np.linalg.norm(k)

            start_point = carrier.pixels
            end_point = carrier.pixels + module * k
            plt.arrow(start_point[1], start_point[0], end_point[1] - start_point[1], end_point[0] - start_point[0],
                      head_width=10, head_length=10, fc='red', ec='red')
            plt.text(end_point[1] * 1.1, end_point[0] * 1.1, f'k_{i + 1}', color='red', fontsize=12, ha='left',
                     va='center')

        plt.axis('off')
        plt.show()

    def show_fft(self, displaced_image):
        fig, axs = plt.subplots(1, 2)
        fft_reference_image = fftshift(np.abs(fft2(self._reference_image - np.mean(self._reference_image))))
        fft_displaced_image = fftshift(np.abs(fft2(displaced_image - np.mean(displaced_image))))

        axs[0].imshow(fft_reference_image, cmap='gray', norm=LogNorm())
        axs[0].set_title('Imagen de referencia')

        axs[1].imshow(fft_displaced_image, cmap='gray', norm=LogNorm())
        axs[1].set_title('Imagen deformada')

        x_ticks = np.linspace(0, self._reference_image.shape[1] - 1, 5, dtype=int)
        y_ticks = np.linspace(0, self._reference_image.shape[0] - 1, 5, dtype=int)
        ticks = fs.pixel_to_wavenumber(self._reference_image.shape, np.array([y_ticks, x_ticks]).T, self.calibration_factor)

        for ax in axs:   # TODO: Tal vez este bloque en Carrier().
            for carrier in self._carriers:
                circle = plt.Circle(carrier.pixels[::-1], carrier.radius, color='r', fill=False)
                ax.add_patch(circle)
            ax.set_xticks(x_ticks, [f'{tick[0] / 1e3:.2f}' for tick in ticks])
            ax.set_yticks(y_ticks, [f'{tick[1] / 1e3:.2f}' for tick in ticks])
            ax.set_xlabel("k_x [1/mm]")
            ax.set_ylabel("k_y [1/mm]")

        plt.show()
