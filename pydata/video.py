import matplotlib.pyplot as plt
from pydata.image import Image
from tkinter import filedialog
from time import sleep
import os

allowed_formats = (".jpg", ".png", ".tiff", ".bmp")  # TODO: implementar lectura de imágenes a color.


class Video:
    def __init__(self, directory_path=None, start_frame=0, roi=None):
        self.directory_path = directory_path if directory_path else filedialog.askdirectory(title="Selecciona una carpeta")
        self.frames = sorted(self.__get_frames())
        self._current_frame_index = start_frame
        self.current_frame = self.__begin_video(roi)

        self.metadata = self.__read_metadata()
        self.fps = int(self.metadata["Record Rate(fps)"])
        self.length = len(self.frames)

    def __read_metadata(self):
        metadata_file = self.directory_path.split("/")[-1] + ".cih"
        metadata = {}
        with open(os.path.join(self.directory_path, metadata_file), 'r') as file:
            for line in file:
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
        return metadata

    def __get_frames(self):
        return [file for file in os.listdir(self.directory_path) if file.endswith(allowed_formats)]

    def __begin_video(self, roi):
        return Image(self._frame_path(), roi)

    def _frame_path(self, frame_index=None):
        frame_index = frame_index if frame_index is not None else self._current_frame_index
        return os.path.join(self.directory_path, self.frames[frame_index])

    def read_frame(self, frame_index=None):
        self.current_frame.path = self._frame_path(frame_index)

    @property
    def current_frame_index(self):
        return self._current_frame_index

    @current_frame_index.setter
    def current_frame_index(self, new_index):
        self._current_frame_index = new_index
        self.read_frame()

    def next_frame(self):
        self.current_frame_index += 1
        return self.current_frame

    def play(self, end_frame=None):
        if end_frame is None:
            end_frame = self.length
        while self.current_frame_index < end_frame - 1:
            yield self.next_frame()

    # TODO: Se podría agregar una función self.pre_process() llamada antes de show donde se haga el track y se pueda modificar desde afuera de la clase tipo decorator.
    def show_movie(self, end_frame=None, pre_process=None):
        plt.ion()
        fig, ax = plt.subplots()
        ax.axis("off")

        fig.canvas.draw()
        for frame in self.play(end_frame):
            ax.clear()
            if pre_process is not None:
                to_show = pre_process(frame)  # TODO: Podría recibir directamente una fig, ax y retornar eso.
                ax.imshow(to_show)
            else:
                frame.show(axis=ax, show_plot=False)
            fig.canvas.flush_events()
            sleep(0.01)

        plt.ioff()
        plt.show()
