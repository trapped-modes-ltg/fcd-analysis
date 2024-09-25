import matplotlib.pyplot as plt
from pydata.video import Video
from pyfcd.layer import Layer
from pydata.image import *
from pyfcd.fcd import FCD
from scipy.fft import fftfreq, fft

if __name__ == '__main__':
    """
    # Ejemplo para la gota de agua.
    displaced_image = Image("C:/Users/Marcelo/Documents/Facultad/Laboratorio 6 y 7/Gota de Agua FCD/30_05/Imagenes/Displaced/0528-Barrido_una_gota_6mm.png ", roi=-1)
    reference_image = Image("C:/Users/Marcelo/Documents/Facultad/Laboratorio 6 y 7/Gota de Agua FCD/30_05/Imagenes/Referencias/0528-Barrido_una_gota_6mm.png", roi=displaced_image.roi)

    layers = [Layer(5e-3, "Air"), Layer(1e-3, "Glass"), Layer(0, "Distilled_water"), Layer(1, "Air")]
    fcd = FCD(reference_image.windowed(), layers=layers, square_size=1e-3)
    height_field = fcd.analyze(displaced_image.windowed())
    height_field.show_slice()
    """

    """
    # Ejemplo para la jeringa.
    video = Video(start_frame=1657)
    roi = video.current_frame.select_roi()
    reference_image = Image(roi=roi)

    fcd = FCD(reference_image.windowed(), square_size=0.0022, height=0.0323625)
    def apply_fcd(frame):
        height_field = fcd.analyze(frame.windowed(), full_output=False)
        return height_field
    video.show_movie(pre_process=apply_fcd)
    """

    """
    # Ejemplo para solo el trackeo del toroide.
    template = Image("C:/Users/Marcelo/Documents/Facultad/Laboratorio 6 y 7/Github/Laboratorio-6-y-7/Tracking de estructuras/Toroide.png", roi=-1)
    template.center_toroid()
    template.rotated(angle=45)

    video = Video("E:/Ignacio Hernando/0611/toroide_con_arena_forzado/202406_1445", start_frame=657)
    
    def pre_process(frame):
        frame.track(template.processed)
        return frame.processed
    video.show_movie(pre_process=pre_process)
    """

    # Ejemplo para el toroide.
    template = Image("C:/Users/Marcelo/Documents/Facultad/Laboratorio 6 y 7/Github/Laboratorio-6-y-7/Tracking de estructuras/Toroide.png", roi=-1)
    template.center_toroid()
    template.rotated(angle=45)

    c = template.roi[2] // 2
    R = int(45 / 1.41)
    reference = Image("C:/Users/Marcelo/Documents/Facultad/Laboratorio 6 y 7/Github/Laboratorio-6-y-7/Tracking de estructuras/Referencia.png", roi=template.roi)
    layers = [Layer(5.7e-2, "Air"), Layer(1.2e-2, "Acrylic"), Layer(4.3e-2, "Distilled_water"), Layer(80e-2, "Air")]
    fcd = FCD(reference.windowed()[c-R:c+R, c-R:c+R], square_size=0.0022, layers=layers)
    mask = reference.make_circular_mask(radius=45)

    video = Video("E:/Ignacio Hernando/0611/toroide_con_arena_forzado/202406_1445", start_frame=657)

    centers = []

    def pre_process(frame):
        frame.track(template.processed)
        reference.roi = frame.roi  # Ir desplazando también el reference.
        frame.masked(mask, reference.windowed())
        height_field = fcd.analyze(frame.windowed()[c-R:c+R, c-R:c+R], full_output=False)
        centers.append(height_field[R, R])
        return height_field
    video.show_movie(pre_process=pre_process, end_frame=1200)

    ### Análisis de los datos procesados. ###

    centers = np.array(centers)
    times = np.arange(len(centers)) / video.fps * 1000

    plt.subplot(211)
    plt.plot(times, centers*1e3)
    plt.xlabel("Tiempo [ms]")
    plt.ylabel("Altura [mm]")

    plt.subplot(212)
    freqs = fftfreq(len(times), times[1]-times[0]) * 1000
    ffts = fft(centers * 1e3)
    plt.plot(freqs[:len(times)//2][::-1], np.abs(ffts)[:len(times)//2][::-1])
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [mm]")
    plt.show()
