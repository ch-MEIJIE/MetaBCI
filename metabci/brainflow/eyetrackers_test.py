import numpy as np
import time
from metabci.brainflow.eyetrackers import Marker
from eyetrackers import TobiiSpectrum
from metabci.brainflow.workers import ProcessWorker

class TobiiWorker(ProcessWorker):
    def __init__(
            self,
            timeout: float = 0.001,
            name: str | None = None):
        super().__init__(timeout, name)
        self.right_eye = None
        self.left_eye = None
    
    def pre(self):
        pass

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        self.right_eye = data[0:1, :]
        self.left_eye = data[2:3, :]
        # calculate the average of the right and left eye
        right_eye_avg = np.mean(self.right_eye, axis=1)
        left_eye_avg = np.mean(self.left_eye, axis=1)
        print(f"Right eye average: {right_eye_avg}")
        print(f"Left eye average: {left_eye_avg}")

    def post(self):
        pass

if __name__ == "__main__":
    stim_label = [i for i in range(1, 255)]
    tobii_worker = TobiiWorker(0.01, "tobii_worker")
    tobii_marker = Marker(
        interval=[0, 0.4],
        srate=600,
        events=stim_label,
    )
    tobii = TobiiSpectrum("Tobii Spectrum", 120)
    tobii.connect()
    tobii.register_worker("tobii_worker", tobii_worker, tobii_marker)
    tobii.up_worker("tobii_worker")
    time.sleep(0.5)
    tobii.start_stream()
    input('press any key to close\n')
    tobii.down_worker("tobii_worker")
    time.sleep(5)
    tobii.stop_stream()