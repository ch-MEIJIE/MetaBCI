import tobii_research as tr
import time
from metabci.brainflow.amplifiers import BaseAmplifier, Marker
from metabci.brainflow.logger import get_logger, disable_log

class TobiiSpectrum(BaseAmplifier):
    """Tobii Spectrum eye tracker."""
    def __init__(
            self,
            eye_tracker_name: str,
            sampling_rate: int,
            ):
        super().__init__()
        self.eye_tracker_name = eye_tracker_name
        self.sampling_rate = sampling_rate
        self.gaze_data = None
        self.trigger_data = None
    
    def connect(self):
        # find eye tracker in local network for three times
        for i in range(3):
            eye_trackers = tr.find_all_eyetrackers()
            if len(eye_trackers) != 0:
                self.eye_tracker = eye_trackers[0]
                break
            time.sleep(1)
    
    def gaze_data_callback(self, gaze_data_pkg):
        self.gaze_data = gaze_data_pkg
    
    def trigger_callback(self, trigger_data_pkg):
        self.trigger_data = trigger_data_pkg
    
    def start_stream(self):
        self.eye_tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA,
            self.gaze_data_callback,
            as_dictionary=True
            )
        self.eye_tracker.subscribe_to(
            tr.EYETRACKER_EXTERNAL_SIGNAL,
            self.trigger_callback,
            as_dictionary=True
            )
        self.start()
    
    def stop_stream(self):
        self.stop()
        self.eye_tracker.unsubscribe_from(
            tr.EYETRACKER_GAZE_DATA,
            self.gaze_data_callback)
        self.eye_tracker.unsubscribe_from(
            tr.EYETRACKER_EXTERNAL_SIGNAL,
            self.trigger_callback
            )
    
    def recv(self):
        # Get a single gaze data frame
        while self.gaze_data is None:
            continue
        gaze_data_frame = self.gaze_data.copy()
        self.gaze_data = None # clear gaze data
        # Channel 1: left eye, Channel 2: right eye Channel 3: trigger
        sample = list()
        sample.extend([*gaze_data_frame["left_gaze_point_on_display_area"]])
        sample.extend([*gaze_data_frame["right_gaze_point_on_display_area"]])
        if self.trigger_data is None:
            sample.append(0)
        else:
            sample.append(self.trigger_data["value"])
            self.trigger_data = None
        return [sample]
    




if __name__ == "__main__":
    disable_log()
    tobii = TobiiSpectrum("Tobii Spectrum", 100)
    tobii.connect()
    tobii.start_stream()
    time.sleep(1)

    time.sleep(2)
    tobii.stop_stream()
  
    