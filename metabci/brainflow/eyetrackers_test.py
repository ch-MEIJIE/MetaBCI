import numpy as np
import time
import math
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import Marker
from metabci.brainflow.eyetrackers import TobiiSpectrum
from metabci.brainflow.workers import ProcessWorker

STIM_POS = {
    "0": ((0.465, 0.149), (0.544, 0.273)),
    "1": ((0.383, 0.727), (0.453, 0.850)),
    "2": ((0.222, 0.329), (0.292, 0.455)),
    "3": ((0.708, 0.329), (0.778, 0.455)),
    "4": ((0.708, 0.547), (0.778, 0.669)),
    "5": ((0.222, 0.547), (0.292, 0.669)),
    "6": ((0.465, 0.547), (0.544, 0.669)),
    "7": ((0.546, 0.727), (0.616, 0.850)),
    "8": ((0.305, 0.305), (0.694, 0.696))
    }

class Tobii_IVT_filter:
    '''filter_mode: fixation/attention/user_define'''

    def __init__(self, sample_fps, filter_mode, seating_dis=600, screen=(606, 354), resolution=[2560, 1440], **filter_kwargs):
        self.sample_fps = sample_fps
        self.filter_mode = filter_mode
        self.seat_dis = seating_dis
        self.resolution = resolution
        self.reference = np.array([self.resolution[0]/2, self.resolution[1]/2])  # 参考点为屏幕中心
        self.screen_w = screen[0]
        self.screen_h = screen[1]
        if filter_mode == 'fixation':
            self.fixation_filter()  # 设置默认参数
        elif filter_mode == 'attention':
            self.attention_filter() # 设置默认参数
        else:
            self.user_defined_filter(**filter_kwargs)   # **将字典转换为关键字参数

    def fixation_filter(self):
        self.max_gap_length = 75    # 75ms
        self.noise_window_size = 2 
        self.velocity_window_size = 20  # 20ms
        self.velocity_threshold = 40    # 40°/s    
        self.merge_max_time = 75
        self.merge_max_angle = 0.5
        self.discard_min_duration = 0   # 去除短注视

    def attention_filter(self):
        self.max_gap_length = 75
        self.noise_window_size = 3
        self.velocity_window_size = 20  
        self.velocity_threshold = 100
        self.merge_max_time = 75
        self.merge_max_angle = 0.5                                                                                          
        self.discard_min_duration = 60

    def user_defined_filter(self, **filter_kwargs):
        self.max_gap_length = filter_kwargs['max_gap_length']
        self.noise_window_size = filter_kwargs['noise_window_size']
        self.velocity_window_size = filter_kwargs['velocity_window_size']
        self.velocity_threshold = filter_kwargs['velocity_threshold']
        self.merge_max_time = filter_kwargs['merge_max_time']
        self.merge_max_angle = filter_kwargs['merge_max_angle']
        self.discard_min_duration = filter_kwargs['discard_min_duration']

    def filter_process(self, fixations):    # fixations: 注视点列表
        '''
        滤波器执行过程
        input: 注视点列表，(n,2)的列表
        output: 平均注视点坐标 [x,y]
        '''
        print('raw len fixations', len(fixations))
        fixations_gap = self.GapFill(fixations)
        # fixations_merge = self.MovingMedian(fixations_gap) 
        fixations_merge = fixations_gap
        velocity_list = self.VelocityCal(fixations_merge)   # 计算得到速度列表
        mean_fixation = self.FixClassifier(fixations_merge, velocity_list)  # 分类注视点
        return mean_fixation

    def GapFill(self, fixations):
        '''
        用线性插值填充nan值
        '''
        # 小于interval_num 需要内插
        interval_num = self.sample_fps * self.max_gap_length / 1000 
        fix_len = len(fixations)
        fixations = np.array(fixations) # 转换为numpy数组
        zero_index = np.all(fixations == 0, axis=1)    # 返回一个布尔数组，True表示全为0
        # FT定义为开始 TF定义为结束
        start_idx, end_idx = [], []
        for ii in range(fix_len-2):
            if (zero_index[ii] == False) & (zero_index[ii+1] == True):
                start_idx.append(ii)
            if (zero_index[ii] == True) & (zero_index[ii+1] == False):
                end_idx.append(ii)
        for start, end in zip(start_idx, end_idx):
            zero_len = end - start
            if zero_len < interval_num:  #　线性插值
                px = [fixations[start][0], fixations[end+1][0]]
                py = [fixations[start][1], fixations[end+1][1]]
                interx = ((px[1] - px[0]) * np.arange(zero_len+1) / float(zero_len+1) + px[0]).tolist()
                intery = ((py[1] - py[0]) * np.arange(zero_len+1) / float(zero_len+1) + py[0]).tolist()
                for ii in range(1, len(interx)):
                    fixations[start+ii] = [interx[ii], intery[ii]]
        return fixations

    def MovingMedian(self, fixations):
        '''
        类似于中值滤波，但是不是取窗口内的中值，而是取窗口内的中位数
        '''
        fixations = np.array(fixations) # 转换为numpy数组
        num = self.noise_window_size    # 3
        for ii in range(num, len(fixations)-num):   # 3 ~ len-3（窗口）
            fix_slice = fixations[ii-num: ii+num+1] # ii-3 ~ ii+3
            det_x = fix_slice[:,0].max() - fix_slice[:,0].min() # x极差
            det_y = fix_slice[:,1].max() - fix_slice[:,1].min() # y极差
            if det_x > det_y:   # 选择极差大的维度
                median_idx = np.argsort(fix_slice[:,0])[num]    # 中位数对应的索引
            else:
                median_idx = np.argsort(fix_slice[:,1])[num]    # 中位数对应的索引
            fixations[ii, :] = fix_slice[median_idx, :] # 替换为中位数
        return fixations.tolist()
    
    def calculate_angle(self, start, end):
        '''
        计算两点之间的夹角，以屏幕中心为出发点
        start,end为np.array
        '''
        def calculate_dist(start, end):
            '''
            返回两点之间的欧氏距离，单位mm
            '''
            dist = np.sqrt(sum(np.power((end - start), 2)))
            pixels_to_mm = self.screen_w / self.resolution[0]
            dist = dist * pixels_to_mm
            return dist
        def calculate_angle(edge_a, edge_b, edge_c):
            '''
            返回两向量之间的夹角(角A)
            '''
            cos_angle = (edge_b**2 + edge_c**2 - edge_a**2) / (2 * edge_b * edge_c)
            angle = math.degrees(math.acos(cos_angle))
            return angle
        edge_c = np.sqrt(calculate_dist(self.reference, start)**2 + self.seat_dis**2)
        edge_b = np.sqrt(calculate_dist(self.reference, end)**2 + self.seat_dis**2)
        edge_a = calculate_dist(start, end)
        angle = calculate_angle(edge_a, edge_b, edge_c)
        return angle

    def VelocityCal(self, fixations):
        '''
        计算视线移动速度
        input: fixations注视点列表
        return: 速度列表vel_list
        '''
        fixations = np.array(fixations) # 转换为numpy数组
        num = int(self.sample_fps * self.velocity_window_size / 1000)   # 采样频率 * 窗口大小 / 1000，20ms
        vel_list = []
        print('len fixations', len(fixations))
        for ii in range(len(fixations) - num):
            # 换算成像素
            start = np.array(fixations[ii])
            end = np.array(fixations[ii+num])
            vel = self.calculate_angle(start, end)*1000/self.velocity_window_size   # 计算速度，单位°/s
            vel_list.append(vel)   # 添加到速度列表
        return vel_list

    def FixClassifier(self, fixations, velocity):
        '''
        根据视线移动速度，分类注视点，计算平均注视位置
        '''
        fixations = np.array(fixations) # 注视点转换为numpy数组
        velocity = np.array(velocity)   # 注视点速度转换为numpy数组
        # nan替换为100，不替换nan比任何数都小
        index = np.where(np.isnan(velocity) == True)    # 返回速度为nan的索引
        velocity[index] = 100   #　nan替换为100
        fix_idx = velocity > self.velocity_threshold    # 速度大于阈值的索引
        # FT定义为开始 TF定义为结束
        start_idx, end_idx = [], []
        for ii in range(len(velocity) - 1): # 根据阈值查找稳定的注视点
            if ii ==0 and fix_idx[ii] == False:
                start_idx.append(ii)
            if (fix_idx[ii] == True) & (fix_idx[ii+1] == False):
                start_idx.append(ii)
            if (fix_idx[ii] == False) & (fix_idx[ii+1] == True):
                end_idx.append(ii)
        if fix_idx[len(velocity)-1] == False:
            end_idx.append(len(velocity)-1)
        num_fixation = max(len(start_idx),len(end_idx))   # 稳定注视点的数量
        num = int(self.sample_fps * self.discard_min_duration / 1000)
        merged_fixations = []
        if len(start_idx) > 0 and len(end_idx) > 0: # 如果有fixation point
            for start, end in zip(start_idx, end_idx):
                if end - start >= num:
                    segment_fixations = fixations[start:end]    # 注视点片段
                    merged_fixations.append(segment_fixations)   # 计算平均注视点
            merged_fixations = np.vstack(merged_fixations)
            mean_fixations = np.mean(merged_fixations, axis=0)
            print('mean_fixations:', mean_fixations)
            return mean_fixations
        else:   # 否则直接返回0
            return [0,0]

class TobiiWorker(ProcessWorker):
    def __init__(
            self,
            timeout: float = 0.001,
            name: str | None = None,
            stim_locs: dict | None = None,
            eye_tracker_srate: int = 120,
            lsl_source_id: str = 'Meta_online',
    ):
        super().__init__(timeout, name)
        self.right_eye = None
        self.left_eye = None
        self.stim_locs = stim_locs
        self.eye_filter = Tobii_IVT_filter(eye_tracker_srate, 'fixation')
        self.lsl_source_id = lsl_source_id

    def pre(self):
        # New a lsl outlet stream for sending the predicted label
        info = StreamInfo(
            name='BrainDriving',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id
        )
        self.outlet = StreamOutlet(info)
        print('Created a lsl outlet stream successfully')
        print('waiting for connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected with marker receiver successfully')

    def consume(self, data):
        data = np.array(data, dtype=np.float64)
        data = data[:, :2]
        fixation = self.eye_filter.filter_process(data)
        p_label = self.find_squares(fixation)
        print('Predicted label: {}'.format(p_label))
        if self.outlet.have_consumers() and p_label != 8 and p_label is not None:
            self.outlet.push_sample([p_label])

    def post(self):
        pass

    def find_squares(self, point):
        x, y = point[0], point[1]
        for key, ((x1, y1), (x2, y2)) in self.stim_locs.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return int(key)
        return None


if __name__ == "__main__":
    eye_tracker_srate = 120
    stim_label = [i for i in range(1, 255)]
    tobii_worker = TobiiWorker(0.01, "tobii_worker", stim_locs=STIM_POS)
    tobii_marker = Marker(
        interval=[0, 0.5],
        srate=eye_tracker_srate,
        events=stim_label,
    )
    tobii = TobiiSpectrum("Tobii Spectrum", eye_tracker_srate)
    tobii.connect()
    tobii.register_worker("tobii_worker", tobii_worker, tobii_marker)
    tobii.up_worker("tobii_worker")
    time.sleep(0.5)
    tobii.start_stream()
    input('press any key to close\n')
    tobii.down_worker("tobii_worker")
    time.sleep(5)
    tobii.stop_stream()
