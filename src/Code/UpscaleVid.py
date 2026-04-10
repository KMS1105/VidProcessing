import sys
import os
import cv2
import torch
import numpy as np
import openvino as ov
import urllib.request
from openvino.runtime import Core
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QSpinBox, QComboBox, QTextEdit, QProgressBar, QVBoxLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from setting import get_device_info_text, get_device_recommendation

try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

def run_split_upscale(input_path, num_splits, target_parts, scale=2, tile=800, output_folder='./Vid', progress_callback=None, log_callback=None):
    core = ov.Core()
    available_devices = core.available_devices
    
    if torch.cuda.is_available():
        device_name = "CUDA"
    elif any("GPU" in d for d in available_devices):
        device_name = "GPU"
    else:
        device_name = "CPU"

    if log_callback:
        log_callback(f"🚀 가속 장치: {device_name}")

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model_filenames = {2: 'RealESRGAN_x2plus', 4: 'RealESRGAN_x4plus', 8: 'RealESRGAN_x8'}
    model_urls = {
        2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        8: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x8.pth'
    }
    
    model_base = model_filenames.get(scale, 'RealESRGAN_x2plus')
    weights_dir = os.path.join(os.getcwd(), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    pth_path = os.path.join(weights_dir, f"{model_base}.pth")
    xml_path = os.path.join(weights_dir, f"{model_base}.xml")

    if not os.path.exists(pth_path):
        if log_callback: log_callback(f"⏳ 모델 다운로드 중: {model_base}")
        urllib.request.urlretrieve(model_urls[scale], pth_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    new_h, new_w = (height // 4) * 4, (width // 4) * 4

    if device_name == "GPU":
        gpu_id = "GPU.1" if "GPU.1" in available_devices else "GPU.0"
        if not os.path.exists(xml_path):
            if log_callback: log_callback("🔄 Intel GPU 최적화 변환 중...")
            model_temp = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            loadnet = torch.load(pth_path, map_location='cpu')
            model_temp.load_state_dict(loadnet['params_ema'] if 'params_ema' in loadnet else loadnet, strict=True)
            model_temp.eval()
            dummy = torch.randn(1, 3, 256, 256)
            ov_model = ov.convert_model(model_temp, example_input=dummy)
            ov.save_model(ov_model, xml_path)
        
        ov_model_loaded = core.read_model(xml_path)
        ov_model_loaded.reshape([1, 3, new_h, new_w])
        compiled_model = core.compile_model(ov_model_loaded, gpu_id)
    else:
        model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        upsampler = RealESRGANer(scale=scale, model_path=pth_path, model=model_arch, tile=tile, half=(device_name=="CUDA"), device='cuda' if device_name=="CUDA" else 'cpu')

    frames_per_part = total_frames // num_splits
    parts_ranges = [(i * frames_per_part, (i + 1) * frames_per_part if i != num_splits - 1 else total_frames) for i in range(num_splits)]
    selected_parts = [idx for idx in target_parts if 0 <= idx < num_splits]
    total_selected_frames = sum(parts_ranges[idx][1] - parts_ranges[idx][0] for idx in selected_parts)
    processed_frames = 0

    os.makedirs(output_folder, exist_ok=True)
    for part_idx in selected_parts:
        start_f, end_f = parts_ranges[part_idx]
        output_path = os.path.join(output_folder, f"part_{part_idx}_x{scale}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_w * scale, new_h * scale))

        try:
            for _ in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret: break
                
                if device_name == "GPU":
                    img = cv2.resize(frame, (new_w, new_h))
                    input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    input_data = input_data.transpose(2, 0, 1)[np.newaxis, ...]
                    res = compiled_model(input_data)[compiled_model.output(0)]
                    output = np.squeeze(res).clip(0, 1).transpose(1, 2, 0)
                    output = cv2.cvtColor((output * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                else:
                    output, _ = upsampler.enhance(frame, outscale=scale)
                
                out.write(output)
                processed_frames += 1
                if progress_callback: progress_callback(int(processed_frames * 100 / total_selected_frames))
        finally:
            out.release()
    cap.release()

class VideoUpscaleWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, input_path, output_folder, num_splits, target_parts, tile, scale):
        super().__init__()
        self.input_path, self.output_folder = input_path, output_folder
        self.num_splits, self.target_parts = num_splits, target_parts
        self.tile, self.scale = tile, scale

    def run(self):
        try:
            self.progress.emit(1)
            run_split_upscale(self.input_path, self.num_splits, self.target_parts, self.scale, self.tile, self.output_folder, self.progress.emit, self.log.emit)
            self.finished.emit(f"✨ 완료: {self.output_folder}")
        except Exception as e:
            self.finished.emit(f"❌ 오류: {str(e)}")

def create_label_with_info(translator, text_key, tooltip_key):
    container = QWidget()
    hl = QHBoxLayout(container)
    hl.setContentsMargins(0, 0, 0, 0)
    label = QLabel(translator.t(text_key))
    info = QPushButton('?')
    info.setToolTip(translator.t(tooltip_key))
    info.setFixedSize(20, 20)
    info.setCursor(Qt.PointingHandCursor)
    hl.addWidget(label)
    hl.addWidget(info)
    if hasattr(translator, 'translations'):
        translator.translations.append((label, 'setText', text_key))
        translator.translations.append((info, 'setToolTip', tooltip_key))
    return container

def create_video_tab(parent, translations):
    page = QWidget()
    layout = QVBoxLayout()

    input_layout = QHBoxLayout()
    input_layout.addWidget(create_label_with_info(parent, 'input_video', 'input_video_tip'))
    parent.vid_input_edit = QLineEdit('')
    input_layout.addWidget(parent.vid_input_edit)
    parent.vid_browse_btn = QPushButton(parent.t('browse'))
    parent.vid_browse_btn.clicked.connect(parent.browse_video_input)
    input_layout.addWidget(parent.vid_browse_btn)
    layout.addLayout(input_layout)

    output_layout = QHBoxLayout()
    output_layout.addWidget(create_label_with_info(parent, 'output_folder', 'output_folder_tip'))
    parent.vid_output_edit = QLineEdit('')
    output_layout.addWidget(parent.vid_output_edit)
    parent.output_browse_btn = QPushButton(parent.t('browse'))
    parent.output_browse_btn.clicked.connect(parent.browse_output_folder)
    output_layout.addWidget(parent.output_browse_btn)
    layout.addLayout(output_layout)

    scale_layout = QHBoxLayout()
    scale_layout.addWidget(create_label_with_info(parent, 'scale', 'scale_tip'))
    parent.vid_scale_combo = QComboBox()
    parent.vid_scale_combo.addItems(['2x', '4x', '8x'])
    scale_layout.addWidget(parent.vid_scale_combo)
    layout.addLayout(scale_layout)

    split_layout = QHBoxLayout()
    split_layout.addWidget(create_label_with_info(parent, 'split_count', 'split_count_tip'))
    parent.split_spin = QSpinBox()
    parent.split_spin.setRange(1, 999)
    parent.split_spin.setValue(10)
    split_layout.addWidget(parent.split_spin)
    layout.addLayout(split_layout)

    target_layout = QHBoxLayout()
    target_layout.addWidget(create_label_with_info(parent, 'target_parts', 'target_parts_tip'))
    parent.target_parts_edit = QLineEdit('0')
    target_layout.addWidget(parent.target_parts_edit)
    layout.addLayout(target_layout)

    tile_layout = QHBoxLayout()
    tile_layout.addWidget(create_label_with_info(parent, 'tile_size', 'tile_size_tip'))
    parent.tile_spin = QSpinBox()
    parent.tile_spin.setRange(0, 4096)
    parent.tile_spin.setValue(800)
    tile_layout.addWidget(parent.tile_spin)
    layout.addLayout(tile_layout)

    parent.vid_device_label = QLabel(get_device_info_text(parent.language))
    layout.addWidget(parent.vid_device_label)

    parent.vid_recommend_label = QLabel(get_device_recommendation(parent.language))
    layout.addWidget(parent.vid_recommend_label)

    parent.vid_progress = QProgressBar()
    layout.addWidget(parent.vid_progress)

    parent.vid_run_btn = QPushButton(parent.t('run_video_upscale'))
    parent.vid_run_btn.clicked.connect(parent.run_video_upscale)
    layout.addWidget(parent.vid_run_btn)

    parent.vid_log = QTextEdit()
    parent.vid_log.setReadOnly(True)
    layout.addWidget(parent.vid_log)

    page.setLayout(layout)
    return page