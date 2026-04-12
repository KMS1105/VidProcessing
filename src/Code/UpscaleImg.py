import sys
import os
import cv2
import torch
import numpy as np
import openvino as ov
import urllib.request
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QProgressBar, QTextEdit, QVBoxLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from setting import get_device_info_text, get_device_recommendation

try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

class ImageUpscaleWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, input_path, output_folder, scale):
        super().__init__()
        self.input_path = input_path
        self.output_folder = output_folder
        self.scale = scale

    def run(self):
        try:
            self.progress.emit(5)
            core = ov.Core()
            available_devices = core.available_devices

            model_filenames = {2: 'RealESRGAN_x2plus', 4: 'RealESRGAN_x4plus', 8: 'RealESRGAN_x8'}
            model_urls = {
                2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                8: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x8.pth'
            }

            model_base = model_filenames.get(self.scale, 'RealESRGAN_x2plus')
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_dir = os.path.join(base_dir, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            pth_path = os.path.join(weights_dir, f"{model_base}.pth")
            xml_path = os.path.join(weights_dir, f"{model_base}.xml")

            if not os.path.exists(pth_path):
                self.log.emit(f"⏳ 모델 가중치 다운로드 중: {model_base}.pth")
                urllib.request.urlretrieve(model_urls[self.scale], pth_path)
                self.log.emit("✅ 다운로드 완료")
            
            self.progress.emit(20)

            if torch.cuda.is_available():
                device_name = "CUDA"
            elif any("GPU" in d for d in available_devices):
                device_name = "GPU"
            else:
                device_name = "CPU"

            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            if device_name == "GPU":
                gpu_id = "GPU.1" if "GPU.1" in available_devices else "GPU.0"
                if not os.path.exists(xml_path):
                    self.log.emit(f"🔄 Intel GPU 최적화 변환 중: {model_base}.xml")
                    model_temp = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)
                    loadnet = torch.load(pth_path, map_location='cpu')
                    model_temp.load_state_dict(loadnet['params_ema'] if 'params_ema' in loadnet else loadnet, strict=True)
                    model_temp.eval()
                    dummy = torch.randn(1, 3, 256, 256)
                    ov_model = ov.convert_model(model_temp, example_input=dummy)
                    ov.save_model(ov_model, xml_path)
                    self.log.emit("✅ 모델 변환 완료")
                
                img = cv2.imread(self.input_path)
                h, w = img.shape[:2]
                new_h, new_w = (h // 4) * 4, (w // 4) * 4
                img = cv2.resize(img, (new_w, new_h))
                
                ov_model_loaded = core.read_model(xml_path)
                ov_model_loaded.reshape([1, 3, new_h, new_w])
                compiled_model = core.compile_model(ov_model_loaded, gpu_id)
                
                input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_data = input_data.transpose(2, 0, 1)[np.newaxis, ...]
                res = compiled_model(input_data)[compiled_model.output(0)]
                output = np.squeeze(res).clip(0, 1).transpose(1, 2, 0)
                output = cv2.cvtColor((output * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)
                upsampler = RealESRGANer(scale=self.scale, model_path=pth_path, model=model_arch, tile=400, half=(device_name=="CUDA"), device='cuda' if device_name=="CUDA" else 'cpu')
                img = cv2.imread(self.input_path)
                output, _ = upsampler.enhance(img, outscale=self.scale)

            self.progress.emit(80)
            
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            
            base_name = os.path.splitext(os.path.basename(self.input_path))[0]
            save_path = os.path.join(self.output_folder, f"{base_name}_x{self.scale}.png")
            cv2.imwrite(save_path, output)
            
            self.progress.emit(100)
            self.finished.emit(f"✨ 완료: {save_path}")
        except Exception as e:
            self.finished.emit(f"❌ 오류: {str(e)}")

def create_label_with_info(parent, text_key, tip_key):
    layout = QHBoxLayout()
    label = QLabel(parent.t(text_key))
    info_btn = QPushButton("?")
    info_btn.setFixedSize(20, 20)
    info_btn.setToolTip(parent.t(tip_key))
    info_btn.setStyleSheet("QPushButton { border-radius: 10px; background-color: #e0e0e0; font-weight: bold; }")
    layout.addWidget(label)
    layout.addWidget(info_btn)
    layout.addStretch()
    container = QWidget()
    container.setLayout(layout)
    return container

def create_image_tab(parent, translations):
    layout = QVBoxLayout()
    
    input_layout = QHBoxLayout()
    input_layout.addWidget(create_label_with_info(parent, 'input_image', 'input_image_tip'))
    parent.img_input_edit = QLineEdit('')
    input_layout.addWidget(parent.img_input_edit)
    parent.img_browse_btn = QPushButton(parent.t('browse'))
    parent.img_browse_btn.clicked.connect(parent.browse_image_input)
    input_layout.addWidget(parent.img_browse_btn)
    layout.addLayout(input_layout)

    output_layout = QHBoxLayout()
    output_layout.addWidget(create_label_with_info(parent, 'output_folder', 'output_folder_tip'))
    parent.img_output_edit = QLineEdit('')
    output_layout.addWidget(parent.img_output_edit)
    parent.img_output_browse_btn = QPushButton(parent.t('browse'))
    parent.img_output_browse_btn.clicked.connect(parent.browse_output_folder)
    output_layout.addWidget(parent.img_output_browse_btn)
    layout.addLayout(output_layout)

    scale_layout = QHBoxLayout()
    scale_layout.addWidget(create_label_with_info(parent, 'scale', 'scale_tip'))
    parent.img_scale_combo = QComboBox()
    parent.img_scale_combo.addItems(['2x', '4x', '8x'])
    scale_layout.addWidget(parent.img_scale_combo)
    layout.addLayout(scale_layout)

    parent.img_recommend_label = QLabel(get_device_recommendation(parent.language))
    layout.addWidget(parent.img_recommend_label)

    parent.img_progress = QProgressBar()
    layout.addWidget(parent.img_progress)

    parent.img_log = QTextEdit()
    parent.img_log.setReadOnly(True)
    layout.addWidget(parent.img_log)

    parent.img_run_btn = QPushButton(parent.t('upscale_image'))
    parent.img_run_btn.setFixedHeight(40)
    parent.img_run_btn.clicked.connect(parent.run_image_upscale)
    layout.addWidget(parent.img_run_btn)

    tab = QWidget()
    tab.setLayout(layout)
    return tab