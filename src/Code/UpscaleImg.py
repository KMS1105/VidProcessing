import sys
import os
import cv2
import torch
import numpy as np
import openvino as ov
import urllib.request
import glob
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QProgressBar, QTextEdit, QVBoxLayout, QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from setting import get_device_info_text, get_device_recommendation

try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

class ModelSetupWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, weights_dir):
        super().__init__()
        self.weights_dir = weights_dir

    def run(self):
        required_models = {
            "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        }

        use_cuda = torch.cuda.is_available()
        from basicsr.archs.rrdbnet_arch import RRDBNet

        for name, url in required_models.items():
            pth_path = os.path.join(self.weights_dir, name)
            xml_path = pth_path.replace('.pth', '.xml')
            
            if not os.path.exists(pth_path):
                self.log.emit(f"⏳ 모델 다운로드 중: {name}")
                try: 
                    urllib.request.urlretrieve(url, pth_path)
                    self.log.emit(f"✅ 다운로드 완료: {name}")
                except Exception as e: 
                    self.log.emit(f"❌ 다운로드 실패: {str(e)}")
            
            if not use_cuda and os.path.exists(pth_path) and not os.path.exists(xml_path):
                self.log.emit(f"🔄 Intel GPU 최적화 변환 중: {name}")
                try:
                    scale = 2 if 'x2' in name.lower() else 8 if 'x8' in name.lower() else 4
                    model_temp = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
                    loadnet = torch.load(pth_path, map_location='cpu')
                    model_temp.load_state_dict(loadnet['params_ema'] if 'params_ema' in loadnet else loadnet, strict=True)
                    model_temp.eval()
                    ov_model = ov.convert_model(model_temp, example_input=torch.randn(1, 3, 256, 256))
                    ov.save_model(ov_model, xml_path)
                    self.log.emit(f"✅ 변환 완료: {os.path.basename(xml_path)}")
                except Exception as e: 
                    self.log.emit(f"❌ 변환 실패: {str(e)}")
        
        self.finished.emit()

class ImageUpscaleWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, input_path, output_folder, model_path):
        super().__init__()
        self.input_path = input_path
        self.output_folder = output_folder
        self.model_path = model_path

    def run(self):
        try:
            self.progress.emit(10)
            core = ov.Core()
            available_devices = core.available_devices
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            scale = 2 if 'x2' in model_name.lower() else 8 if 'x8' in model_name.lower() else 4

            img_array = np.fromfile(self.input_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None: raise Exception("이미지를 읽을 수 없습니다.")

            use_cuda = torch.cuda.is_available()
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            if not use_cuda and (self.model_path.endswith('.xml') or self.model_path.endswith('.onnx')):
                target_device = "GPU.1" if "GPU.1" in available_devices else "GPU.0" if "GPU.0" in available_devices else "CPU"
                h, w = img.shape[:2]
                new_h, new_w = (h // 4) * 4, (w // 4) * 4
                img_input = cv2.resize(img, (new_w, new_h))
                ov_model = core.read_model(self.model_path)
                ov_model.reshape([1, 3, new_h, new_w])
                compiled_model = core.compile_model(ov_model, target_device)
                input_data = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_data = input_data.transpose(2, 0, 1)[np.newaxis, ...]
                res = compiled_model(input_data)[compiled_model.output(0)]
                output = np.squeeze(res).clip(0, 1).transpose(1, 2, 0)
                output = cv2.cvtColor((output * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
                upsampler = RealESRGANer(scale=scale, model_path=self.model_path, model=model_arch, tile=400, half=use_cuda, device='cuda' if use_cuda else 'cpu')
                output, _ = upsampler.enhance(img, outscale=scale)

            if not os.path.exists(self.output_folder): os.makedirs(self.output_folder)
            save_path = os.path.join(self.output_folder, f"up_{model_name}_{os.path.basename(self.input_path)}")
            res, en_img = cv2.imencode(os.path.splitext(save_path)[1], output)
            if res: en_img.tofile(save_path)
            self.progress.emit(100)
            self.finished.emit(f"✨ 업스케일 완료")
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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

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

    model_sel_layout = QHBoxLayout()
    model_sel_layout.addWidget(QLabel("모델 선택:"))
    parent.img_model_combo = QComboBox()
    
    def refresh_models():
        parent.img_model_combo.clear()
        use_cuda = torch.cuda.is_available()
        patterns = ["*.pth"] if use_cuda else ["*.onnx", "*.xml"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(weights_dir, "**", p), recursive=True))
        for f in files:
            parent.img_model_combo.addItem(os.path.relpath(f, weights_dir), f)
        parent.img_log.append("🔄 모델 목록이 갱신되었습니다.")
            
    model_sel_layout.addWidget(parent.img_model_combo)
    btn_refresh = QPushButton("🔄")
    btn_refresh.setFixedSize(30, 30)
    btn_refresh.clicked.connect(refresh_models)
    model_sel_layout.addWidget(btn_refresh)
    layout.addLayout(model_sel_layout)

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

    parent.setup_worker = ModelSetupWorker(weights_dir)
    parent.setup_worker.log.connect(parent.img_log.append)
    parent.setup_worker.finished.connect(refresh_models)
    
    QTimer.singleShot(500, parent.setup_worker.start)

    tab = QWidget()
    tab.setLayout(layout)
    return tab