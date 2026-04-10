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
            model_url = model_urls.get(self.scale, model_urls[2])
            weights_dir = os.path.join(os.getcwd(), 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            pth_path = os.path.join(weights_dir, f"{model_base}.pth")
            xml_path = os.path.join(weights_dir, f"{model_base}.xml")

            if not os.path.exists(pth_path):
                urllib.request.urlretrieve(model_url, pth_path)

            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            # 1. NVIDIA CUDA 체크
            if torch.cuda.is_available():
                device_display_name = "NVIDIA CUDA"
                model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)
                upsampler = RealESRGANer(scale=self.scale, model_path=pth_path, model=model_arch, tile=0, half=True, device='cuda')
                img = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
                output, _ = upsampler.enhance(img, outscale=self.scale)

            # 2. Intel GPU (OpenVINO) 체크
            elif any("GPU" in d for d in available_devices):
                gpu_device = "GPU.1" if "GPU.1" in available_devices else "GPU.0"
                device_display_name = f"Intel {gpu_device} (OpenVINO)"

                img = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
                if img is None: return
                
                # 오류 방지: 이미지 크기를 짝수로 패딩 (중요!)
                h, w = img.shape[:2]
                new_h = (h // 4) * 4
                new_w = (w // 4) * 4
                if h != new_h or w != new_w:
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                if not os.path.exists(xml_path):
                    model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)
                    loadnet = torch.load(pth_path, map_location='cpu')
                    model_arch.load_state_dict(loadnet['params_ema'] if 'params_ema' in loadnet else loadnet, strict=True)
                    model_arch.eval()
                    # 모델을 동적 입력이 가능하도록 변환
                    dummy_input = torch.randn(1, 3, new_h, new_w)
                    ov_model = ov.convert_model(model_arch, example_input=dummy_input)
                    ov.save_model(ov_model, xml_path)

                # 모델 로드 및 입력 크기 재조정 (이미지 크기에 맞춤)
                model = core.read_model(xml_path)
                model.reshape([1, 3, new_h, new_w]) 
                compiled_model = core.compile_model(model, gpu_device)
                
                input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...]
                
                result = compiled_model(input_img)[compiled_model.output(0)]
                output = np.squeeze(result).clip(0, 1).transpose(1, 2, 0)
                output = (output * 255.0).astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            # 3. CPU
            else:
                device_display_name = "CPU"
                model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)
                upsampler = RealESRGANer(scale=self.scale, model_path=pth_path, model=model_arch, tile=0, device='cpu')
                img = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
                output, _ = upsampler.enhance(img, outscale=self.scale)

            self.progress.emit(80)
            os.makedirs(self.output_folder, exist_ok=True)
            input_basename = os.path.splitext(os.path.basename(self.input_path))[0]
            output_file = os.path.join(self.output_folder, f"{input_basename}_x{self.scale}.png")
            cv2.imwrite(output_file, output)
            
            self.progress.emit(100)
            self.finished.emit(f"✨ 완료 ({device_display_name}): {os.path.abspath(output_file)}")

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

def create_image_tab(parent, translations):
    page = QWidget()
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

    parent.img_device_label = QLabel(get_device_info_text(parent.language))
    layout.addWidget(parent.img_device_label)

    parent.img_recommend_label = QLabel(get_device_recommendation(parent.language))
    layout.addWidget(parent.img_recommend_label)

    parent.img_progress = QProgressBar()
    layout.addWidget(parent.img_progress)

    parent.img_run_btn = QPushButton(parent.t('upscale_image'))
    parent.img_run_btn.clicked.connect(parent.run_image_upscale)
    layout.addWidget(parent.img_run_btn)

    parent.img_log = QTextEdit()
    parent.img_log.setReadOnly(True)
    layout.addWidget(parent.img_log)

    page.setLayout(layout)
    return page