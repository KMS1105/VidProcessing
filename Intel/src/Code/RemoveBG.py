import os
import cv2
import openvino as ov
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QProgressBar, QTextEdit,
    QFileDialog, QColorDialog
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor
from setting import prepare_bg_model, DragLineEdit
from rembg import remove, new_session

rvm_weights_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
u2net_dir = os.path.join(rvm_weights_dir, 'RemBG')
os.environ["HF_HOME"] = rvm_weights_dir
os.environ["U2NET_HOME"] = u2net_dir

def is_scene_cut(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([prev_gray], [0], None, [64], [0,256])
    hist2 = cv2.calcHist([curr_gray], [0], None, [64], [0,256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    diff = np.sum(np.abs(hist1 - hist2))

    return diff > 0.4   

class RemoveBGWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_path, output_path, chromakey_rgb):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        # PyQt QColor 값을 BGR 튜플 형태로 변환
        self.chromakey_rgb = chromakey_rgb

    def run(self):
        cap = None
        out = None
        bg_color = (self.chromakey_rgb.blue(), self.chromakey_rgb.green(), self.chromakey_rgb.red())

        try:
            self.log.emit("model_loading")
            core = ov.Core()

            rvm_model_path = os.path.join(rvm_weights_dir, "robust-video-matting-mobilenetv3")
            model_paths = prepare_bg_model(lambda m: self.log.emit(m))
            
            def load(path):
                try: return core.compile_model(path, "GPU.1")
                except:
                    try: return core.compile_model(path, "GPU.0")
                    except: return core.compile_model(path, "CPU")

            rvm_model = load(rvm_model_path) if os.path.exists(rvm_model_path) else \
                        (load(model_paths["modnet"]) if model_paths and "modnet" in model_paths else None)
            
            cap = cv2.VideoCapture(self.input_path)
            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(5)
            total = int(cap.get(7))

            output_file = os.path.join(self.output_path, "result.mp4")
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            try:
                os.environ["OMP_NUM_THREADS"] = "4"
                rembg_session = new_session("u2net", providers=["CPUExecutionProvider"])
            except:
                rembg_session = new_session()

            prev_alpha = None
            prev_frame = None
            i = 0

            while i < total:
                ret, frame = cap.read()
                if not ret:
                    break

                scene_cut = False
                if prev_frame is not None:
                    scene_cut = is_scene_cut(prev_frame, frame)

                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(2.0, (8,8))
                l = clahe.apply(l)
                frame_pre = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

                gray_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                lift = np.clip((0.5 - gray_pre) * 2.0, 0, 1)
                frame_pre = frame_pre.astype(np.float32)
                frame_pre = frame_pre * (1 + 0.2 * lift[..., None])
                frame_pre = np.clip(frame_pre, 0, 255).astype(np.uint8)

                gamma = 1.15
                frame_norm = frame_pre.astype(np.float32) / 255.0
                frame_norm = np.power(frame_norm, gamma)
                frame_norm = (frame_norm * 255).astype(np.uint8)

                hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:,:,1] *= 1.1
                hsv[:,:,2] *= 1.05
                hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
                hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)

                frame_input = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                img_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)

                main_alpha = None
                if rvm_model:
                    inp = cv2.resize(img_rgb, (512, 512)).astype(np.float32) / 255.0
                    inp = inp.transpose(2, 0, 1)[None]
                    res = list(rvm_model([inp]).values())
                    main_alpha = res[1].squeeze() if len(res) > 1 else res[0].squeeze()
                    main_alpha = cv2.resize(main_alpha, (w, h))

                res_rembg = remove(img_rgb, session=rembg_session)
                rembg_alpha = cv2.resize(res_rembg[:, :, 3].astype(np.float32) / 255.0, (w, h))

                if main_alpha is not None:
                    conf = cv2.GaussianBlur(main_alpha, (11, 11), 0)
                    alpha = conf * main_alpha + (1 - conf) * rembg_alpha
                else:
                    alpha = rembg_alpha

                alpha = np.clip(alpha, 0, 1)
                hard_fail = False

                if prev_alpha is not None and not scene_cut:
                    if np.mean(alpha) < 0.12:
                        hard_fail = True

                    h_part = int(h * 0.4)
                    prev_top = prev_alpha[:h_part]
                    curr_top = alpha[:h_part]

                    prev_fg_top = np.mean(prev_top > 0.5)
                    curr_fg_top = np.mean(curr_top > 0.5)

                    if prev_fg_top > 0.05 and curr_fg_top < prev_fg_top * 0.3:
                        hard_fail = True

                if hard_fail and prev_alpha is not None:
                    alpha = prev_alpha.copy()
                    
                if prev_alpha is not None and prev_frame is not None and not scene_cut:
                    tile = 32
                    alpha_new = alpha.copy()

                    for y in range(0, h, tile):
                        for x in range(0, w, tile):
                            y2 = min(y + tile, h)
                            x2 = min(x + tile, w)

                            curr_patch = frame[y:y2, x:x2]
                            prev_patch = prev_frame[y:y2, x:x2]

                            diff = np.mean(np.abs(curr_patch.astype(np.float32) - prev_patch.astype(np.float32))) / 255.0

                            if diff < 0.08:
                                prev_a = prev_alpha[y:y2, x:x2]
                                curr_a = alpha[y:y2, x:x2]

                                restore_mask = (curr_a < 0.3).astype(np.float32)

                                alpha_new[y:y2, x:x2] = np.maximum(
                                    curr_a,
                                    0.7 * prev_a * restore_mask
                                )

                    alpha = alpha_new
                    
                alpha_u8 = (alpha * 255).astype(np.uint8)
                kernel = np.ones((7,7), np.uint8)
                alpha_fill = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
                alpha = np.maximum(alpha, alpha_fill.astype(np.float32)/255.0)

                alpha_bin = (alpha > 0.6).astype(np.float32)
                strong_fg = (alpha > 0.8).astype(np.float32)
                alpha_bin = np.maximum(alpha_bin, strong_fg)

                alpha_bin = cv2.medianBlur((alpha_bin * 255).astype(np.uint8), 3) / 255.0
                alpha = alpha_bin

                bg_frame = np.full_like(frame, bg_color)
                fg_part = (frame * alpha[..., None]).astype(np.float32)
                bg_part = (bg_frame * (1.0 - alpha[..., None])).astype(np.float32)
                result = (fg_part + bg_part).astype(np.uint8)

                out.write(result)

                prev_alpha = alpha
                prev_frame = frame.copy()
                i += 1

                if i % 10 == 0 or i == total:
                    self.progress.emit(int(i * 100 / total))

            self.log.emit("done")

        except Exception as e:
            self.log.emit(f"error {str(e)}")

        finally:
            if cap: cap.release()
            if out: out.release()
            self.finished.emit()

class RemoveBGTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        self.input_label = QLabel(self.parent.t('input_video'))
        self.input_edit = DragLineEdit(self)
        self.input_edit.dropped.connect(self.update_default_output)
        self.browse_btn = QPushButton(self.parent.t('browse'))
        self.browse_btn.clicked.connect(self.select_input)
        row1.addWidget(self.input_label)
        row1.addWidget(self.input_edit)
        row1.addWidget(self.browse_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.output_label = QLabel(self.parent.t('output_folder'))
        self.output_edit = QLineEdit()
        self.browse_out_btn = QPushButton(self.parent.t('browse'))
        self.browse_out_btn.clicked.connect(lambda: self.output_edit.setText(QFileDialog.getExistingDirectory()))
        row2.addWidget(self.output_label)
        row2.addWidget(self.output_edit)
        row2.addWidget(self.browse_out_btn)
        layout.addLayout(row2)

        # [수정] 크로마키 색상 선택 버튼 및 라벨 레이아웃 (출력폴더 아래, 프로그레스 바 위)
        chroma_row = QHBoxLayout()
        self.chroma_label = QLabel("Chroma Key:")
        self.chroma_btn = QPushButton("Select Color")
        self.chroma_btn.setStyleSheet("background-color: #000000; color: #ffffff;")
        self.chroma_btn.clicked.connect(self.select_color)
        
        # 기본 색상 설정 (검은색 #000000)
        self.selected_qcolor = QColor(0, 0, 0)
        
        chroma_row.addWidget(self.chroma_label)
        chroma_row.addWidget(self.chroma_btn)
        layout.addLayout(chroma_row)

        self.prog = QProgressBar()
        layout.addWidget(self.prog)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.run_btn = QPushButton(self.parent.t('rbg_start_btn'))
        self.run_btn.clicked.connect(self.start_task)
        layout.addWidget(self.run_btn)

    def select_color(self):
        # 색상 팔레트(QColorDialog) 열기
        color = QColorDialog.getColor(self.selected_qcolor, self, "Select Chromakey Color")
        if color.isValid():
            self.selected_qcolor = color
            hex_val = self.selected_qcolor.name()
            # 텍스트 및 배경색상 업데이트
            self.chroma_btn.setStyleSheet(f"background-color: {hex_val}; color: #ffffff;")

    def select_input(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.input_edit.setText(path)
            self.update_default_output(path)

    def update_default_output(self, file_path):
        self.output_edit.setText(os.path.dirname(file_path))

    def update_ui_texts(self):
        self.run_btn.setText(self.parent.t('rbg_start_btn'))
        self.browse_btn.setText(self.parent.t('browse'))
        self.browse_out_btn.setText(self.parent.t('browse'))
        self.input_label.setText(self.parent.t('input_video'))
        self.output_label.setText(self.parent.t('output_folder'))

    def start_task(self):
        if not self.output_edit.text():
            return

        self.run_btn.setEnabled(False)
        self.prog.setValue(0)
        self.log.clear()

        self.worker = RemoveBGWorker(
            self.input_edit.text(), 
            self.output_edit.text(), 
            chromakey_rgb=self.selected_qcolor
        )
        self.worker.progress.connect(self.prog.setValue)
        self.worker.log.connect(self.log.append)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()