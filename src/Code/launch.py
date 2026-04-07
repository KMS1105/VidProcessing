import os
import sys
import re

# DLL 경로 설정 (PyTorch 관련)
dll_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)

from PyQt5.QtWidgets import (QApplication, QWidget, QTabWidget, QVBoxLayout, 
                             QMenuBar, QFileDialog, QHBoxLayout, QLabel, 
                             QPushButton, QListWidget, QAbstractItemView, QMessageBox)
from PyQt5.QtCore import Qt

from setting import UI_TEXTS, get_device_info_text, get_device_recommendation, apply_app_theme
from UpscaleImg import create_image_tab, ImageUpscaleWorker
from UpscaleVid import create_video_tab, VideoUpscaleWorker

# 영상 합치기 로직
from moviepy.editor import VideoFileClip, concatenate_videoclips

def merge_videos(video_paths, output_path):
    if not video_paths:
        return
    
    clips = []
    try:
        for path in video_paths:
            clip = VideoFileClip(path)
            # 첫 번째 영상의 해상도에 맞춰 리사이징
            if clips and (clip.size != clips[0].size):
                clip = clip.resize(width=clips[0].size[0], height=clips[0].size[1])
            clips.append(clip)
        
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        for clip in clips:
            clip.close()
            
    except Exception as e:
        raise e

# 영상 합치기 전용 탭 클래스
class VideoMergeTab(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        self.label = QLabel(self.main_app.t('merge_video_list')) # 번역 키 필요
        layout.addWidget(self.label)

        # 리스트 위젯 (드래그로 순서 변경 가능)
        self.video_list = QListWidget()
        self.video_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.video_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.video_list)

        # 버튼 레이아웃
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton(self.main_app.t('add_video'))
        self.btn_remove = QPushButton(self.main_app.t('remove_selected'))
        self.btn_clear = QPushButton(self.main_app.t('clear_all'))
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

        # 실행 버튼
        self.btn_run = QPushButton(self.main_app.t('run_merge'))
        self.btn_run.setProperty("class", "run-button") # 기존 스타일 유지용
        self.btn_run.setFixedHeight(45)
        layout.addWidget(self.btn_run)

        self.setLayout(layout)

        # 연결
        self.btn_add.clicked.connect(self.add_videos)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.video_list.clear)
        self.btn_run.clicked.connect(self.run_video_merge)

    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select Videos', '', 'Videos (*.mp4 *.mov *.avi *.mkv)')
        if files:
            self.video_list.addItems(files)

    def remove_selected(self):
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))

    def run_video_merge(self):
        items = [self.video_list.item(i).text() for i in range(self.video_list.count())]
        if len(items) < 2:
            QMessageBox.warning(self, "Warning", self.main_app.t('error_min_videos'))
            return

        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Merged Video', 'merged.mp4', 'Video (*.mp4)')
        if save_path:
            try:
                merge_videos(items, save_path)
                QMessageBox.information(self, "Success", self.main_app.t('success_merge'))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"{str(e)}")

class UpscaleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.language = 'ko'
        self.theme = 'light'
        self.ui_texts = UI_TEXTS
        self.translations = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Upscaler')
        self.setGeometry(120, 120, 760, 650) # 높이 소폭 조정
        self.menu_bar = QMenuBar()
        main_layout = QVBoxLayout()
        main_layout.setMenuBar(self.menu_bar)
        
        self.tabs = QTabWidget()
        
        # 1. 이미지 업스케일 탭
        self.tabs.addTab(create_image_tab(self, self.translations), self.t('tab_image'))
        # 2. 비디오 업스케일 탭
        self.tabs.addTab(create_video_tab(self, self.translations), self.t('tab_video'))
        # 3. 비디오 합치기 탭 (추가됨)
        self.video_merge_tab = VideoMergeTab(self)
        self.tabs.addTab(self.video_merge_tab, self.t('tab_video_merge')) # 번역 키: tab_video_merge
        
        main_layout.addWidget(self.tabs)
        self.set_menu()
        self.setLayout(main_layout)
        self.set_theme(self.theme)
        self.update_language()

    def t(self, key):
        return self.ui_texts[self.language].get(key, key)

    def set_menu(self):
        self.menu_bar.clear()
        theme_menu = self.menu_bar.addMenu(self.t('menu_theme'))
        theme_menu.addAction(self.t('menu_light'), lambda: self.set_theme('light'))
        theme_menu.addAction(self.t('menu_dark'), lambda: self.set_theme('dark'))
        lang_menu = self.menu_bar.addMenu(self.t('menu_language'))
        lang_menu.addAction(self.t('lang_ko'), lambda: self.set_language('ko'))
        lang_menu.addAction(self.t('lang_en'), lambda: self.set_language('en'))

    def set_theme(self, theme):
        self.theme = theme
        apply_app_theme(self, theme)

    def set_language(self, language):
        self.language = language
        self.update_language()

    def update_language(self):
        self.setWindowTitle(self.t('window_title'))
        self.set_menu()
        self.tabs.setTabText(0, self.t('tab_image'))
        self.tabs.setTabText(1, self.t('tab_video'))
        self.tabs.setTabText(2, self.t('tab_video_merge'))
        
        for widget, method, key in self.translations:
            getattr(widget, method)(self.t(key))
            
        # UI_TEXTS에 새 키가 정의되어 있어야 합니다. (아래 팁 참고)
        self.img_device_label.setText(get_device_info_text(self.language))
        self.vid_device_label.setText(get_device_info_text(self.language))
        self.img_recommend_label.setText(get_device_recommendation(self.language))
        self.vid_recommend_label.setText(get_device_recommendation(self.language))
        
        # 버튼 텍스트 갱신
        self.img_run_btn.setText(self.t('upscale_image'))
        self.vid_run_btn.setText(self.t('run_video_upscale'))
        # 합치기 탭 버튼 갱신
        self.video_merge_tab.label.setText(self.t('merge_video_list'))
        self.video_merge_tab.btn_add.setText(self.t('add_video'))
        self.video_merge_tab.btn_remove.setText(self.t('remove_selected'))
        self.video_merge_tab.btn_clear.setText(self.t('clear_all'))
        self.video_merge_tab.btn_run.setText(self.t('run_merge'))
        
        # 브라우즈 버튼 공통
        for btn in [self.img_browse_btn, self.img_output_browse_btn, 
                    self.vid_browse_btn, self.output_browse_btn]:
            btn.setText(self.t('browse'))

    # ... (기본 이미지/비디오 실행 함수들은 동일) ...
    def browse_image_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Input Image', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if file_path: self.img_input_edit.setText(file_path)

    def browse_video_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Input Video', '', 'Videos (*.mp4 *.mov *.avi *.mkv)')
        if file_path: self.vid_input_edit.setText(file_path)

    def browse_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder_path:
            self.img_output_edit.setText(folder_path)
            self.vid_output_edit.setText(folder_path)

    def run_image_upscale(self):
        input_path = self.img_input_edit.text().strip()
        output_folder = self.img_output_edit.text().strip()
        scale_text = self.img_scale_combo.currentText().replace('x', '')
        scale = int(scale_text)
        if not os.path.exists(input_path):
            self.append_image_log(self.t('error_input_missing').format(input_path))
            return
        if not output_folder: output_folder = os.path.dirname(input_path) or '.'
        self.append_image_log(self.t('start_image'))
        self.img_progress.setValue(5)
        self.image_worker = ImageUpscaleWorker(input_path, output_folder, scale)
        self.image_worker.progress.connect(self.img_progress.setValue)
        self.image_worker.finished.connect(self.on_image_finished)
        self.image_worker.start()

    def run_video_upscale(self):
        input_path = self.vid_input_edit.text().strip()
        output_folder = self.vid_output_edit.text().strip()
        num_splits = self.split_spin.value()
        tile = self.tile_spin.value()
        scale = int(self.vid_scale_combo.currentText().replace('x', ''))
        if not os.path.exists(input_path):
            self.append_video_log(self.t('error_input_missing').format(input_path))
            return
        if not output_folder:
            self.append_video_log(self.t('error_no_output'))
            return
        target_text = self.target_parts_edit.text().strip()
        try:
            target_parts = [int(x) for x in re.split(r'\s*,\s*', target_text) if x != '']
        except ValueError:
            self.append_video_log(self.t('error_target_parts'))
            return
        if not target_parts:
            self.append_video_log(self.t('error_no_target_parts'))
            return
        self.append_video_log(self.t('start_video'))
        self.vid_progress.setValue(5)
        self.video_worker = VideoUpscaleWorker(input_path, output_folder, num_splits, target_parts, tile, scale)
        self.video_worker.progress.connect(self.vid_progress.setValue)
        self.video_worker.log.connect(self.append_video_log)
        self.video_worker.finished.connect(self.on_video_finished)
        self.video_worker.start()

    def append_image_log(self, message): self.img_log.append(message)
    def append_video_log(self, message): self.vid_log.append(message)
    def on_image_finished(self, message):
        self.append_image_log(message)
        self.img_progress.setValue(100)
    def on_video_finished(self, message):
        self.append_video_log(message+"\n")
        self.vid_progress.setValue(100)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UpscaleApp()
    window.show()
    sys.exit(app.exec_())