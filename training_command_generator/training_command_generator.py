#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import configparser
from collections import defaultdict
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                             QPushButton, QFileDialog, QCheckBox, QGroupBox,
                             QTextEdit, QScrollArea, QTabWidget, QMessageBox)
from PyQt5.QtCore import Qt

class TrainingCommandGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
        self.settings = self.load_settings()
        self.initUI()
        self.config_file = None
        self.config_data = defaultdict(dict)
        self.checkboxes = defaultdict(dict)
        
        # 시작 시 default_config.ini 파일 로드 시도
        self.load_default_config()
        # 이후 settings에 저장된 기본 INI 파일 로드 시도
        if not self.config_file:
            self.load_default_ini()
        
    def load_settings(self):
        """설정 파일 로드"""
        settings = {
            'default_ini_path': '',
            'last_script': 'train.py'
        }
        
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    settings.update(loaded_settings)
            except Exception as e:
                print(f"설정 파일 로드 오류: {e}")
        
        return settings
    
    def load_default_config(self):
        """실행 파일과 같은 경로의 default_config.ini 파일 로드 시도"""
        default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_config.ini')
        if os.path.exists(default_config):
            self.file_path_edit.setText(default_config)
            self.load_config()
            
    def save_settings(self):
        """설정 파일 저장"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"설정 파일 저장 오류: {e}")
            
    def load_default_ini(self):
        """settings에 저장된 기본 INI 파일 로드 시도"""
        default_ini = self.settings.get('default_ini_path', '')
        if default_ini and os.path.exists(default_ini):
            self.file_path_edit.setText(default_ini)
            self.load_config()
        
    def initUI(self):
        self.setWindowTitle('머신러닝 학습 명령어 생성기')
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # File selection section
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText('INI 파일 경로')
        self.file_path_edit.setReadOnly(True)
        
        browse_button = QPushButton('찾아보기')
        browse_button.clicked.connect(self.browse_ini_file)
        
        load_button = QPushButton('로드')
        load_button.clicked.connect(self.load_config)
        
        set_default_button = QPushButton('기본 경로로 설정')
        set_default_button.clicked.connect(self.set_default_ini_path)
        
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_button)
        file_layout.addWidget(load_button)
        file_layout.addWidget(set_default_button)
        
        main_layout.addLayout(file_layout)
        
        # Script file input
        script_layout = QHBoxLayout()
        script_label = QLabel('학습 스크립트:')
        self.script_edit = QLineEdit()
        self.script_edit.setText(self.settings.get('last_script', 'train.py'))  # 기본값 설정
        self.script_edit.setPlaceholderText('학습 파이썬 파일 경로 (예: train.py)')
        
        script_browse_button = QPushButton('찾아보기')
        script_browse_button.clicked.connect(self.browse_script_file)
        
        script_layout.addWidget(script_label)
        script_layout.addWidget(self.script_edit)
        script_layout.addWidget(script_browse_button)
        
        main_layout.addLayout(script_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Section for displaying config options
        self.config_tab = QWidget()
        self.system_tab = QWidget()
        
        self.tabs.addTab(self.config_tab, '학습 설정')
        self.tabs.addTab(self.system_tab, '시스템 설정')
        
        # Setup layouts for tabs
        self.config_layout = QVBoxLayout()
        self.system_layout = QVBoxLayout()
        
        self.config_tab.setLayout(self.config_layout)
        self.system_tab.setLayout(self.system_layout)
        
        # Scroll areas for each tab
        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_layout.addWidget(self.config_scroll)
        
        self.system_scroll = QScrollArea()
        self.system_scroll.setWidgetResizable(True)
        self.system_layout.addWidget(self.system_scroll)
        
        # Content widgets for scroll areas
        self.config_content = QWidget()
        self.config_scroll.setWidget(self.config_content)
        self.config_content_layout = QVBoxLayout(self.config_content)
        
        self.system_content = QWidget()
        self.system_scroll.setWidget(self.system_content)
        self.system_content_layout = QVBoxLayout(self.system_content)
        
        # Command output section
        command_group = QGroupBox('생성된 명령어')
        command_layout = QVBoxLayout()
        
        self.command_output = QTextEdit()
        self.command_output.setReadOnly(True)
        command_layout.addWidget(self.command_output)
        
        generate_button = QPushButton('명령어 생성')
        generate_button.clicked.connect(self.generate_command)
        command_layout.addWidget(generate_button)
        
        run_button = QPushButton('명령어 실행')
        run_button.clicked.connect(self.run_command)
        command_layout.addWidget(run_button)
        
        command_group.setLayout(command_layout)
        main_layout.addWidget(command_group)
        
    def set_default_ini_path(self):
        """현재 선택된 INI 파일을 기본 경로로 설정"""
        current_path = self.file_path_edit.text()
        if not current_path or not os.path.exists(current_path):
            QMessageBox.warning(self, '오류', '먼저 INI 파일을 선택해주세요.')
            return
            
        self.settings['default_ini_path'] = current_path
        self.save_settings()
        QMessageBox.information(self, '성공', f'기본 INI 파일 경로가 설정되었습니다:\n{current_path}')
        
    def browse_ini_file(self):
        # 기본 디렉토리를 현재 설정된 INI 파일 디렉토리로 설정
        default_dir = ''
        if self.settings.get('default_ini_path'):
            default_dir = os.path.dirname(self.settings['default_ini_path'])
            
        file_path, _ = QFileDialog.getOpenFileName(self, '설정 파일 선택', default_dir, 'INI Files (*.ini)')
        if file_path:
            self.file_path_edit.setText(file_path)
    
    def browse_script_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '학습 스크립트 선택', '', 'Python Files (*.py)')
        if file_path:
            # 파일 이름만 추출
            script_name = os.path.basename(file_path)
            self.script_edit.setText(script_name)
            # 스크립트 이름 설정에 저장
            self.settings['last_script'] = script_name
            self.save_settings()
    
    def load_config(self):
        config_path = self.file_path_edit.text()
        if not config_path or not os.path.exists(config_path):
            QMessageBox.warning(self, '오류', 'INI 파일을 선택해주세요.')
            return
        
        self.config_file = config_path
        self.config_data.clear()
        self.checkboxes.clear()
        
        # Clear the content layouts
        self._clear_layout(self.config_content_layout)
        self._clear_layout(self.system_content_layout)
        
        # Parse config file
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        
        # Process sections and options
        for section in config.sections():
            is_system = section.lower().startswith('system')
            
            # Choose appropriate layout based on section type
            layout = self.system_content_layout if is_system else self.config_content_layout
            
            # Create a group box for each section
            group_box = QGroupBox(section)
            group_layout = QVBoxLayout()
            
            for option in config[section]:
                value = config[section][option]
                option_values = [val.strip() for val in value.split(',')]
                
                if option_values:
                    option_layout = QVBoxLayout()
                    option_label = QLabel(f"{option}:")
                    option_layout.addWidget(option_label)
                    
                    checkbox_layout = QHBoxLayout()
                    checkbox_group = []
                    
                    for val in option_values:
                        checkbox = QCheckBox(val)
                        checkbox.setChecked(True)  # Default all to checked
                        checkbox_layout.addWidget(checkbox)
                        checkbox_group.append(checkbox)
                    
                    option_layout.addLayout(checkbox_layout)
                    group_layout.addLayout(option_layout)
                    
                    # Store checkboxes for later reference
                    self.checkboxes[section][option] = checkbox_group
                    
                    # Store config data
                    self.config_data[section][option] = option_values
            
            group_box.setLayout(group_layout)
            layout.addWidget(group_box)
        
        # Message if successful
        QMessageBox.information(self, '성공', '설정 파일을 성공적으로 로드했습니다.')
    
    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self._clear_layout(item.layout())
    
    def generate_command(self):
        if not self.config_data:
            QMessageBox.warning(self, '오류', '먼저 설정 파일을 로드해주세요.')
            return
        
        # 사용자가 지정한 학습 스크립트 파일 사용
        script_name = self.script_edit.text().strip()
        if not script_name:
            script_name = "train.py"  # 비어 있으면 기본값 사용
        
        # 스크립트 이름 업데이트
        self.settings['last_script'] = script_name
        self.save_settings()
        
        command = script_name
        
        # Add selected options from config sections
        for section in self.checkboxes:
            for option in self.checkboxes[section]:
                selected_values = []
                
                for i, checkbox in enumerate(self.checkboxes[section][option]):
                    if checkbox.isChecked():
                        selected_values.append(self.config_data[section][option][i])
                
                if selected_values:
                    command += f" -{option}"
                    for val in selected_values:
                        command += f" {val}"
        
        self.command_output.setText(command)
    
    def run_command(self):
        command = self.command_output.toPlainText()
        if not command:
            QMessageBox.warning(self, '오류', '먼저 명령어를 생성해주세요.')
            return
        
        try:
            # 운영체제 별로 다른 방식으로 새 터미널에서 명령어 실행
            platform = sys.platform.lower()
            
            if platform.startswith('win'):  # Windows
                # Windows에서는 cmd 창에서 명령어 실행
                # /k는 명령 실행 후 창을 유지함
                terminal_command = f'start cmd /k "{command}"'
                subprocess.Popen(terminal_command, shell=True)
                message = '새 명령 프롬프트 창에서 명령어가 실행되었습니다.'
                
            elif platform.startswith('linux'):  # Linux
                # 일반적인 Linux 터미널 에뮬레이터 시도
                terminal_emulators = [
                    ['gnome-terminal', '--', 'bash', '-c', f'{command}; exec bash'],
                    ['konsole', '--', 'bash', '-c', f'{command}; exec bash'],
                    ['xterm', '-e', f'bash -c "{command}; exec bash"'],
                    ['x-terminal-emulator', '-e', f'bash -c "{command}; exec bash"']
                ]
                
                success = False
                for emulator in terminal_emulators:
                    try:
                        subprocess.Popen(emulator)
                        success = True
                        break
                    except FileNotFoundError:
                        continue
                    
                if success:
                    message = '새 터미널 창에서 명령어가 실행되었습니다.'
                else:
                    # 모든 터미널 에뮬레이터 시도 실패 시 기존 방식으로 실행
                    result = subprocess.run(command, shell=True, check=True, 
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True)
                    message = f'명령어가 백그라운드에서 실행되었습니다. (터미널 에뮬레이터를 찾을 수 없음)\n출력:\n{result.stdout}'
                
            elif platform.startswith('darwin'):  # macOS
                # macOS에서는 새 Terminal 창에서 명령어 실행
                terminal_command = ['osascript', '-e', f'tell app "Terminal" to do script "{command}"']
                subprocess.Popen(terminal_command)
                message = '새 Terminal 창에서 명령어가 실행되었습니다.'
                
            else:  # 지원되지 않는 OS
                # 기본 방식으로 실행
                result = subprocess.run(command, shell=True, check=True, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
                message = f'명령어가 성공적으로 실행되었습니다. (별도의 터미널을 지원하지 않는 OS)\n출력:\n{result.stdout}'
            
            QMessageBox.information(self, '성공', message)
            
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, '오류', f'명령어 실행 중 오류가 발생했습니다.\n{e.stderr}')
        except Exception as e:
            QMessageBox.critical(self, '오류', f'명령어 실행 중 예상치 못한 오류가 발생했습니다.\n{str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    generator = TrainingCommandGenerator()
    generator.show()
    sys.exit(app.exec_()) 