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
                             QTextEdit, QScrollArea, QTabWidget, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QComboBox, QDialog, QDialogButtonBox, QInputDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pandas as pd
import datetime
import platform
import csv

# 구성 CSV 관리자 모듈 임포트
from config_csv_manager import ConfigCSVManager
from command_history_manager import CommandHistoryManager

class TrainingCommandGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
        self.settings = self.load_settings()
        self.config_file = None
        self.config_data = defaultdict(dict)
        self.checkboxes = defaultdict(dict)
        self.custom_configs = []  # 사용자 정의 설정 값 저장
        self.pre_commands = []  # 사전 실행 명령어 저장
        self.env_variables = []  # 환경 변수 저장
        
        # CSV 관리자 초기화 (UI 초기화 전에 먼저 실행)
        self.csv_manager = ConfigCSVManager()
        self.command_manager = CommandHistoryManager()  # 명령어 히스토리 관리자 추가
        
        # UI 초기화
        self.initUI()
        
        # 시작 시 default_config.ini 파일 로드 시도
        self.load_default_config()
        # 이후 settings에 저장된 기본 INI 파일 로드 시도
        if not self.config_file:
            self.load_default_ini()
        
    def load_settings(self):
        """설정 파일 로드"""
        settings = {
            'default_ini_path': '',
            'last_script': 'train.py',
            'custom_configs': [],  # 사용자 정의 설정 저장
            'pre_commands': [],  # 사전 실행 명령어 저장
            'env_variables': []  # 환경 변수 저장
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
        
        # 상태 표시줄 초기화
        self.statusBar = self.statusBar()
        self.statusBar.showMessage('프로그램이 시작되었습니다.')
        
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
        
        # 구성 관리 레이아웃
        config_management_layout = QHBoxLayout()
        config_label = QLabel('구성 관리:')
        
        self.config_combo = QComboBox()
        self.config_combo.setMinimumWidth(200)
        self.update_config_combo()
        
        capture_config_button = QPushButton('현재 구성 캡처')
        capture_config_button.clicked.connect(self.capture_current_config)
        
        load_config_button = QPushButton('구성 불러오기')
        load_config_button.clicked.connect(self.load_selected_config)
        
        manage_config_button = QPushButton('구성 관리')
        manage_config_button.clicked.connect(self.show_config_manager)
        
        # 구성 파일 위치 버튼 추가
        show_files_button = QPushButton('구성 파일 위치')
        show_files_button.clicked.connect(self.show_config_file_locations)
        
        config_management_layout.addWidget(config_label)
        config_management_layout.addWidget(self.config_combo)
        config_management_layout.addWidget(capture_config_button)
        config_management_layout.addWidget(load_config_button)
        config_management_layout.addWidget(manage_config_button)
        config_management_layout.addWidget(show_files_button)
        
        main_layout.addLayout(config_management_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Section for displaying config options
        self.config_tab = QWidget()
        self.custom_tab = QWidget()  # 사용자 정의 설정
        self.pre_commands_tab = QWidget()  # 사전 실행 명령어 탭
        self.env_variables_tab = QWidget()  # 환경 변수 탭
        self.command_history_tab = QWidget()  # 명령어 히스토리 탭 추가
        
        # 탭 추가
        self.tabs.addTab(self.env_variables_tab, 'GPU/환경 변수')
        self.tabs.addTab(self.config_tab, '설정 옵션')
        self.tabs.addTab(self.custom_tab, '사용자 정의 설정')
        self.tabs.addTab(self.pre_commands_tab, '사전 명령어')
        self.tabs.addTab(self.command_history_tab, '명령어 히스토리')
        
        # Setup layouts for tabs
        self.config_layout = QVBoxLayout()
        self.custom_layout = QVBoxLayout()
        self.pre_commands_layout = QVBoxLayout()
        self.env_variables_layout = QVBoxLayout()
        self.command_history_layout = QVBoxLayout()
        
        self.config_tab.setLayout(self.config_layout)
        self.custom_tab.setLayout(self.custom_layout)
        self.pre_commands_tab.setLayout(self.pre_commands_layout)
        self.env_variables_tab.setLayout(self.env_variables_layout)
        self.command_history_tab.setLayout(self.command_history_layout)
        
        # Scroll areas for config tab
        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_layout.addWidget(self.config_scroll)
        
        # Content widgets for scroll areas
        self.config_content = QWidget()
        self.config_scroll.setWidget(self.config_content)
        self.config_content_layout = QVBoxLayout(self.config_content)
        
        # 사용자 정의 설정 탭 구성
        self.setup_custom_config_tab()
        
        # 사전 명령어 탭 구성
        self.setup_pre_commands_tab()
        
        # 환경 변수 탭 구성
        self.setup_env_variables_tab()
        
        # 명령어 히스토리 탭 구성
        self.setup_command_history_tab()
        
        # Command output section
        command_group = QGroupBox('생성된 명령어')
        command_layout = QVBoxLayout()
        
        # 설명 입력 추가
        description_layout = QHBoxLayout()
        description_label = QLabel('명령어 설명:')
        self.command_description_edit = QLineEdit()
        self.command_description_edit.setPlaceholderText('명령어 설명 (히스토리에 저장됨)')
        
        description_layout.addWidget(description_label)
        description_layout.addWidget(self.command_description_edit)
        
        command_layout.addLayout(description_layout)
        
        self.command_output = QTextEdit()
        self.command_output.setReadOnly(True)
        command_layout.addWidget(self.command_output)
        
        button_layout = QHBoxLayout()
        
        generate_button = QPushButton('명령어 생성')
        generate_button.clicked.connect(self.generate_command)
        
        run_button = QPushButton('명령어 실행')
        run_button.clicked.connect(self.run_command)
        
        save_button = QPushButton('히스토리에 저장')
        save_button.clicked.connect(self.save_command_to_history)
        
        button_layout.addWidget(generate_button)
        button_layout.addWidget(run_button)
        button_layout.addWidget(save_button)
        
        command_layout.addLayout(button_layout)
        
        command_group.setLayout(command_layout)
        main_layout.addWidget(command_group)
        
    def setup_custom_config_tab(self):
        """사용자 정의 설정 탭 설정"""
        # 테이블 위젯 생성
        self.custom_config_table = QTableWidget()
        self.custom_config_table.setColumnCount(3)
        self.custom_config_table.setHorizontalHeaderLabels(['파라미터', '값', '활성화'])
        self.custom_config_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.custom_config_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.custom_config_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        add_row_button = QPushButton('행 추가')
        add_row_button.clicked.connect(self.add_custom_config_row)
        delete_row_button = QPushButton('행 삭제')
        delete_row_button.clicked.connect(self.delete_custom_config_row)
        save_config_button = QPushButton('설정 저장')
        save_config_button.clicked.connect(self.save_custom_configs)
        
        button_layout.addWidget(add_row_button)
        button_layout.addWidget(delete_row_button)
        button_layout.addWidget(save_config_button)
        
        # 레이아웃에 위젯 추가
        self.custom_layout.addWidget(self.custom_config_table)
        self.custom_layout.addLayout(button_layout)
        
        # 저장된 사용자 정의 설정 로드
        self.load_custom_configs()
    
    def setup_pre_commands_tab(self):
        """사전 실행 명령어 탭 설정"""
        # 설명 레이블
        description_label = QLabel("학습 명령어 실행 전에 실행할 명령어를 설정합니다. (예: conda activate my_env)")
        description_label.setWordWrap(True)
        self.pre_commands_layout.addWidget(description_label)
        
        # 테이블 위젯 생성
        self.pre_commands_table = QTableWidget()
        self.pre_commands_table.setColumnCount(3)
        self.pre_commands_table.setHorizontalHeaderLabels(['명령어', '설명', '활성화'])
        self.pre_commands_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.pre_commands_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.pre_commands_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        add_cmd_button = QPushButton('명령어 추가')
        add_cmd_button.clicked.connect(self.add_pre_command_row)
        delete_cmd_button = QPushButton('명령어 삭제')
        delete_cmd_button.clicked.connect(self.delete_pre_command_row)
        move_up_button = QPushButton('위로 이동')
        move_up_button.clicked.connect(lambda: self.move_pre_command(-1))
        move_down_button = QPushButton('아래로 이동')
        move_down_button.clicked.connect(lambda: self.move_pre_command(1))
        save_cmd_button = QPushButton('명령어 저장')
        save_cmd_button.clicked.connect(self.save_pre_commands)
        
        button_layout.addWidget(add_cmd_button)
        button_layout.addWidget(delete_cmd_button)
        button_layout.addWidget(move_up_button)
        button_layout.addWidget(move_down_button)
        button_layout.addWidget(save_cmd_button)
        
        # 레이아웃에 위젯 추가
        self.pre_commands_layout.addWidget(self.pre_commands_table)
        self.pre_commands_layout.addLayout(button_layout)
        
        # 저장된 사전 명령어 로드
        self.load_pre_commands()
    
    def add_pre_command_row(self):
        """사전 명령어에 새 행 추가"""
        row = self.pre_commands_table.rowCount()
        self.pre_commands_table.insertRow(row)
        
        # 명령어 열
        cmd_item = QTableWidgetItem("")
        self.pre_commands_table.setItem(row, 0, cmd_item)
        
        # 설명 열
        desc_item = QTableWidgetItem("")
        self.pre_commands_table.setItem(row, 1, desc_item)
        
        # 활성화 체크박스 열
        checkbox = QTableWidgetItem()
        checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        checkbox.setCheckState(Qt.Checked)
        self.pre_commands_table.setItem(row, 2, checkbox)
    
    def delete_pre_command_row(self):
        """선택한 사전 명령어 행 삭제"""
        selected_rows = list(set([index.row() for index in self.pre_commands_table.selectedIndexes()]))
        if not selected_rows:
            self.show_status_message('삭제할 명령어를 선택해주세요.', True)
            return
            
        # 선택한 행을 역순으로 삭제 (인덱스 변화 방지)
        for row in sorted(selected_rows, reverse=True):
            self.pre_commands_table.removeRow(row)
        
        self.show_status_message('선택한 명령어가 삭제되었습니다.')
    
    def move_pre_command(self, direction):
        """사전 명령어 순서 이동 (위/아래)"""
        selected_rows = list(set([index.row() for index in self.pre_commands_table.selectedIndexes()]))
        if not selected_rows:
            self.show_status_message('이동할 명령어를 선택해주세요.', True)
            return
        
        # 한 번에 하나의 행만 이동 가능
        if len(selected_rows) > 1:
            self.show_status_message('한 번에 하나의 명령어만 이동할 수 있습니다.', True)
            return
        
        current_row = selected_rows[0]
        target_row = current_row + direction
        
        # 테이블 범위를 벗어나면 이동 불가
        if target_row < 0 or target_row >= self.pre_commands_table.rowCount():
            return
        
        # 행 데이터 백업
        command = self.pre_commands_table.item(current_row, 0).text()
        description = self.pre_commands_table.item(current_row, 1).text()
        is_enabled = self.pre_commands_table.item(current_row, 2).checkState()
        
        # 현재 행 삭제
        self.pre_commands_table.removeRow(current_row)
        
        # 새 위치에 행 삽입
        self.pre_commands_table.insertRow(target_row)
        
        # 데이터 복원
        self.pre_commands_table.setItem(target_row, 0, QTableWidgetItem(command))
        self.pre_commands_table.setItem(target_row, 1, QTableWidgetItem(description))
        checkbox = QTableWidgetItem()
        checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        checkbox.setCheckState(is_enabled)
        self.pre_commands_table.setItem(target_row, 2, checkbox)
        
        # 이동된 행 선택
        self.pre_commands_table.selectRow(target_row)
    
    def save_pre_commands(self):
        """사전 명령어 저장"""
        pre_commands = []
        for row in range(self.pre_commands_table.rowCount()):
            command = self.pre_commands_table.item(row, 0).text().strip()
            description = self.pre_commands_table.item(row, 1).text().strip()
            is_enabled = self.pre_commands_table.item(row, 2).checkState() == Qt.Checked
            
            if command:  # 명령어가 빈 값이 아닌 경우만 저장
                pre_commands.append({
                    'command': command,
                    'description': description,
                    'enabled': is_enabled
                })
        
        self.pre_commands = pre_commands
        self.settings['pre_commands'] = pre_commands
        self.save_settings()
        self.show_status_message('사전 명령어가 저장되었습니다.')
    
    def load_pre_commands(self):
        """저장된 사전 명령어 로드"""
        self.pre_commands = self.settings.get('pre_commands', [])
        
        # 테이블 초기화
        self.pre_commands_table.setRowCount(0)
        
        # 저장된 명령어 추가
        for cmd in self.pre_commands:
            row = self.pre_commands_table.rowCount()
            self.pre_commands_table.insertRow(row)
            
            # 명령어 열
            cmd_item = QTableWidgetItem(cmd.get('command', ''))
            self.pre_commands_table.setItem(row, 0, cmd_item)
            
            # 설명 열
            desc_item = QTableWidgetItem(cmd.get('description', ''))
            self.pre_commands_table.setItem(row, 1, desc_item)
            
            # 활성화 체크박스 열
            checkbox = QTableWidgetItem()
            checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox.setCheckState(Qt.Checked if cmd.get('enabled', True) else Qt.Unchecked)
            self.pre_commands_table.setItem(row, 2, checkbox)
    
    def add_custom_config_row(self):
        """사용자 정의 설정에 새 행 추가"""
        row = self.custom_config_table.rowCount()
        self.custom_config_table.insertRow(row)
        
        # 파라미터 열
        param_item = QTableWidgetItem("")
        self.custom_config_table.setItem(row, 0, param_item)
        
        # 값 열
        value_item = QTableWidgetItem("")
        self.custom_config_table.setItem(row, 1, value_item)
        
        # 활성화 체크박스 열
        checkbox = QTableWidgetItem()
        checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        checkbox.setCheckState(Qt.Checked)
        self.custom_config_table.setItem(row, 2, checkbox)
    
    def delete_custom_config_row(self):
        """선택한 행 삭제"""
        selected_rows = list(set([index.row() for index in self.custom_config_table.selectedIndexes()]))
        if not selected_rows:
            self.show_status_message('삭제할 행을 선택해주세요.', True)
            return
            
        # 선택한 행을 역순으로 삭제 (인덱스 변화 방지)
        for row in sorted(selected_rows, reverse=True):
            self.custom_config_table.removeRow(row)
        
        self.show_status_message('선택한 행이 삭제되었습니다.')
    
    def save_custom_configs(self):
        """사용자 정의 설정 저장"""
        custom_configs = []
        for row in range(self.custom_config_table.rowCount()):
            param = self.custom_config_table.item(row, 0).text().strip()
            value = self.custom_config_table.item(row, 1).text().strip()
            is_enabled = self.custom_config_table.item(row, 2).checkState() == Qt.Checked
            
            if param:  # 파라미터가 빈 값이 아닌 경우만 저장
                custom_configs.append({
                    'param': param,
                    'value': value,
                    'enabled': is_enabled
                })
        
        self.custom_configs = custom_configs
        self.settings['custom_configs'] = custom_configs
        self.save_settings()
        self.show_status_message('사용자 정의 설정이 저장되었습니다.')
    
    def load_custom_configs(self):
        """저장된 사용자 정의 설정 로드"""
        self.custom_configs = self.settings.get('custom_configs', [])
        
        # 테이블 초기화
        self.custom_config_table.setRowCount(0)
        
        # 저장된 설정 추가
        for config in self.custom_configs:
            row = self.custom_config_table.rowCount()
            self.custom_config_table.insertRow(row)
            
            # 파라미터 열
            param_item = QTableWidgetItem(config.get('param', ''))
            self.custom_config_table.setItem(row, 0, param_item)
            
            # 값 열
            value_item = QTableWidgetItem(config.get('value', ''))
            self.custom_config_table.setItem(row, 1, value_item)
            
            # 활성화 체크박스 열
            checkbox = QTableWidgetItem()
            checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox.setCheckState(Qt.Checked if config.get('enabled', True) else Qt.Unchecked)
            self.custom_config_table.setItem(row, 2, checkbox)
    
    def set_default_ini_path(self):
        """현재 선택된 INI 파일을 기본 경로로 설정"""
        current_path = self.file_path_edit.text()
        if not current_path or not os.path.exists(current_path):
            self.show_status_message('먼저 INI 파일을 선택해주세요.', True)
            return
            
        self.settings['default_ini_path'] = current_path
        self.save_settings()
        self.show_status_message(f'기본 INI 파일 경로가 설정되었습니다: {current_path}')
        
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
            self.show_status_message('INI 파일을 선택해주세요.', True)
            return
        
        self.config_file = config_path
        self.config_data.clear()
        self.checkboxes.clear()
        
        # Clear the content layouts
        self._clear_layout(self.config_content_layout)
        
        # Parse config file - 원본 대소문자 유지를 위해 옵션 추가
        config = configparser.ConfigParser()
        
        # 대소문자 유지 옵션을 추가
        try:
            # ConfigParser 인스턴스 생성 시 대소문자를 유지하는 옵션 설정
            config._sections = dict()  # 섹션 이름 대소문자 유지를 위한 설정
            
            # 파일을 직접 읽어서 처리
            with open(config_path, encoding='utf-8') as f:
                config.read_file(f)
            
            # 디버그 메시지
            print(f"INI 파일 대소문자 유지 로드 성공: {config.sections()}")
            
        except Exception as e:
            # 실패 시 기본 방식으로 다시 시도
            print(f"대소문자 유지 로드 실패, 기본 방식으로 재시도: {e}")
            config = configparser.ConfigParser()
            config.read(config_path, encoding='utf-8')
        
        # Process sections and options
        for section in config.sections():
            # 새 컨셉: 섹션이 명령어 파라미터 이름, 변수는 UI에 표시할 이름, 값은 실제 명령어 값
            param_name = section  # 명령어 파라미터 이름
            
            # 모든 설정을 config_content_layout에 추가 (시스템 설정 구분 없이)
            layout = self.config_content_layout
            
            # Create a group box for each section (parameter)
            group_box = QGroupBox(param_name)
            group_layout = QVBoxLayout()
            
            # 체크박스를 그리드 레이아웃으로 변경하여 여러 줄로 표시되도록 함
            checkbox_grid = QGridLayout()
            checkbox_grid.setAlignment(Qt.AlignLeft)
            checkbox_grid.setHorizontalSpacing(10)  # 체크박스 간 간격 설정
            checkbox_grid.setVerticalSpacing(5)     # 줄 간 간격 설정
            
            # 체크박스 관리 구조 변경: 배열 -> 딕셔너리 (키로 원래 이름 사용)
            checkbox_dict = {}
            
            # 섹션 내 옵션이 없는 경우
            if len(config[section]) == 0:
                # 빈 섹션인 경우 "ON" 옵션 하나만 추가
                checkbox = QCheckBox("ON")
                checkbox.setChecked(True)  # 기본적으로 체크됨
                checkbox.setToolTip("빈 섹션: 값 없이 파라미터만 추가")
                checkbox_grid.addWidget(checkbox, 0, 0)
                checkbox_dict["ON"] = checkbox
            else:
                # 섹션 내 옵션이 있는 경우 그리드에 추가 (한 줄에 최대 4개)
                max_columns = 4
                row, col = 0, 0
                
                # 첫 번째 옵션을 체크할지 여부를 추적하는 변수
                first_option = True
                
                for option_name in config[section]:
                    # option_name: INI 파일에 적힌 원본 이름 (대소문자 유지)
                    cmd_value = config[section][option_name]  # 실제 명령어 값
                    
                    # 체크박스 레이블에 원본 이름 사용 (대소문자 유지)
                    checkbox = QCheckBox(str(option_name))  # str()로 감싸서 대소문자 유지
                    
                    # 첫 번째 옵션만 체크, 나머지는 체크 해제
                    checkbox.setChecked(first_option)
                    first_option = False  # 첫 번째 옵션 처리 후 플래그 해제
                    
                    checkbox.setToolTip(f"실제 값: {cmd_value}")  # 툴툽으로 실제 값 표시
                    
                    # 그리드에 체크박스 추가
                    checkbox_grid.addWidget(checkbox, row, col)
                    checkbox_dict[option_name] = checkbox
                    
                    # 다음 위치 계산
                    col += 1
                    if col >= max_columns:
                        col = 0
                        row += 1
            
            # 체크박스 그리드를 전체 레이아웃에 추가
            group_layout.addLayout(checkbox_grid)
            group_box.setLayout(group_layout)
            layout.addWidget(group_box)
            
            # Store checkboxes and values for later reference
            self.checkboxes[section] = checkbox_dict
            
            # 디버그용: 체크박스 정보 출력
            print(f"섹션 '{section}'에 체크박스 {len(checkbox_dict)}개 생성")
        
        # Message if successful
        self.show_status_message('설정 파일을 성공적으로 로드했습니다.')
    
    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self._clear_layout(item.layout())
    
    def show_status_message(self, message, is_error=False):
        """상태 표시줄에 메시지 표시"""
        if is_error:
            self.statusBar.setStyleSheet("background-color: #ffcccc;")
        else:
            self.statusBar.setStyleSheet("background-color: #ccffcc;")
        self.statusBar.showMessage(message, 5000)  # 5초 동안 메시지 표시
    
    def setup_env_variables_tab(self):
        """환경 변수 설정 탭 구성"""
        # 설명 레이블
        description_label = QLabel("CUDA_VISIBLE_DEVICES와 같은 GPU 관련 환경 변수나 시스템 환경 변수를 설정합니다.\n"
                                  "이 환경 변수들은 명령어 실행 시 가장 먼저 적용됩니다.")
        description_label.setWordWrap(True)
        self.env_variables_layout.addWidget(description_label)
        
        # GPU 설정 바로가기 그룹
        gpu_group = QGroupBox("GPU 설정 바로가기")
        gpu_layout = QVBoxLayout()
        
        # GPU 선택 레이아웃
        gpu_select_layout = QHBoxLayout()
        gpu_select_label = QLabel("CUDA_VISIBLE_DEVICES:")
        self.gpu_select_edit = QLineEdit()
        self.gpu_select_edit.setPlaceholderText("0,1,2,3")
        self.gpu_select_edit.setToolTip("사용할 GPU 인덱스를 쉼표로 구분하여 입력하세요. 예: 0,1,2")
        
        gpu_select_layout.addWidget(gpu_select_label)
        gpu_select_layout.addWidget(self.gpu_select_edit)
        
        # GPU 메모리 제한 레이아웃
        gpu_memory_layout = QHBoxLayout()
        gpu_memory_label = QLabel("GPU 메모리 제한:")
        self.gpu_memory_edit = QLineEdit()
        self.gpu_memory_edit.setPlaceholderText("4096")
        self.gpu_memory_edit.setToolTip("GPU 메모리 사용량을 MB 단위로 제한합니다. 비워두면 적용하지 않습니다.")
        
        gpu_memory_layout.addWidget(gpu_memory_label)
        gpu_memory_layout.addWidget(self.gpu_memory_edit)
        
        # GPU 바로가기 버튼
        add_cuda_button = QPushButton("CUDA 설정 추가")
        add_cuda_button.clicked.connect(self.add_cuda_to_env_table)
        
        gpu_layout.addLayout(gpu_select_layout)
        gpu_layout.addLayout(gpu_memory_layout)
        gpu_layout.addWidget(add_cuda_button)
        
        gpu_group.setLayout(gpu_layout)
        self.env_variables_layout.addWidget(gpu_group)
        
        # 테이블 위젯 생성
        self.env_variables_table = QTableWidget()
        self.env_variables_table.setColumnCount(3)
        self.env_variables_table.setHorizontalHeaderLabels(['환경 변수명', '값', '활성화'])
        self.env_variables_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.env_variables_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.env_variables_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        add_var_button = QPushButton('변수 추가')
        add_var_button.clicked.connect(self.add_env_variable_row)
        delete_var_button = QPushButton('변수 삭제')
        delete_var_button.clicked.connect(self.delete_env_variable_row)
        save_var_button = QPushButton('변수 저장')
        save_var_button.clicked.connect(self.save_env_variables)
        
        button_layout.addWidget(add_var_button)
        button_layout.addWidget(delete_var_button)
        button_layout.addWidget(save_var_button)
        
        # 레이아웃에 위젯 추가
        self.env_variables_layout.addWidget(self.env_variables_table)
        self.env_variables_layout.addLayout(button_layout)
        
        # 저장된 환경 변수 로드
        self.load_env_variables()
    
    def add_cuda_to_env_table(self):
        """GPU 설정을 환경 변수 테이블에 추가"""
        # CUDA_VISIBLE_DEVICES 추가
        cuda_devices = self.gpu_select_edit.text().strip()
        if cuda_devices:
            # 기존 행이 있는지 확인
            for row in range(self.env_variables_table.rowCount()):
                if self.env_variables_table.item(row, 0).text() == "CUDA_VISIBLE_DEVICES":
                    # 기존 행 업데이트
                    self.env_variables_table.item(row, 1).setText(cuda_devices)
                    self.env_variables_table.item(row, 2).setCheckState(Qt.Checked)
                    break
            else:
                # 새 행 추가
                row = self.env_variables_table.rowCount()
                self.env_variables_table.insertRow(row)
                
                name_item = QTableWidgetItem("CUDA_VISIBLE_DEVICES")
                self.env_variables_table.setItem(row, 0, name_item)
                
                value_item = QTableWidgetItem(cuda_devices)
                self.env_variables_table.setItem(row, 1, value_item)
                
                checkbox = QTableWidgetItem()
                checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checkbox.setCheckState(Qt.Checked)
                self.env_variables_table.setItem(row, 2, checkbox)
        
        # TF_MEMORY_LIMIT 추가
        memory_limit = self.gpu_memory_edit.text().strip()
        if memory_limit:
            for row in range(self.env_variables_table.rowCount()):
                if self.env_variables_table.item(row, 0).text() == "TF_MEMORY_LIMIT":
                    # 기존 행 업데이트
                    self.env_variables_table.item(row, 1).setText(memory_limit)
                    self.env_variables_table.item(row, 2).setCheckState(Qt.Checked)
                    break
            else:
                # 새 행 추가
                row = self.env_variables_table.rowCount()
                self.env_variables_table.insertRow(row)
                
                name_item = QTableWidgetItem("TF_MEMORY_LIMIT")
                self.env_variables_table.setItem(row, 0, name_item)
                
                value_item = QTableWidgetItem(memory_limit)
                self.env_variables_table.setItem(row, 1, value_item)
                
                checkbox = QTableWidgetItem()
                checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checkbox.setCheckState(Qt.Checked)
                self.env_variables_table.setItem(row, 2, checkbox)
        
        self.show_status_message("GPU 설정이 환경 변수에 추가되었습니다.")
        self.save_env_variables()
    
    def add_env_variable_row(self):
        """환경 변수에 새 행 추가"""
        row = self.env_variables_table.rowCount()
        self.env_variables_table.insertRow(row)
        
        # 환경 변수명 열
        name_item = QTableWidgetItem("")
        self.env_variables_table.setItem(row, 0, name_item)
        
        # 값 열
        value_item = QTableWidgetItem("")
        self.env_variables_table.setItem(row, 1, value_item)
        
        # 활성화 체크박스 열
        checkbox = QTableWidgetItem()
        checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        checkbox.setCheckState(Qt.Checked)
        self.env_variables_table.setItem(row, 2, checkbox)
    
    def delete_env_variable_row(self):
        """선택한 환경 변수 행 삭제"""
        selected_rows = list(set([index.row() for index in self.env_variables_table.selectedIndexes()]))
        if not selected_rows:
            self.show_status_message('삭제할 환경 변수를 선택해주세요.', True)
            return
            
        # 선택한 행을 역순으로 삭제 (인덱스 변화 방지)
        for row in sorted(selected_rows, reverse=True):
            self.env_variables_table.removeRow(row)
        
        self.show_status_message('선택한 환경 변수가 삭제되었습니다.')
    
    def save_env_variables(self):
        """환경 변수 저장"""
        env_variables = []
        for row in range(self.env_variables_table.rowCount()):
            name = self.env_variables_table.item(row, 0).text().strip()
            value = self.env_variables_table.item(row, 1).text().strip()
            is_enabled = self.env_variables_table.item(row, 2).checkState() == Qt.Checked
            
            if name:  # 환경 변수명이 빈 값이 아닌 경우만 저장
                env_variables.append({
                    'name': name,
                    'value': value,
                    'enabled': is_enabled
                })
        
        self.env_variables = env_variables
        self.settings['env_variables'] = env_variables
        self.save_settings()
        self.show_status_message('환경 변수가 저장되었습니다.')
    
    def load_env_variables(self):
        """저장된 환경 변수 로드"""
        self.env_variables = self.settings.get('env_variables', [])
        
        # 테이블 초기화
        self.env_variables_table.setRowCount(0)
        
        # 저장된 환경 변수 추가
        for var in self.env_variables:
            row = self.env_variables_table.rowCount()
            self.env_variables_table.insertRow(row)
            
            # 환경 변수명 열
            name_item = QTableWidgetItem(var.get('name', ''))
            self.env_variables_table.setItem(row, 0, name_item)
            
            # 값 열
            value_item = QTableWidgetItem(var.get('value', ''))
            self.env_variables_table.setItem(row, 1, value_item)
            
            # 활성화 체크박스 열
            checkbox = QTableWidgetItem()
            checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox.setCheckState(Qt.Checked if var.get('enabled', True) else Qt.Unchecked)
            self.env_variables_table.setItem(row, 2, checkbox)
            
            # GPU 관련 변수면 해당 설정 폼에도 표시
            if var.get('name') == "CUDA_VISIBLE_DEVICES" and var.get('enabled', True):
                self.gpu_select_edit.setText(var.get('value', ''))
            elif var.get('name') == "TF_MEMORY_LIMIT" and var.get('enabled', True):
                self.gpu_memory_edit.setText(var.get('value', ''))
    
    def setup_command_history_tab(self):
        """명령어 히스토리 탭 설정"""
        # 설명 레이블
        description_label = QLabel("이전에 생성/실행된 명령어 목록입니다. 재사용하거나 삭제할 수 있습니다.")
        description_label.setWordWrap(True)
        self.command_history_layout.addWidget(description_label)
        
        # 테이블 위젯 생성
        self.command_history_table = QTableWidget()
        self.command_history_table.setColumnCount(4)
        self.command_history_table.setHorizontalHeaderLabels(['ID', '시간', '설명', '작업'])
        self.command_history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.command_history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.command_history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.command_history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        refresh_button = QPushButton('새로고침')
        refresh_button.clicked.connect(self.refresh_command_history)
        clear_button = QPushButton('모두 삭제')
        clear_button.clicked.connect(self.clear_command_history)
        
        button_layout.addWidget(refresh_button)
        button_layout.addWidget(clear_button)
        
        # 레이아웃에 위젯 추가
        self.command_history_layout.addWidget(self.command_history_table)
        self.command_history_layout.addLayout(button_layout)
        
        # 명령어 히스토리 로드
        self.refresh_command_history()
    
    def refresh_command_history(self):
        """명령어 히스토리 새로고침"""
        self.command_history_table.setRowCount(0)
        
        # 모든 명령어 가져오기
        commands = self.command_manager.get_all_commands()
        commands.reverse()  # 최신 명령어가 위에 오도록 역순 정렬
        
        # 테이블에 표시
        for i, cmd in enumerate(commands):
            self.command_history_table.insertRow(i)
            
            # ID
            id_item = QTableWidgetItem(cmd.get('id', ''))
            self.command_history_table.setItem(i, 0, id_item)
            
            # 시간
            time_item = QTableWidgetItem(cmd.get('timestamp', ''))
            self.command_history_table.setItem(i, 1, time_item)
            
            # 설명
            desc_item = QTableWidgetItem(cmd.get('description', ''))
            desc_item.setToolTip(cmd.get('command', ''))
            self.command_history_table.setItem(i, 2, desc_item)
            
            # 작업 버튼
            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            button_layout.setContentsMargins(0, 0, 0, 0)
            
            load_button = QPushButton('로드')
            load_button.clicked.connect(lambda _, cmd_id=cmd.get('id'): self.load_command_from_history(cmd_id))
            
            run_button = QPushButton('실행')
            run_button.clicked.connect(lambda _, cmd_id=cmd.get('id'): self.run_command_from_history(cmd_id))
            
            delete_button = QPushButton('삭제')
            delete_button.clicked.connect(lambda _, cmd_id=cmd.get('id'): self.delete_command_from_history(cmd_id))
            
            button_layout.addWidget(load_button)
            button_layout.addWidget(run_button)
            button_layout.addWidget(delete_button)
            
            self.command_history_table.setCellWidget(i, 3, button_widget)
    
    def load_command_from_history(self, command_id):
        """히스토리에서 명령어 로드"""
        command = self.command_manager.get_command_by_id(command_id)
        if command:
            self.command_output.setText(command.get('command', ''))
            self.command_description_edit.setText(command.get('description', ''))
            self.show_status_message(f'명령어 #{command_id}를 로드했습니다.')
    
    def run_command_from_history(self, command_id):
        """히스토리에서 명령어 실행"""
        command = self.command_manager.get_command_by_id(command_id)
        if command:
            self.command_output.setText(command.get('command', ''))
            self.command_description_edit.setText(command.get('description', ''))
            self.run_command()
    
    def delete_command_from_history(self, command_id):
        """히스토리에서 명령어 삭제"""
        # 삭제 확인 다이얼로그
        reply = QMessageBox.question(self, '명령어 삭제', 
                                     f'명령어 #{command_id}를 삭제하시겠습니까?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            success = self.command_manager.delete_command(command_id)
            if success:
                self.show_status_message(f'명령어 #{command_id}가 삭제되었습니다.')
                self.refresh_command_history()
            else:
                self.show_status_message('명령어 삭제 중 오류가 발생했습니다.', True)
    
    def clear_command_history(self):
        """모든 명령어 히스토리 삭제"""
        # 삭제 확인 다이얼로그
        reply = QMessageBox.question(self, '모든 명령어 삭제', 
                                     '모든 명령어 히스토리를 삭제하시겠습니까?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 모든 명령어 가져와서 하나씩 삭제
            commands = self.command_manager.get_all_commands()
            for cmd in commands:
                self.command_manager.delete_command(cmd.get('id'))
            
            self.show_status_message('모든 명령어 히스토리가 삭제되었습니다.')
            self.refresh_command_history()
    
    def save_command_to_history(self):
        """현재 명령어를 히스토리에 저장"""
        command = self.command_output.toPlainText().strip()
        description = self.command_description_edit.text().strip()
        
        if not command:
            self.show_status_message('저장할 명령어가 없습니다.', True)
            return
        
        if not description:
            description = f"생성된 명령어 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        command_id = self.command_manager.add_command(command, description)
        if command_id:
            self.show_status_message(f'명령어가 히스토리에 저장되었습니다. (ID: {command_id})')
            self.refresh_command_history()
        else:
            self.show_status_message('명령어 저장 중 오류가 발생했습니다.', True)
    
    def generate_command(self):
        """현재 UI 상태를 기반으로 명령어 생성"""
        script_path = self.script_edit.text().strip()
        if not script_path:
            self.command_output.setText("스크립트 파일을 선택해주세요.")
            return
            
        # 파이썬 실행 명령어
        python_cmd = "python" if platform.system() == "Windows" else "python3"
        command_parts = [python_cmd, script_path]
        
        # 각 섹션별 선택된 옵션 처리
        for param_name, checkboxes in self.checkboxes.items():
            selected_values = []
            empty_section = False
            
            # 체크박스 딕셔너리 반복
            for option_name, checkbox in checkboxes.items():
                if checkbox.isChecked():
                    # 빈 섹션 처리: ON 옵션이 체크되고 다른 옵션이 없으면 빈 섹션으로 간주
                    if option_name == "ON" and len(checkboxes) == 1:
                        empty_section = True
                    else:
                        # INI 파일에서 실제 값 가져오기
                        if self.config_file and os.path.exists(self.config_file):
                            try:
                                config = configparser.ConfigParser()
                                with open(self.config_file, encoding='utf-8') as f:
                                    config.read_file(f)
                                
                                # INI 파일에서 해당 옵션 값 가져오기
                                if param_name in config and option_name in config[param_name]:
                                    value = config[param_name][option_name]
                                    if value:  # 값이 있으면 추가
                                        selected_values.append(value)
                                    else:  # 값이 없으면 옵션 이름만 추가
                                        selected_values.append(option_name)
                                else:
                                    # INI에 없는 경우 옵션 이름 그대로 사용
                                    selected_values.append(option_name)
                            except Exception as e:
                                print(f"INI 파일 읽기 오류: {e}")
                                selected_values.append(option_name)
                        else:
                            # INI 파일이 없는 경우 옵션 이름 사용
                            selected_values.append(option_name)
            
            # 빈 섹션이면 파라미터만 추가
            if empty_section:
                command_parts.append(f"--{param_name}")
            # 파라미터 값이 있으면 추가
            elif selected_values:
                # 여러 값이 있으면 empty space로 구분하여 추가
                values_str = " ".join(selected_values)
                command_parts.append(f"--{param_name} {values_str}")
        
        # 환경 변수 처리
        env_vars = []
        for var in self.env_variables:
            if var.get('enabled', False):
                env_vars.append(f"{var['name']}={var['value']}")
        
        # 사용자 정의 설정 처리
        for config in self.custom_configs:
            if config.get('enabled', False):
                command_parts.append(f"--{config['param']} {config['value']}")
        
        # 최종 명령어 생성
        env_vars_str = " ".join(env_vars)
        pre_commands_str = ""
        
        # 사전 실행 명령어 처리
        pre_commands_enabled = []
        for cmd in self.pre_commands:
            if cmd.get('enabled', False):
                pre_commands_enabled.append(cmd['command'])
        
        if pre_commands_enabled:
            pre_commands_str = " && ".join(pre_commands_enabled) + " && "
        
        # 환경 변수가 있을 경우 추가
        if env_vars:
            # Windows는 SET, Linux/Mac은 export 사용
            if platform.system() == "Windows":
                env_cmd = " && ".join([f"SET {var}" for var in env_vars]) + " && "
            else:
                env_cmd = " ".join([f"export {var}" for var in env_vars]) + " && "
            
            final_command = f"{pre_commands_str}{env_cmd}{' '.join(command_parts)}"
        else:
            final_command = f"{pre_commands_str}{' '.join(command_parts)}"
        
        # 명령어 출력
        self.command_output.setText(final_command)
        self.show_status_message('명령어가 생성되었습니다.')
        
        # 자동으로 히스토리에 저장 (설명은 현재 날짜/시간 사용)
        if self.command_description_edit.text().strip() == "":
            self.command_description_edit.setText(f"생성된 명령어 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_command(self):
        command = self.command_output.toPlainText()
        if not command:
            self.show_status_message('먼저 명령어를 생성해주세요.', True)
            return
        
        try:
            # 명령어 자동 저장 (실행 시)
            description = self.command_description_edit.text().strip()
            if not description:
                description = f"실행된 명령어 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            command_id = self.command_manager.add_command(command, description)
            
            # 운영체제 별로 다른 방식으로 새 터미널에서 명령어 실행
            platform = sys.platform.lower()
            
            # 활성화된 사전 명령어 가져오기
            pre_commands = []
            for row in range(self.pre_commands_table.rowCount()):
                cmd = self.pre_commands_table.item(row, 0).text().strip()
                is_enabled = self.pre_commands_table.item(row, 2).checkState() == Qt.Checked
                if cmd and is_enabled:
                    pre_commands.append(cmd)
            
            # 사전 명령어와 메인 명령어 결합
            if pre_commands:
                if platform.startswith('win'):  # Windows
                    combined_command = " && ".join(pre_commands + [command])
                else:  # Linux/macOS
                    combined_command = "; ".join(pre_commands + [command])
            else:
                combined_command = command
            
            if platform.startswith('win'):  # Windows
                # Windows에서는 cmd 창에서 명령어 실행
                # /k는 명령 실행 후 창을 유지함
                terminal_command = f'start cmd /k "{combined_command}"'
                subprocess.Popen(terminal_command, shell=True)
                message = '새 명령 프롬프트 창에서 명령어가 실행되었습니다.'
                
            elif platform.startswith('linux'):  # Linux
                # 일반적인 Linux 터미널 에뮬레이터 시도
                terminal_emulators = [
                    ['gnome-terminal', '--', 'bash', '-c', f'{combined_command}; exec bash'],
                    ['konsole', '--', 'bash', '-c', f'{combined_command}; exec bash'],
                    ['xterm', '-e', f'bash -c "{combined_command}; exec bash"'],
                    ['x-terminal-emulator', '-e', f'bash -c "{combined_command}; exec bash"']
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
                    result = subprocess.run(combined_command, shell=True, check=True, 
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True)
                    message = f'명령어가 백그라운드에서 실행되었습니다. (터미널 에뮬레이터를 찾을 수 없음)\n출력:\n{result.stdout}'
                
            elif platform.startswith('darwin'):  # macOS
                # macOS에서는 새 Terminal 창에서 명령어 실행
                terminal_command = ['osascript', '-e', f'tell app "Terminal" to do script "{combined_command}"']
                subprocess.Popen(terminal_command)
                message = '새 Terminal 창에서 명령어가 실행되었습니다.'
                
            else:  # 지원되지 않는 OS
                # 기본 방식으로 실행
                result = subprocess.run(combined_command, shell=True, check=True, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
                message = f'명령어가 성공적으로 실행되었습니다. (별도의 터미널을 지원하지 않는 OS)\n출력:\n{result.stdout}'
            
            self.show_status_message(message)
            
            # 히스토리 업데이트
            self.refresh_command_history()
            
        except subprocess.CalledProcessError as e:
            self.show_status_message(f'명령어 실행 중 오류가 발생했습니다: {e.stderr}', True)
        except Exception as e:
            self.show_status_message(f'명령어 실행 중 예상치 못한 오류가 발생했습니다: {str(e)}', True)

    def update_config_combo(self):
        """구성 콤보박스 업데이트"""
        try:
            # 현재 선택된 구성 저장
            current_config = self.config_combo.currentText()
            
            # 구성 콤보박스 초기화
            self.config_combo.clear()
            self.config_combo.addItem("")  # 빈 항목 추가
            
            # CSV 파일이 존재하는지 확인
            if os.path.exists(self.csv_manager.csv_file_path):
                # CSV 파일 읽기
                configs = []
                with open(self.csv_manager.csv_file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    if len(rows) >= 3:  # 헤더 2행 + 데이터 1행 이상
                        for i, row in enumerate(rows[2:]):
                            if row and len(row) > 0:
                                # 첫 번째 열에 구성 번호 또는 이름이 있음
                                config_name = row[0]
                                configs.append(config_name)
                
                # 구성 콤보박스에 추가
                for config_name in configs:
                    self.config_combo.addItem(f"{config_name}")
            
            # 이전에 선택된 구성 다시 선택
            if current_config:
                index = self.config_combo.findText(current_config)
                if index >= 0:
                    self.config_combo.setCurrentIndex(index)
                    
        except Exception as e:
            self.show_status_message(f"구성 목록 업데이트 오류: {str(e)}", True)
            import traceback
            traceback.print_exc()
    
    def capture_current_config(self):
        """현재 UI 설정을 캡처하여 CSV 파일에 저장"""
        try:
            # 체크박스 설정 캡처
            config_data = {}
            for section_name, section_checkboxes in self.checkboxes.items():
                for param_name, checkbox in section_checkboxes.items():
                    is_checked = checkbox.isChecked()
                    
                    # 체크박스 상태에 따라 값 설정
                    if is_checked:
                        # INI 파일에서 실제 값을 읽어옴
                        value = "1"  # 기본값
                        if self.config_file and os.path.exists(self.config_file):
                            try:
                                config = configparser.ConfigParser()
                                config.read(self.config_file, encoding='utf-8')
                                if section_name in config and param_name in config[section_name]:
                                    value = config[section_name][param_name]
                            except Exception as e:
                                print(f"INI 파일 읽기 오류(값 추출): {e}")
                    else:
                        value = "0"  # 체크 해제된 경우 0으로 저장
                        
                    if section_name not in config_data:
                        config_data[section_name] = {}
                    
                    config_data[section_name][param_name] = {
                        'display_name': param_name,
                        'checked': is_checked,
                        'value': value
                    }
            
            # 명칭 입력하기
            config_name, ok = QInputDialog.getText(
                self, '구성 저장', '구성 이름 (빈 칸은 자동 번호 부여):',
                QLineEdit.Normal, '')
            
            if not ok:
                return
            
            # 구성 저장
            success, idx = self.csv_manager.capture_current_config(
                config_data, self.env_variables, self.custom_configs, self.pre_commands, config_name)
            
            if success:
                self.show_status_message(f"구성 {idx}가 저장되었습니다.")
                self.update_config_combo()
                
                # 저장된 구성 선택
                if idx:
                    for i in range(self.config_combo.count()):
                        if self.config_combo.itemText(i) == str(idx):
                            self.config_combo.setCurrentIndex(i)
                            break
            else:
                self.show_status_message("구성 저장 중 오류가 발생했습니다.", True)
                
        except Exception as e:
            self.show_status_message(f"구성 저장 오류: {str(e)}", True)
            import traceback
            traceback.print_exc()
    
    def load_selected_config(self):
        """현재 선택된 구성 로드"""
        try:
            config_idx = self.config_combo.currentText()
            if not config_idx:
                return
            
            # CSV 파일에서 구성 로드
            result = self.csv_manager.load_config_from_csv(config_idx)
            if result is None:
                self.show_status_message("구성을 로드할 수 없습니다.", True)
                return
                
            checkboxes_config, env_variables, custom_configs, pre_commands = result
            
            # 먼저 모든 체크박스를 해제
            for section_name, section_checkboxes in self.checkboxes.items():
                for param_name, checkbox in section_checkboxes.items():
                    checkbox.setChecked(False)
                    
            # 디버그용: 로드된 구성 출력
            print(f"로드된 체크박스 구성: {checkboxes_config}")
            
            # 체크박스 설정 적용
            for section_name, section_config in checkboxes_config.items():
                if section_name in self.checkboxes:
                    for param_name, param_config in section_config.items():
                        if param_name in self.checkboxes[section_name]:
                            is_checked = param_config.get('checked', False)  # 기본값은 체크 해제됨
                            print(f"체크박스 설정 적용: {section_name}.{param_name}={is_checked}")
                            self.checkboxes[section_name][param_name].setChecked(is_checked)
                        else:
                            print(f"체크박스를 찾을 수 없음: {section_name}.{param_name}")
                else:
                    # 섹션을 찾을 수 없는 경우
                    print(f"섹션을 찾을 수 없음: {section_name}")
            
            # 환경 변수 설정
            self.env_variables = env_variables
            self.load_env_variables()
            
            # 사용자 정의 설정
            self.custom_configs = custom_configs
            self.load_custom_configs()
            
            # 사전 실행 명령어
            self.pre_commands = pre_commands
            self.load_pre_commands()
            
            # 명령어 자동 생성
            self.generate_command()
            
            self.show_status_message(f"구성 {config_idx}이(가) 로드되었습니다.")
            
        except Exception as e:
            self.show_status_message(f"구성 로드 오류: {str(e)}", True)
            import traceback
            traceback.print_exc()

    def show_config_manager(self):
        """구성 관리 다이얼로그 표시"""
        dialog = ConfigManagerDialog(self, self.csv_manager)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            self.update_config_combo()
            self.show_status_message('구성 관리가 완료되었습니다.')

    def show_config_file_locations(self):
        """구성 파일 위치 정보 표시"""
        # 현재 사용 중인 모든 구성 파일 경로
        config_files = {
            "설정 파일 (settings.json)": self.settings_file,
            "CSV 구성 파일 (model_configs.csv)": self.csv_manager.csv_file_path,
            "현재 INI 파일": self.config_file if self.config_file else "로드된 INI 파일 없음",
            "기본 INI 파일": self.settings.get('default_ini_path', '설정된 기본 INI 파일 없음')
        }
        
        # 파일 위치 다이얼로그 표시
        dialog = ConfigFileLocationsDialog(self, config_files)
        dialog.exec_()

# 구성 관리 다이얼로그
class ConfigManagerDialog(QDialog):
    def __init__(self, parent, csv_manager):
        super().__init__(parent)
        self.parent = parent
        self.csv_manager = csv_manager
        self.setWindowTitle('구성 관리')
        self.setMinimumSize(500, 400)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 구성 목록 테이블
        self.config_table = QTableWidget()
        self.config_table.setColumnCount(2)
        self.config_table.setHorizontalHeaderLabels(['구성 번호', '작업'])
        # 구성 번호 열은 내용에 맞게 자동 조절, 작업 열이 남는 공간을 차지하도록 설정
        self.config_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.config_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.config_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # 테이블 행 높이 조정
        self.config_table.verticalHeader().setDefaultSectionSize(36)
        # 헤더 보이지 않게 설정
        self.config_table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.config_table)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        
        change_idx_button = QPushButton('번호 변경')
        change_idx_button.clicked.connect(self.change_config_idx)
        
        delete_button = QPushButton('삭제')
        delete_button.clicked.connect(self.delete_config)
        
        refresh_button = QPushButton('새로고침')
        refresh_button.clicked.connect(self.refresh_config_list)
        
        button_layout.addWidget(change_idx_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(refresh_button)
        
        layout.addLayout(button_layout)
        
        # 닫기 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
        # 초기 구성 목록 로드
        self.refresh_config_list()
    
    def refresh_config_list(self):
        """구성 목록 새로고침"""
        self.config_table.setRowCount(0)
        
        configs = self.csv_manager.get_available_configs()
        for i, config in enumerate(configs):
            self.config_table.insertRow(i)
            
            # 구성 번호
            try:
                # 부동 소수점 문자열('1.0') 처리
                if config.replace('.', '', 1).isdigit() and '.' in config:
                    config_idx = int(float(config))
                else:
                    config_idx = int(config)
                idx_item = QTableWidgetItem(f"구성 #{config_idx}")
            except (ValueError, TypeError):
                # 숫자로 변환할 수 없는 경우 원본 그대로 표시
                idx_item = QTableWidgetItem(f"구성 #{config}")
                
            idx_item.setTextAlignment(Qt.AlignCenter)  # 가운데 정렬
            self.config_table.setItem(i, 0, idx_item)
            
            # 작업 버튼
            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            # 여백 최소화
            button_layout.setContentsMargins(2, 0, 2, 0)
            button_layout.setSpacing(4)
            
            change_idx_button = QPushButton('번호 변경')
            change_idx_button.clicked.connect(lambda _, c=config: self.change_config_idx(c))
            change_idx_button.setMaximumWidth(70)
            
            delete_button = QPushButton('삭제')
            delete_button.clicked.connect(lambda _, c=config: self.delete_config(c))
            delete_button.setMaximumWidth(50)
            
            button_layout.addWidget(change_idx_button)
            button_layout.addWidget(delete_button)
            
            self.config_table.setCellWidget(i, 1, button_widget)
        
        # 내용에 맞게 열 너비 조정
        self.config_table.resizeColumnsToContents()
        # 내용에 맞게 행 높이 조정
        self.config_table.resizeRowsToContents()
    
    def change_config_idx(self, config=None):
        """구성 인덱스 변경"""
        # 선택된 구성 가져오기
        if config is None:
            selected_rows = self.config_table.selectedItems()
            if not selected_rows:
                QMessageBox.warning(self, '경고', '번호를 변경할 구성을 선택해주세요.')
                return
            config_text = selected_rows[0].text()
            config = config_text.split('#')[1]
        
        # 새 인덱스 입력 다이얼로그
        new_idx, ok = QInputDialog.getText(self, '구성 번호 변경', 
                                         f'구성 #{config}의 새 번호를 입력하세요:', 
                                         QLineEdit.Normal, '')
        if not ok or not new_idx or new_idx == config:
            return
            
        # 숫자가 아니면 오류
        if not new_idx.isdigit():
            QMessageBox.warning(self, '오류', '번호는 숫자여야 합니다.')
            return
        
        # 이름 변경
        success = self.csv_manager.rename_config(config, new_idx)
        if success:
            QMessageBox.information(self, '성공', f'구성 번호가 #{new_idx}(으)로 변경되었습니다.')
            self.refresh_config_list()
        else:
            QMessageBox.warning(self, '오류', '구성 번호 변경 중 오류가 발생했습니다.')
    
    def delete_config(self, config=None):
        """구성 삭제"""
        # 선택된 구성 가져오기
        if config is None:
            selected_rows = self.config_table.selectedItems()
            if not selected_rows:
                QMessageBox.warning(self, '경고', '삭제할 구성을 선택해주세요.')
                return
            config_text = selected_rows[0].text()
            config = config_text.split('#')[1]
        
        # 삭제 확인 다이얼로그
        reply = QMessageBox.question(self, '구성 삭제', 
                                    f'구성 #{config}을(를) 삭제하시겠습니까?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 구성 삭제
            success = self.csv_manager.delete_config(config)
            if success:
                QMessageBox.information(self, '성공', f'구성 #{config}이(가) 삭제되었습니다.')
                self.refresh_config_list()
            else:
                QMessageBox.warning(self, '오류', '구성 삭제 중 오류가 발생했습니다.')

# 구성 파일 위치 다이얼로그
class ConfigFileLocationsDialog(QDialog):
    def __init__(self, parent, config_files):
        super().__init__(parent)
        self.setWindowTitle('구성 파일 위치')
        self.setMinimumSize(700, 400)
        self.initUI(config_files)
        
    def initUI(self, config_files):
        layout = QVBoxLayout()
        
        # 설명 레이블
        description = QLabel("프로그램에서 사용하는 구성 파일의 위치입니다.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # 파일 위치 테이블
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(3)
        self.files_table.setHorizontalHeaderLabels(['파일 종류', '경로', '작업'])
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        
        # 테이블에 파일 정보 추가
        self.files_table.setRowCount(len(config_files))
        for i, (file_type, file_path) in enumerate(config_files.items()):
            # 파일 종류
            type_item = QTableWidgetItem(file_type)
            self.files_table.setItem(i, 0, type_item)
            
            # 파일 경로
            path_item = QTableWidgetItem(file_path)
            path_item.setToolTip(file_path)
            self.files_table.setItem(i, 1, path_item)
            
            # 작업 버튼
            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            button_layout.setContentsMargins(0, 0, 0, 0)
            
            # 파일 존재하는 경우에만 버튼 활성화
            if os.path.exists(file_path):
                open_folder_button = QPushButton('폴더 열기')
                open_folder_button.clicked.connect(lambda _, path=file_path: self.open_containing_folder(path))
                button_layout.addWidget(open_folder_button)
            else:
                # 파일이 없는 경우 비활성화된 버튼 표시
                no_file_label = QLabel("파일 없음")
                button_layout.addWidget(no_file_label)
            
            self.files_table.setCellWidget(i, 2, button_widget)
        
        layout.addWidget(self.files_table)
        
        # 닫기 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def open_containing_folder(self, file_path):
        """파일이 있는 폴더 열기"""
        try:
            folder_path = os.path.dirname(file_path)
            
            # 플랫폼에 따라 다른 명령어 실행
            if sys.platform.startswith('win'):  # Windows
                os.startfile(folder_path)
            elif sys.platform.startswith('darwin'):  # macOS
                subprocess.call(['open', folder_path])
            else:  # Linux
                subprocess.call(['xdg-open', folder_path])
                
        except Exception as e:
            QMessageBox.warning(self, '오류', f'폴더를 열 수 없습니다: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    generator = TrainingCommandGenerator()
    generator.show()
    sys.exit(app.exec_()) 