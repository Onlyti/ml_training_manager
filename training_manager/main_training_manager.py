#!/usr/bin/env python
import os
import sys
import time
import logging
import argparse
import threading
import signal
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import re
import pandas as pd
import subprocess

# 상대 모듈 임포트
# add_path = os.path.abspath(os.path.join( os.path.dirname(__file__), '..'))
# sys.path.append(add_path)
from csv_handler import CSVHandler
from process_manager import ProcessManager
from wandb_monitor import WandbMonitor
from terminal_ui import TerminalUI, run_simple_terminal_ui
from notification import NotificationManager
from config_handler import ConfigHandler

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_training_manager.log')
    ]
)
logger = logging.getLogger(__name__)

class MLTrainingManager:
    """
    # ML Training Manager main class
    """
    def __init__(self, args: argparse.Namespace):
        """
        # Initialize ML Training Manager
        # args.csv: Path to ML_Experiment_Table.csv (overrides config file)
        # args.config: Path to INI configuration file
        # args.training_file_path: Path to model default INI file (INI format)
        # args.check_interval: Interval (in seconds) to check training status (overrides config file)
        # args.wandb_entity: WandB entity (username or team name) (overrides config file)
        # args.wandb_project: WandB project name (overrides config file)
        # args.no_ui: Whether to use terminal UI
        """
        self.start_time = time.time()
        self.running = False
        self.monitor_thread = None
        self.terminal_ui = None
        
        # CSV 파일 경로 설정
        self.csv_file_path = None
        
        # 설정 파일 경로 설정: 현재 디렉토리 기준 절대 경로로 변환
        self.ini_config_file_path = os.path.join(args.current_dir, args.config)
        if not os.path.exists(self.ini_config_file_path):
            raise ValueError(f"설정 파일을 찾을 수 없습니다: {self.ini_config_file_path}")
        self.ini_config_handler = ConfigHandler(self.ini_config_file_path)
        
        # 커맨드 라인 인자에서 CSV 파일 경로를 명시적으로 지정한 경우
        if args.csv:
            self.csv_file_path = args.csv
        else:
            # 설정 파일에서 CSV 파일 경로 가져오기
            try:
                self.csv_file_path = self.ini_config_handler.get_csv_file_path()
            except:
                pass
        
        # 여전히 CSV 파일 경로가 없는 경우 기본값 사용
        if not self.csv_file_path:
            self.csv_file_path = "ML_Experiment_Table.csv"
        
        # 현재 디렉토리 기준 절대 경로로 변환
        if not os.path.isabs(self.csv_file_path) and hasattr(args, 'current_dir'):
            self.csv_file_path = os.path.join(args.current_dir, self.csv_file_path)
            
        # CSV 핸들러 초기화
        try:
            self.csv_handler = CSVHandler(self.csv_file_path)
        except Exception as e:
            raise ValueError(f"CSV 파일 초기화 실패: {e}")
        
        # 프로세스 매니저 초기화
        self.process_manager = ProcessManager(self.ini_config_handler)
        
        # WandB 모니터 초기화
        self.wandb_entity = args.wandb_entity
        self.wandb_project = args.wandb_project
        self.wandb_monitor = None
        
        # 초기 설정값
        self.check_interval = args.check_interval or 5  # 기본값: 5초
        self.max_training_process = args.max_training_process or 1  # 기본값: 1개 프로세스
        self.use_terminal_ui = not args.no_ui  # UI 사용 여부
        self.auto_continue = args.auto_continue  # 자동 연속 학습 모드
        self.auto_log_terminal = getattr(args, 'auto_log_terminal', True)  # 자동 로그 터미널 기능 (기본적으로 활성화)
        
        # 알림 매니저 초기화
        self.notification_manager = NotificationManager()
        
        # 설정 파일에서 추가 설정 불러오기
        self._update_from_config()
        
        # 모델 기본 INI 파일 경로 가져오기
        self.training_file_path = args.training_file_path or self.ini_config_handler.get('general', 'training_file_path', '')
        if not self.training_file_path:
            raise ValueError("학습 파일 경로를 지정해야 합니다. (인자 또는 설정 파일): {self.training_file_path}")
        
        # 모델 기본 INI 파일 경로 절대 경로로 변환
        self.training_file_path = os.path.join(args.current_dir, self.training_file_path)
        self.training_file_path = os.path.abspath(self.training_file_path)
        if not os.path.exists(self.training_file_path):
            raise ValueError(f"학습 파일 경로를 찾을 수 없습니다: {self.training_file_path}")

        # 작업 디렉토리
        self.base_dir = self.training_file_path
        
        # 명령행 인자로 제공된 값으로 설정 덮어쓰기
        if args.check_interval:
            self.check_interval = args.check_interval
        if args.wandb_entity:
            self.wandb_entity = args.wandb_entity
        if args.wandb_project:
            self.wandb_project = args.wandb_project
        
        # 모듈 초기화
        csv_file_global_path = os.path.join(args.current_dir, self.csv_file_path)
        if not os.path.exists(csv_file_global_path):
            raise ValueError(f"CSV 파일을 찾을 수 없습니다: {csv_file_global_path}")
        self.csv_handler = CSVHandler(csv_file_global_path)
        self.process_manager = ProcessManager(self.ini_config_handler)
        
        # WandB 설정
        if self.wandb_entity and self.wandb_project:
            self.wandb_monitor = WandbMonitor(self.wandb_entity, self.wandb_project)
        else:
            self.wandb_monitor = None
            logger.warning("WandB 설정이 없어 WandB 모니터링을 사용할 수 없습니다.")
        
        # 이메일 설정이 활성화된 경우 설정
        email_config = self.ini_config_handler.get_email_config()
        self.notification_manager = NotificationManager(
            enable_email=email_config.get('enable', False),
            enable_desktop=self.ini_config_handler.getboolean('notification', 'enable_desktop', True),
            enable_sound=self.ini_config_handler.getboolean('notification', 'enable_sound', True)
        )
        
        # 이메일 설정이 활성화된 경우 설정
        if email_config.get('enable', False):
            self.notification_manager.configure_email(
                smtp_server=email_config.get('smtp_server', ''),
                smtp_port=email_config.get('smtp_port', 587),
                username=email_config.get('username', ''),
                password=email_config.get('password', ''),
                from_addr=email_config.get('from_addr', ''),
                to_addr=email_config.get('to_addr', '')
            )
        
        # 사운드 설정
        self.notification_manager.configure_sound(
            success_sound=self.ini_config_handler.get('notification', 'success_sound', ''),
            error_sound=self.ini_config_handler.get('notification', 'error_sound', '')
        )
        
        # UI 설정
        self.terminal_ui = None
        
        # 모니터링 스레드
        self.monitor_thread = None
        
        # 학습 실행 시작 시간
        self.start_time = None
        
        logger.info("ML 학습 관리자 초기화 완료")
    
    def _update_from_config(self):
        """
        # Update settings from configuration file
        """
        # General settings
        self.check_interval = self.ini_config_handler.getint('general', 'check_interval', 30)
        self.max_training_process = self.ini_config_handler.getint('general', 'max_training_process', 1)
        self.auto_continue = self.ini_config_handler.getboolean('general', 'auto_continue', True)
        
        # WandB settings
        wandb_config = self.ini_config_handler.get_wandb_config()
        self.wandb_entity = wandb_config.get('entity', '')
        self.wandb_project = wandb_config.get('project', 'Controller-Imitator-Multi-Final')
    
    def start(self):
        """
        # Start the training manager
        """
        if self.running:
            logger.warning("이미 실행 중입니다.")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # 프로세스 인덱스 카운터 리셋
        self.process_manager.reset_process_index_counter()
        
        # 모니터링 스레드 시작
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # UI 시작
        if self.use_terminal_ui:
            self.terminal_ui = TerminalUI(
                get_models_callback=self.get_models_status,
                stop_training_callback=self.stop_training,
                exit_callback=self.stop,
                show_log_callback=self.show_process_log,
                show_all_logs_callback=self.show_all_process_logs
            )
            self.terminal_ui.start()
            
            # 초기 로그 추가
            self.terminal_ui.add_log("ML 학습 관리자가 시작되었습니다.", "info")
            self.terminal_ui.add_log(f"CSV 파일: {self.csv_file_path}", "info")
            if self.ini_config_file_path:
                self.terminal_ui.add_log(f"설정 파일: {self.ini_config_file_path}", "info")
            if self.wandb_monitor:
                self.terminal_ui.add_log(f"WandB 프로젝트: {self.wandb_entity}/{self.wandb_project}", "info")
            self.terminal_ui.add_log(f"동시 학습 최대 수: {self.max_training_process}", "info")
            self.terminal_ui.add_log(f"자동 연속 학습: {'활성화' if self.auto_continue else '비활성화'}", "info")
            self.terminal_ui.add_log("단축키 'l'을 눌러 선택한 모델의 로그를 볼 수 있습니다.", "info")
            self.terminal_ui.add_log("단축키 'a'를 눌러 모든 실행 중인 모델의 로그를 볼 수 있습니다.", "info")
        else:
            logger.info("터미널 UI 없이 실행 중입니다.")
            # 간단한 텍스트 기반 UI
            text_ui_thread = threading.Thread(
                target=run_simple_terminal_ui,
                args=(self.get_summary_status,)
            )
            text_ui_thread.daemon = True
            text_ui_thread.start()
        
        logger.info("ML 학습 관리자가 시작되었습니다.")
    
    def stop(self):
        """
        # Stop the training manager
        """
        if not self.running:
            return
        
        self.running = False
        
        # 모든 실행 중인 학습 프로세스 중지
        for model_id in list(self.process_manager.get_all_processes().keys()):
            self.stop_training(model_id)
        
        # UI 종료
        if self.terminal_ui:
            self.terminal_ui.stop()
        
        # 모니터링 스레드 종료 대기
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # 총 실행 시간 계산
        if self.start_time:
            total_runtime = (time.time() - self.start_time) / 3600  # 시간 단위
            logger.info(f"총 실행 시간: {total_runtime:.2f} 시간")
        
        logger.info("ML 학습 관리자가 종료되었습니다.")
    
    def _monitoring_loop(self):
        """
        # Main monitoring loop
        """
        while self.running:
            try:
                # CSV 파일 다시 로드
                self.csv_handler.reload()
                
                # 진행 중인 학습 확인
                self._check_running_trainings()
                
                # 새로운 학습 시작 (최대 동시 실행 개수까지)
                self._start_new_trainings()
                
                # 학습 log를 별도의 터미널 창으로 출력 (설정된 경우)
                if self.auto_log_terminal:
                    self._print_training_logs()
                
                # 프로세스 정리
                self.process_manager.cleanup_old_processes()
                
            except Exception as e:
                logger.error(f"모니터링 루프 중 오류 발생: {e}")
                if self.terminal_ui:
                    self.terminal_ui.add_log(f"오류 발생: {e}", "error")
            
            # 지정된 간격으로 체크
            time.sleep(self.check_interval)
    
    def _check_running_trainings(self):
        """
        # Check status of running trainings
        """
        # 학습 중으로 표시된 모델 가져오기
        models_in_training = self.csv_handler.get_models_in_training()
        
        for _, model in models_in_training.iterrows():
            model_id = model['ID']
            
            # 실제로 프로세스가 실행 중인지 확인
            if not self.process_manager.is_process_running(model_id):
                # 프로세스가 실행 중이 아니면 WandB 상태 확인
                run_id = model.get('WandbRunID')
                
                if not pd.isna(run_id) and run_id and self.wandb_monitor:
                    # WandB에서 상태 확인
                    if self.wandb_monitor.is_run_finished(run_id):
                        if self.wandb_monitor.is_run_crashed(run_id):
                            logger.warning(f"모델 {model_id}의 학습이 크래시되었습니다.")
                            self.csv_handler.update_model_status(model_id, "Crash")
                            if self.terminal_ui:
                                self.terminal_ui.add_log(f"모델 {model_id}의 학습이 크래시되었습니다.", "error")
                            
                            # 알림 보내기
                            self.notification_manager.notify_training_crashed(
                                model_id=model_id,
                                model_name=model.get('Name', ''),
                                crash_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        else:
                            logger.info(f"모델 {model_id}의 학습이 완료되었습니다.")
                            self.csv_handler.update_model_status(model_id, "Done")
                            if self.terminal_ui:
                                self.terminal_ui.add_log(f"모델 {model_id}의 학습이 완료되었습니다.", "success")
                            
                            # WeightFile을 WandB run 이름으로 설정
                            # (WeightFile은 실제 가중치 파일이 아닌, 가중치 파일이 담긴 폴더를 의미)
                            
                            # 먼저 로그에서 감지된 run_name을 확인
                            run_name = self.process_manager.get_run_name(model_id)
                            
                            if run_name:
                                # 로그에서 직접 찾은 run_name 사용
                                self.csv_handler.update_weight_file(model_id, run_name)
                                logger.info(f"모델 {model_id}의 WeightFile 폴더를 로그에서 찾은 run name으로 설정: {run_name}")
                                if self.terminal_ui:
                                    self.terminal_ui.add_log(f"모델 {model_id}의 WeightFile 폴더를 로그에서 찾은 run name으로 설정: {run_name}", "info")
                            else:
                                # 로그에서 찾지 못한 경우 API를 통해 가져오기 시도
                                run_name = self.wandb_monitor.get_run_name(run_id)
                                if run_name:
                                    # 찾은 run 이름(폴더 이름)을 저장
                                    self.csv_handler.update_weight_file(model_id, run_name)
                                    logger.info(f"모델 {model_id}의 WeightFile 폴더를 WandB API로 찾은 run 이름으로 설정: {run_name}")
                                    if self.terminal_ui:
                                        self.terminal_ui.add_log(f"모델 {model_id}의 WeightFile 폴더를 WandB API로 찾은 run 이름으로 설정: {run_name}", "info")
                            
                            # 알림 보내기
                            self.notification_manager.notify_training_completed(
                                model_id=model_id,
                                model_name=model.get('Name', ''),
                                completion_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                            
                            # 자동으로 다음 모델 학습 시작 (설정된 경우)
                            if self.auto_continue and len(self.csv_handler.get_untrained_models()) > 0:
                                logger.info("자동 연속 학습 모드: 다음 모델 학습을 준비합니다.")
                                if self.terminal_ui:
                                    self.terminal_ui.add_log("자동 연속 학습 모드: 다음 모델 학습을 준비합니다.", "info")
                    else:
                        logger.warning(f"모델 {model_id}의 WandB Run은 종료되지 않았지만 프로세스가 실행되고 있지 않습니다.")
                        if self.terminal_ui:
                            self.terminal_ui.add_log(f"모델 {model_id}의 WandB Run은 종료되지 않았지만 프로세스가 실행되고 있지 않습니다.", "warning")
                else:
                    # WandB 정보 없음
                    logger.warning(f"모델 {model_id}의 프로세스가 종료되었지만 WandB 정보가 없습니다.")
                    self.csv_handler.update_model_status(model_id, "Crash")  # WandB 정보가 없으면 크래시로 처리
                    if self.terminal_ui:
                        self.terminal_ui.add_log(f"모델 {model_id}의 프로세스가 종료되었지만 WandB 정보가 없습니다.", "error")
            else:
                # 프로세스가 실행 중인 경우 상태 업데이트
                # WandB RunID 가져오기 (없는 경우)
                if pd.isna(model.get('WandbRunID')) or model.get('WandbRunID') == '':
                    run_id = self.process_manager.get_wandb_run_id(model_id)
                    if run_id:
                        logger.info(f"모델 {model_id}의 WandB Run ID 업데이트: {run_id}")
                        self.csv_handler.update_value(model_id, 'WandbRunID', run_id)
                        if self.terminal_ui:
                            self.terminal_ui.add_log(f"모델 {model_id}의 WandB Run ID 업데이트: {run_id}", "info")
                
                # WeightFile 필드가 비어있으면 WandB run 이름으로 업데이트
                if (pd.isna(model.get('WeightFile')) or model.get('WeightFile') == '') and self.wandb_monitor:
                    # 먼저 로그에서 감지된 run_name을 확인
                    run_name = self.process_manager.get_run_name(model_id)
                    
                    if run_name:
                        # 로그에서 직접 찾은 run_name 사용
                        logger.info(f"모델 {model_id}의 WeightFile 필드를 로그에서 찾은 run name으로 업데이트: {run_name}")
                        self.csv_handler.update_value(model_id, 'WeightFile', run_name)
                        if self.terminal_ui:
                            self.terminal_ui.add_log(f"모델 {model_id}의 WeightFile 필드를 로그에서 찾은 run name으로 업데이트: {run_name}", "info")
                    else:
                        # 로그에서 찾지 못한 경우 기존 방식으로 WandB API 시도
                        run_id = model.get('WandbRunID') or self.process_manager.get_wandb_run_id(model_id)
                        if pd.isna(run_id) or run_id == '':
                            run_id = self.process_manager.get_wandb_run_id(model_id)
                        
                        if run_id:
                            run_name = self.wandb_monitor.get_run_name(run_id)
                            if run_name:
                                logger.info(f"모델 {model_id}의 WeightFile 필드를 WandB API로 찾은 run 이름으로 업데이트: {run_name}")
                                self.csv_handler.update_value(model_id, 'WeightFile', run_name)
                                if self.terminal_ui:
                                    self.terminal_ui.add_log(f"모델 {model_id}의 WeightFile 필드를 WandB API로 찾은 run 이름으로 업데이트: {run_name}", "info")
    
    def _find_pretrained_weight_file(self, pretrained_model_id: str) -> Optional[str]:
        """
        # Find weight file for pretrained model
        # pretrained_model_id: Model ID of the pretrained model
        # Returns: Path to weight file or None if not found
        """
        try:
            # Get model information from CSV
            all_models = self.csv_handler.get_all_models()
            
            # Find model by ID
            model = all_models[all_models['ID'] == pretrained_model_id]
            
            if model.empty:
                logger.warning(f"사전 학습 모델 ID를 찾을 수 없음: {pretrained_model_id}")
                return None
            
            # Get weight directory path
            weight_dir = model.iloc[0].get('WeightFile')
            if not weight_dir or str(weight_dir).strip() == '':
                logger.warning(f"사전 학습 모델에 가중치 폴더가 없음: {pretrained_model_id}")
                return None
            
            # Check if the directory exists
            full_weight_dir = weight_dir
            if not os.path.exists(full_weight_dir) or not os.path.isdir(full_weight_dir):
                # Try with base directory
                full_weight_dir = os.path.join(self.base_dir, weight_dir)
                if not os.path.exists(full_weight_dir) or not os.path.isdir(full_weight_dir):
                    logger.warning(f"가중치 폴더를 찾을 수 없음: {weight_dir}")
                    return None
            
            # Find the best weight file (lowest loss)
            weight_file = self._find_best_weight_file(full_weight_dir)
            if not weight_file:
                logger.warning(f"가중치 폴더 내에 적합한 가중치 파일을 찾을 수 없음: {full_weight_dir}")
                return None
            
            logger.info(f"사전 학습 모델의 가중치 파일을 찾음: {pretrained_model_id} -> {weight_file}")
            return weight_file
            
        except Exception as e:
            logger.error(f"사전 학습 모델 가중치 파일 검색 중 오류: {e}")
            return None
    
    def _find_best_weight_file(self, weight_dir: str) -> Optional[str]:
        """
        # Find the best weight file in the directory based on loss value
        # weight_dir: Directory containing weight files
        # Returns: Path to the best weight file or None if not found
        """
        try:
            # Find all .pth files in the directory
            weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pth')]
            if not weight_files:
                return None
            
            # Extract loss values from filenames with pattern 'model_{loss}_{idx}.pth'
            best_file = None
            lowest_loss = float('inf')
            
            import re
            pattern = re.compile(r'model_([0-9.]+)_([0-9]+)\.pth')
            
            for file in weight_files:
                match = pattern.match(file)
                if match:
                    try:
                        loss = float(match.group(1))
                        if loss < lowest_loss:
                            lowest_loss = loss
                            best_file = file
                    except (ValueError, IndexError):
                        continue
            
            if best_file:
                return os.path.join(weight_dir, best_file)
            
            # If no file matches the pattern, return the first file
            if weight_files:
                logger.warning(f"패턴에 맞는 가중치 파일이 없어 첫 번째 파일을 선택: {weight_files[0]}")
                return os.path.join(weight_dir, weight_files[0])
            
            return None
        
        except Exception as e:
            logger.error(f"최적 가중치 파일 검색 중 오류: {e}")
            return None
    
    def _start_new_trainings(self):
        """
        # Start new training processes
        """
        # 현재 실행 중인 프로세스 수 확인
        running_count = len([p for p in self.process_manager.get_all_processes().values() 
                            if p.get('status') == 'running'])
        
        # 최대 동시 실행 개수를 초과하면 시작하지 않음
        if running_count >= self.max_training_process:
            return
        
        # 학습되지 않은 모델 가져오기
        untrained_models = self.csv_handler.get_untrained_models()
        
        # 최대 동시 실행 개수까지 새 학습 시작
        for idx, model in untrained_models.iterrows():
            if running_count >= self.max_training_process:
                break
            
            model_id = model['ID']
            
            # 학습 명령어 가져오기
            training_command = model.get('TrainingCommand')
            if not training_command:
                logger.warning(f"모델 {model_id}에 학습 명령어가 없습니다.")
                continue
            
            # 사전 학습 모델 처리
            pretrained_args = {}
            if 'PretrainedModelId' in model and model['PretrainedModelId']:
                pretrained_model_id = str(model['PretrainedModelId']).strip()
                
                if pretrained_model_id != 'nan':
                    weight_file = self._find_pretrained_weight_file(pretrained_model_id)
                    
                    if weight_file:
                        # Add pretrained path to args
                        pretrained_args['pretrained_path'] = weight_file
                        if self.terminal_ui:
                            self.terminal_ui.add_log(
                                f"모델 {model_id}의 사전 학습 가중치 지정: {pretrained_model_id} -> {weight_file}", 
                                "info"
                            )
                    else:
                        # Mark as not found by setting pretrained_model_id to negative
                        try:
                            # Try to convert to int and set to negative
                            neg_id = -abs(int(pretrained_model_id))
                            self.csv_handler.update_value(model_id, 'PretrainedModelId', neg_id)
                            if self.terminal_ui:
                                self.terminal_ui.add_log(
                                    f"모델 {model_id}의 사전 학습 가중치를 찾을 수 없어 ID를 음수로 변경: {pretrained_model_id} -> {neg_id}", 
                                    "warning"
                                )
                        except ValueError:
                            # If not an int, just log the error
                            if self.terminal_ui:
                                self.terminal_ui.add_log(
                                    f"모델 {model_id}의 사전 학습 가중치를 찾을 수 없음: {pretrained_model_id}", 
                                    "warning"
                                )
            
            # 모델 상태를 'Training'으로 업데이트
            self.csv_handler.update_model_status(model_id, "Training")
            
            # GPU ID 가져오기 (CSV에 GpuID 열이 있는 경우 사용)
            # 프로세스 기반 할당이 활성화된 경우 자동 할당으로 전환
            gpu_id = None
            if not self.ini_config_handler.getboolean('gpu', 'use_process_order', True):
                gpu_id = model.get('GpuID', None)
                if gpu_id is not None and gpu_id == '':
                    gpu_id = None  # 빈 문자열인 경우 None으로 처리
            
            # 학습 프로세스 시작
            success, msg = self.process_manager.start_training_process(
                model_id, training_command, self.base_dir, gpu_id, training_args=pretrained_args
            )
            
            if success:
                logger.info(f"모델 {model_id}의 학습을 시작했습니다.")
                if self.terminal_ui:
                    # 할당된 GPU 정보 포함
                    process_info = self.process_manager.get_process_status(model_id)
                    gpu_info = f", GPU: {process_info.get('gpu_id', 'None')}" if 'gpu_id' in process_info else ""
                    process_idx = f", 프로세스 인덱스: {process_info.get('process_index', 'N/A')}" if 'process_index' in process_info else ""
                    self.terminal_ui.add_log(f"모델 {model_id}의 학습을 시작했습니다{gpu_info}{process_idx}", "info")
                
                running_count += 1
                
                # 알림 보내기
                self.notification_manager.notify_training_started(
                    model_id, model.get('Name', 'Unknown')
                )
            else:
                logger.error(f"모델 {model_id}의 학습 시작에 실패했습니다: {msg}")
                self.csv_handler.update_model_status(model_id, "Crash")
                if self.terminal_ui:
                    self.terminal_ui.add_log(f"모델 {model_id}의 학습 시작에 실패했습니다: {msg}", "error")
        if running_count == 0:
            logger.info("모든 학습이 완료되었습니다.")
            self.stop()
    
    def stop_training(self, model_id: str) -> bool:
        """
        # Stop a running training process
        """
        # 프로세스 중지
        if self.process_manager.stop_training_process(model_id):
            logger.info(f"모델 {model_id}의 학습을 중지했습니다.")
            # 상태 업데이트
            self.csv_handler.update_model_status(model_id, "Crash")  # 직접 멈춘 경우 Crash로 표시
            if self.terminal_ui:
                self.terminal_ui.add_log(f"모델 {model_id}의 학습을 중지했습니다.", "warning")
            return True
        else:
            logger.warning(f"모델 {model_id}의 학습 중지에 실패했습니다.")
            if self.terminal_ui:
                self.terminal_ui.add_log(f"모델 {model_id}의 학습 중지에 실패했습니다.", "error")
            return False
    
    def get_models_status(self) -> Dict[str, Any]:
        """
        # Get status of all models with process information
        """
        # CSV에서 모든 모델 정보 가져오기
        all_models = self.csv_handler.get_all_models().to_dict('records')
        
        # 프로세스 정보 추가
        process_info = self.process_manager.get_all_processes()
        
        result = {}
        for model in all_models:
            model_id = model['ID']
            
            # 모델 기본 정보
            result[model_id] = model
            
            # 프로세스 정보 추가
            if model_id in process_info:
                process_data = process_info[model_id]
                # 실행 시간 계산
                if process_data.get('start_time'):
                    runtime = time.time() - process_data['start_time']
                    result[model_id]['runtime'] = runtime
                
                # WandB Run ID 추가
                run_id = process_data.get('run_id')
                if run_id:
                    result[model_id]['run_id'] = run_id
                    result[model_id]['WandbRunID'] = run_id
                
                # GPU ID 추가
                gpu_id = process_data.get('gpu_id')
                if gpu_id:
                    result[model_id]['gpu_id'] = gpu_id
                    result[model_id]['GpuID'] = gpu_id
                
                # 프로세스 인덱스 추가
                process_index = process_data.get('process_index')
                if process_index is not None:
                    result[model_id]['process_index'] = process_index
                
                # 기타 프로세스 정보 추가
                for key, value in process_data.items():
                    if key not in result[model_id]:
                        result[model_id][key] = value
            
            # WandB 정보 추가 (프로세스에서 얻지 못한 경우)
            if self.wandb_monitor and 'WandbRunID' in model and model['WandbRunID']:
                run_id = model['WandbRunID']
                try:
                    run_status = self.wandb_monitor.get_run_status(run_id)
                    for key, value in run_status.items():
                        if key not in result[model_id]:
                            result[model_id][key] = value
                except Exception as e:
                    logger.error(f"WandB 정보 가져오기 중 오류 발생: {e}")
        
        return result
    
    def get_summary_status(self) -> Dict[str, Any]:
        """
        # Get summary status for simple UI
        """
        try:
            # 통계 정보 계산
            waiting = len(self.csv_handler.get_untrained_models())
            training = len(self.csv_handler.get_models_in_training())
            done = len(self.csv_handler.get_trained_models())
            crashed = len(self.csv_handler.get_crashed_models())
            
            # 현재 학습 중인 모델 정보
            models_in_training = self.csv_handler.get_models_in_training().to_dict('records')
            process_info = self.process_manager.get_all_processes()
            
            current = {}
            for model in models_in_training:
                model_id = model['ID']
                current[model_id] = {
                    'name': model.get('Name', 'Unknown'),
                    'status': 'Training',
                }
                
                if model_id in process_info:
                    # 실행 시간 계산
                    if 'start_time' in process_info[model_id]:
                        runtime = time.time() - process_info[model_id]['start_time']
                        current[model_id]['runtime'] = runtime
                    
                    # GPU ID 추가
                    if 'gpu_id' in process_info[model_id]:
                        current[model_id]['gpu_id'] = process_info[model_id]['gpu_id']
                    
                    # 프로세스 인덱스 추가
                    if 'process_index' in process_info[model_id]:
                        current[model_id]['process_index'] = process_info[model_id]['process_index']
            
            return {
                'waiting': waiting,
                'training': training,
                'done': done,
                'crashed': crashed,
                'current': current,
                'max_training_process': self.max_training_process,
                'auto_continue': self.auto_continue
            }
        except Exception as e:
            logger.error(f"상태 요약 가져오기 중 오류 발생: {e}")
            return {
                'waiting': 0,
                'training': 0,
                'done': 0,
                'crashed': 0,
                'error': str(e),
                'max_training_process': self.max_training_process,
                'auto_continue': self.auto_continue
            }
    
    def _extract_weight_file_from_output_dir(self, output_dir: str) -> Optional[str]:
        """
        # Extract weight file from output directory
        """
        if not output_dir or not os.path.exists(output_dir):
            return None
        
        # .pth 파일 찾기
        pth_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
        
        if not pth_files:
            return None
        
        # 파일 중 가장 최근의 것 또는 loss 값이 작은 것 선택
        try:
            # 'EPOCH_XXX_LOSS_XXX.pth' 패턴에서 loss 값을 추출
            loss_pattern = re.compile(r'EPOCH_\d+_LOSS_([\d\.]+)\.pth')
            
            loss_files = []
            for pth_file in pth_files:
                match = loss_pattern.search(pth_file)
                if match:
                    loss = float(match.group(1))
                    loss_files.append((pth_file, loss))
            
            if loss_files:
                # loss 값이 가장 작은 파일 선택
                best_file = min(loss_files, key=lambda x: x[1])[0]
                return os.path.join(output_dir, best_file)
            else:
                # loss 패턴이 없으면 가장 최근 파일 선택
                latest_file = max(pth_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
                return os.path.join(output_dir, latest_file)
        except Exception as e:
            logger.error(f"가중치 파일 추출 중 오류 발생: {e}")
            if pth_files:
                # 오류가 발생하면 첫 번째 파일 반환
                return os.path.join(output_dir, pth_files[0])
            return None
    
    def create_default_config(self, config_file: str) -> bool:
        """
        # Create a default configuration file if it doesn't exist
        """
        return self.ini_config_handler.create_default_config(config_file)
    
    def show_process_log(self, model_id: str) -> bool:
        """
        # Show log of a running process in a new terminal window
        # model_id: ID of the model to monitor
        # Returns: True if successful, False otherwise
        # Shows both stdout and stderr logs in the same terminal
        """
        # 모델 ID 유효성 검사
        if not model_id:
            if self.terminal_ui:
                self.terminal_ui.add_log("유효한 모델 ID를 입력하세요.", "error")
            logger.error("유효한 모델 ID를 입력하세요.")
            return False
        
        # 모델 존재 여부 확인
        model = self.csv_handler.get_model_by_id(model_id)
        if model is None:
            if self.terminal_ui:
                self.terminal_ui.add_log(f"모델 ID '{model_id}'를 찾을 수 없습니다.", "error")
            logger.error(f"모델 ID '{model_id}'를 찾을 수 없습니다.")
            return False
        
        # 프로세스가 실행 중인지 확인
        if not self.process_manager.is_process_running(model_id):
            msg = f"모델 {model_id}는 현재 실행 중이 아닙니다."
            if self.terminal_ui:
                self.terminal_ui.add_log(msg, "warning")
            logger.warning(msg)
            return False
        
        # 통합 로그 모니터링 실행 (stdout과 stderr 모두 표시)
        result = self.process_manager.show_combined_logs(model_id)
        if result:
            msg = f"모델 {model_id}의 로그 모니터링이 시작되었습니다."
            if self.terminal_ui:
                self.terminal_ui.add_log(msg, "info")
            logger.info(msg)
        return result
    
    def show_all_process_logs(self) -> None:
        """
        # Show logs of all running processes in new terminal windows
        """
        self.process_manager.show_all_process_logs()
        if self.terminal_ui:
            self.terminal_ui.add_log("모든 실행 중인 프로세스의 로그 모니터링이 시작되었습니다.", "info")
        logger.info("모든 실행 중인 프로세스의 로그 모니터링이 시작되었습니다.")
    
    def _print_training_logs(self):
        """
        # Open separate terminal windows for each training process
        # Each window has the model ID in the title
        # Windows remain open even after training completes
        # Shows both stdout and stderr logs in the same terminal
        """
        try:
            # 학습 중으로 표시된 모델 가져오기
            models_in_training = self.csv_handler.get_models_in_training()
            
            for _, model in models_in_training.iterrows():
                model_id = model['ID']
                
                # 프로세스 상태 정보 가져오기
                process_info = self.process_manager.get_process_status(model_id)
                
                # 이미 로그 터미널이 열려 있는지 확인
                if process_info and "log_terminal_opened" not in process_info:
                    # 통합 로그 모니터링 실행 (stdout과 stderr 모두 표시)
                    stdout_exists = "stdout_log" in process_info and os.path.exists(process_info["stdout_log"])
                    stderr_exists = "stderr_log" in process_info and os.path.exists(process_info["stderr_log"])
                    
                    if stdout_exists and stderr_exists:
                        # 로그 파일이 모두 존재하는 경우에만 로그 터미널 열기
                        result = self.process_manager.show_combined_logs(model_id)
                        
                        # 로그 터미널이 열렸음을 표시
                        if result:
                            with self.process_manager.lock:
                                if model_id in self.process_manager.processes:
                                    self.process_manager.processes[model_id]["log_terminal_opened"] = True
                                    logger.info(f"모델 {model_id}의 통합 로그 터미널이 열렸습니다.")
                    else:
                        logger.warning(f"모델 {model_id}의 로그 파일이 아직 준비되지 않았습니다. stdout: {stdout_exists}, stderr: {stderr_exists}")
            
        except Exception as e:
            logger.error(f"로그 터미널 생성 중 오류 발생: {e}")

def main():
    """
    # Main entry point
    """
    parser = argparse.ArgumentParser(description='ML Training Manager')
    parser.add_argument('--csv', type=str, default=None, help='Path to ML_Experiment_Table.csv (overrides config file)')
    parser.add_argument('--config', type=str, default='manager_config.ini', help='Path to configuration file (INI format)')
    parser.add_argument('--training_file_path', type=str, default=None, help='Path to folder containing training files')
    parser.add_argument('--create_config', type=str, default=None, help='Create a default configuration file at the specified path')
    parser.add_argument('--check_interval', type=int, default=None, help='Interval (in seconds) to check training status')
    parser.add_argument('--max_training_process', type=int, default=None, help='Maximum number of concurrent training processes')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team name)')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    parser.add_argument('--no_ui', action='store_true', default=False, help='Disable terminal UI')
    parser.add_argument('--auto_continue', action='store_true', default=False, help='Automatically continue to next model after one completes')
    parser.add_argument('--show_logs', action='store_true', default=False, help='Show logs of running processes in separate terminal windows')
    parser.add_argument('--show_log', type=str, default=None, help='Show log of a specific model ID in a separate terminal window')
    parser.add_argument('--no_auto_log_terminal', action='store_true', default=False, help='Disable automatic terminal windows for logs')
    
    args = parser.parse_args()
    
    # 자동 로그 터미널 설정
    args.auto_log_terminal = not args.no_auto_log_terminal
    
    # 기본 설정 파일 생성 (요청된 경우)
    if args.create_config:
        ini_config_handler = ConfigHandler()
        if ini_config_handler.create_default_config(args.create_config):
            logger.info(f"기본 설정 파일 생성 완료: {args.create_config}")
        else:
            logger.error(f"기본 설정 파일 생성 실패: {args.create_config}")
        return
    
    try:
        # 현재 디렉토리 경로 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.current_dir = current_dir
        args.config = os.path.join(current_dir, args.config)

        # 관리자 인스턴스 생성
        manager = MLTrainingManager(args)
        
        # 명령행 인자에서 제공된 경우 최대 동시 실행 개수 설정 (설정 파일보다 우선)
        if args.max_training_process:
            manager.max_training_process = args.max_training_process
        
        # 로그 표시 옵션 처리
        if args.show_log:
            if manager.show_process_log(args.show_log):
                return  # 로그 창만 표시하고 종료
            else:
                logger.error(f"모델 {args.show_log}의 로그를 표시할 수 없습니다.")
                return
        
        # 시그널 핸들러 설정 (Ctrl+C 등으로 안전하게 종료)
        def signal_handler(sig, frame):
            logger.info("종료 신호를 받았습니다. 정리 중...")
            manager.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 관리자 시작
        manager.start()
        
        # 모든 실행 중인 프로세스의 로그 표시 (요청된 경우)
        if args.show_logs:
            manager.show_all_process_logs()
        
        # 메인 스레드는 여기서 대기
        while True:
            time.sleep(1)
            
    except ValueError as e:
        logger.error(f"오류: {e}")
        print(f"오류: {e}")
        parser.print_help()
        sys.exit(1)
    except KeyboardInterrupt:
        # Ctrl+C로 종료
        if 'manager' in locals():
            manager.stop()

if __name__ == "__main__":
    main()
