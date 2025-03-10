import os
import subprocess
import signal
import time
import logging
import psutil
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
import threading
import shlex

logger = logging.getLogger(__name__)

class ProcessManager:
    """
    # Manage training processes
    """
    def __init__(self, config_handler=None):
        """
        # Initialize process manager
        # config_handler: ConfigHandler instance for configuration
        """
        self.processes = {}  # model_id -> process info
        self.lock = threading.Lock()
        self.config_handler = config_handler
        self.process_index_counter = 0  # Counter for assigning process indices
        logger.info("프로세스 매니저 초기화 완료")
    
    def start_training_process(self, model_id: str, command: str, cwd: Optional[str] = None,
                              gpu_id: Optional[Union[str, List[str]]] = None, 
                              env_setup: Optional[Dict[str, Any]] = None,
                              training_args: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        # Start a new training process
        # model_id: Unique identifier for the model
        # command: Training command to execute
        # cwd: Working directory for the process
        # gpu_id: GPU ID or list of GPU IDs to use (overrides automatic assignment)
        # env_setup: Environment setup configuration (overrides config_handler)
        # training_args: Additional training arguments to append to command
        # Returns: (success, run_id or error message)
        """
        with self.lock:
            if model_id in self.processes and self.is_process_running(model_id):
                error_msg = f"모델 {model_id}는 이미 학습 중입니다."
                logger.warning(error_msg)
                return False, error_msg
            
            try:
                # Assign process index
                process_index = self.process_index_counter
                self.process_index_counter += 1
                
                # Get GPU assignment if not provided
                if gpu_id is None and self.config_handler:
                    # Use process index for GPU assignment
                    gpu_id = self.config_handler.assign_gpu_to_process_index(process_index)
                
                # Get environment setup if not provided
                if env_setup is None and self.config_handler:
                    env_setup = self.config_handler.get_environment_setup()
                else:
                    env_setup = env_setup or {}
                
                # Prepare environment variables
                env = os.environ.copy()
                
                # Set CUDA_VISIBLE_DEVICES if GPU assignment is enabled
                if gpu_id:
                    if isinstance(gpu_id, list):
                        # Join multiple GPU IDs with comma
                        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_id)
                        logger.info(f"모델 {model_id}에 다중 GPU 할당: {env['CUDA_VISIBLE_DEVICES']} (프로세스 인덱스: {process_index})")
                    else:
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                        logger.info(f"모델 {model_id}에 GPU {gpu_id} 할당 (프로세스 인덱스: {process_index})")
                
                # Add custom environment variables
                if 'env_vars' in env_setup and env_setup['env_vars']:
                    for key, value in env_setup['env_vars'].items():
                        env[key] = value
                        logger.debug(f"환경 변수 설정: {key}={value}")
                
                # Modify command if training_args is provided
                if training_args:
                    for key, value in training_args.items():
                        if value is not None:
                            # Add or replace argument in command
                            if isinstance(value, bool) and value:
                                # For boolean flags, just add the flag
                                if f"--{key}" not in command:
                                    command += f" --{key}"
                            elif not isinstance(value, bool):
                                # For value args, add key=value
                                arg_pattern = f"--{key}[= ]\\S+"
                                import re
                                if re.search(arg_pattern, command):
                                    # Replace existing arg
                                    command = re.sub(arg_pattern, f"--{key}={value}", command)
                                else:
                                    # Add new arg
                                    command += f" --{key}={value}"
                
                # Prepare full command with environment setup
                full_command = self._prepare_command(command, env_setup)
                
                # 로그 디렉토리 생성
                log_dir = os.path.join(os.getcwd(), "logs")
                os.makedirs(log_dir, exist_ok=True)
                
                # 로그 파일 경로 설정
                stdout_log = os.path.join(log_dir, f"{model_id}_stdout.log")
                stderr_log = os.path.join(log_dir, f"{model_id}_stderr.log")

                # 이전 로그가 있다면 삭제
                if os.path.exists(stdout_log):
                    os.remove(stdout_log)
                if os.path.exists(stderr_log):
                    os.remove(stderr_log)
                
                # 로그 파일 초기화
                with open(stdout_log, 'w') as f_out, open(stderr_log, 'w') as f_err:
                    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    header = f"===== 프로세스 시작: {start_time} =====\n" \
                           f"모델 ID: {model_id}\n" \
                           f"명령어: {command}\n" \
                           f"GPU: {gpu_id}\n" \
                           f"========================================\n\n"
                    f_out.write(header)
                    f_err.write(header)
                
                # Start the process
                process = subprocess.Popen(
                    full_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                    shell=True  # Use shell for complex command with environment setup
                )
                
                # Create separate threads to read stdout and stderr
                stdout_thread = threading.Thread(
                    target=self._read_output_stream,
                    args=(process.stdout, stdout_log, model_id, "stdout")
                )
                stderr_thread = threading.Thread(
                    target=self._read_output_stream,
                    args=(process.stderr, stderr_log, model_id, "stderr")
                )
                
                stdout_thread.daemon = True
                stderr_thread.daemon = True
                stdout_thread.start()
                stderr_thread.start()
                
                # Store process info
                self.processes[model_id] = {
                    "process": process,
                    "pid": process.pid,
                    "command": command,
                    "full_command": full_command,
                    "start_time": time.time(),
                    "stdout_thread": stdout_thread,
                    "stderr_thread": stderr_thread,
                    "stdout_log": stdout_log,  # 로그 파일 경로 저장
                    "stderr_log": stderr_log,  # 로그 파일 경로 저장
                    "run_id": None,  # To be set later when obtained from WandB
                    "gpu_id": gpu_id,
                    "process_index": process_index,  # Store process index
                    "training_args": training_args,   # Store any additional training args
                    "log_terminal_opened": False,  # Flag to track if a log terminal has been opened
                    "run_name": None  # Added for run_name
                }
                
                # Log GPU allocation info
                if isinstance(gpu_id, list):
                    gpu_info = f"GPUs: {','.join(str(g) for g in gpu_id)}"
                else:
                    gpu_info = f"GPU: {gpu_id or 'None'}"
                
                logger.info(f"모델 {model_id}의 학습 프로세스 시작 (PID: {process.pid}, {gpu_info}, 프로세스 인덱스: {process_index})")
                
                # Wait briefly to check for immediate failure
                time.sleep(2)
                if process.poll() is not None:
                    return_code = process.poll()
                    error_msg = f"프로세스가 즉시 종료됨 (반환 코드: {return_code})"
                    logger.error(error_msg)
                    self.processes[model_id]["status"] = "error"
                    stdout, stderr = process.communicate()
                    print(f"표준 출력:\n{stdout}")
                    print(f"에러 출력:\n{stderr}")
                    return False, error_msg
                
                self.processes[model_id]["status"] = "running"
                return True, ""
                
            except Exception as e:
                error_msg = f"학습 프로세스 시작 중 오류 발생: {e}"
                logger.error(error_msg)
                return False, error_msg
    
    def _prepare_command(self, command: str, env_setup: Dict[str, Any]) -> str:
        """
        # Prepare full command with environment setup
        # command: Original training command
        # env_setup: Environment setup configuration
        # Returns: Full command string
        """
        prefixes = []
        
        # Add setup script if provided
        if 'setup_script' in env_setup and env_setup['setup_script']:
            # If it's a file that exists, source it
            if os.path.exists(env_setup['setup_script']):
                prefixes.append(f"source {env_setup['setup_script']}")
            else:
                # Otherwise assume it's a direct script command
                prefixes.append(env_setup['setup_script'])
        
        # Add conda environment activation if needed
        if 'use_conda' in env_setup and env_setup['use_conda'] and 'conda_env' in env_setup and env_setup['conda_env']:
            # Support for both conda and mamba
            # conda_cmd = "mamba" if shutil.which("mamba") else "conda" # 미사용
            conda_cmd = "conda"
            prefixes.append(f"{conda_cmd} activate {env_setup['conda_env']}")
        
        # Combine prefixes with the main command
        if prefixes:
            # Join all commands with && to ensure they run in sequence
            full_command = " && ".join(prefixes + [command])
            
            # Wrap in bash -c for shell execution
            return f"bash -c '{full_command}'"
        else:
            return command
    
    def _read_output_stream(self, stream, log_file: str, model_id: str, stream_name: str):
        """
        # Read and log output from process streams
        # 문자 단위로 출력을 읽어 tqdm 진행 표시줄 등이 제대로 표시되도록 함
        """
        try:
            with open(log_file, 'a') as f:
                # 버퍼 관리 (WandB Run ID 감지용)
                line_buffer = ""
                
                # 문자 단위로 읽기
                while True:
                    char = stream.read(1)
                    
                    # EOF 검사
                    if not char:
                        break
                    
                    # 파일에 바로 쓰기
                    f.write(char)
                    f.flush()
                    
                    # 로그 내용 확인 
                    if stream_name == "stderr" and char not in ('\n', '\r', ' ', '\t'):
                        # stderr 로그가 생성되고 있음을 로그로 남김
                        if len(line_buffer) == 0:
                            logger.debug(f"모델 {model_id}의 stderr에 데이터가 기록되고 있습니다.")
                    
                    # 줄 버퍼 관리 (WandB Run ID와 Name 항목 감지용)
                    if char == '\n':
                        # 줄이 완료되면 WandB Run ID 검사
                        if "wandb" in line_buffer.lower() and "run-" in line_buffer:
                            try:
                                # Try to extract the run ID
                                parts = line_buffer.split("run-")
                                if len(parts) > 1:
                                    run_id_part = parts[1].split()[0].strip()
                                    if run_id_part:
                                        run_id = f"run-{run_id_part}"
                                        with self.lock:
                                            if model_id in self.processes:
                                                self.processes[model_id]["run_id"] = run_id
                                                logger.info(f"모델 {model_id}의 WandB Run ID 감지: {run_id}")
                            except Exception as e:
                                logger.error(f"WandB Run ID 추출 중 오류: {e}")
                        
                        # WandB Syncing run 패턴 검사 - 로그에서 실제 발견된 형식
                        if "wandb:" in line_buffer and ("syncing run" in line_buffer.lower() or "Syncing run" in line_buffer):
                            try:
                                # 다양한 분할 패턴 시도
                                split_patterns = [
                                    "wandb: Syncing run ",
                                    "wandb: syncing run ",
                                    "wandb: Syncing run\t",
                                    "wandb: syncing run\t",
                                    "wandb:Syncing run ",
                                    "wandb:syncing run ",
                                    "wandb: Syncing run",
                                    "wandb: syncing run"
                                ]
                                
                                name_part = None
                                for pattern in split_patterns:
                                    if pattern in line_buffer:
                                        name_part = line_buffer.split(pattern, 1)[1]
                                        break
                                
                                # 패턴이 정확히 일치하지 않으면 공백을 기준으로 분리
                                if name_part is None and "wandb:" in line_buffer and "run" in line_buffer:
                                    parts = line_buffer.split()
                                    for i, part in enumerate(parts):
                                        if part.lower() == "run" and i < len(parts) - 1:
                                            name_part = parts[i+1]
                                            break
                                
                                if name_part:
                                    # 공백이나 쉼표 등으로 잘라내기
                                    run_name = name_part.split()[0].strip()
                                    
                                    # 빈 이름이거나 너무 길면 스킵
                                    if run_name and len(run_name) <= 100:
                                        with self.lock:
                                            if model_id in self.processes:
                                                self.processes[model_id]["run_name"] = run_name
                                                logger.info(f"모델 {model_id}의 Run Name 감지 (Syncing run 패턴): {run_name}")
                            except Exception as e:
                                logger.error(f"Syncing run 패턴에서 Run Name 추출 중 오류: {e}")
                                logger.debug(f"문제가 된 줄: '{line_buffer}'")
                        
                        # 버퍼 초기화
                        line_buffer = ""
                    else:
                        # 줄 버퍼에 문자 추가
                        line_buffer += char
                
                # 종료 시간 기록
                end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n===== 프로세스 스트림 종료: {end_time} =====\n")
                
        except Exception as e:
            logger.error(f"프로세스 출력 스트림 읽기 중 오류: {e}")
            if model_id in self.processes:
                with self.lock:
                    if "process" in self.processes[model_id]:
                        logger.info(f"종료 코드 확인: {self.processes[model_id]['process'].poll()}")
                        
        # 처리 완료 로그
        logger.debug(f"모델 {model_id}의 {stream_name} 스트림 읽기 완료")
    
    def stop_training_process(self, model_id: str) -> bool:
        """
        # Stop a running training process
        """
        with self.lock:
            if model_id not in self.processes:
                logger.warning(f"모델 {model_id}의 프로세스를 찾을 수 없습니다.")
                return False
            
            process_info = self.processes[model_id]
            process = process_info["process"]
            
            if process.poll() is None:  # Process is still running
                try:
                    # First try gentle termination
                    process.terminate()
                    
                    # Wait for process to terminate
                    for _ in range(10):  # Wait up to 10 seconds
                        if process.poll() is not None:
                            break
                        time.sleep(1)
                    
                    # If still running, force kill
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    
                    logger.info(f"모델 {model_id}의 학습 프로세스 중지 완료")
                    self.processes[model_id]["status"] = "stopped"
                    return True
                except Exception as e:
                    logger.error(f"프로세스 중지 중 오류 발생: {e}")
                    return False
            else:
                logger.info(f"모델 {model_id}의 프로세스는 이미 종료되었습니다.")
                self.processes[model_id]["status"] = "completed"
                return True
    
    def get_process_status(self, model_id: str) -> Dict[str, Any]:
        """
        # Get status of a specific process
        """
        with self.lock:
            if model_id not in self.processes:
                return {"status": "not_found"}
            
            process_info = self.processes[model_id].copy()
            process = process_info["process"]
            
            # Remove process objects that can't be serialized
            if "process" in process_info:
                del process_info["process"]
            if "stdout_thread" in process_info:
                del process_info["stdout_thread"]
            if "stderr_thread" in process_info:
                del process_info["stderr_thread"]
            
            # Add current status
            if process.poll() is None:
                process_info["status"] = "running"
                process_info["runtime"] = time.time() - process_info["start_time"]
            else:
                process_info["status"] = "completed" if process.returncode == 0 else "error"
                process_info["return_code"] = process.returncode
            
            return process_info
    
    def is_process_running(self, model_id: str) -> bool:
        """
        # Check if a process is still running
        """
        with self.lock:
            if model_id not in self.processes:
                return False
            
            process = self.processes[model_id]["process"]
            return process.poll() is None
    
    def get_wandb_run_id(self, model_id: str) -> Optional[str]:
        """
        # Get WandB run ID for a process if available
        """
        with self.lock:
            if model_id not in self.processes:
                return None
            if "run_id" not in self.processes[model_id]:
                return None
            return self.processes[model_id]["run_id"]
    
    def get_process_index(self, model_id: str) -> Optional[int]:
        """
        # Get process index for a specific model
        """
        with self.lock:
            if model_id not in self.processes:
                return None
            
            return self.processes[model_id].get("process_index")
    
    def get_all_processes(self) -> Dict[str, Dict[str, Any]]:
        """
        # Get status of all processes
        """
        result = {}
        with self.lock:
            for model_id, process_info in self.processes.items():
                status = self.get_process_status(model_id)
                result[model_id] = status
        
        return result
    
    def cleanup_old_processes(self):
        """
        # Clean up completed/failed processes
        """
        with self.lock:
            to_remove = []
            for model_id, process_info in self.processes.items():
                process = process_info["process"]
                if process.poll() is not None:  # Process has completed
                    # Wait for output threads to finish
                    if "stdout_thread" in process_info:
                        process_info["stdout_thread"].join(timeout=1)
                    if "stderr_thread" in process_info:
                        process_info["stderr_thread"].join(timeout=1)
                    to_remove.append(model_id)
            
            for model_id in to_remove:
                logger.info(f"모델 {model_id}의 프로세스 정보 정리")
                del self.processes[model_id]
    
    def reset_process_index_counter(self):
        """
        # Reset the process index counter to 0
        """
        with self.lock:
            self.process_index_counter = 0
            logger.info("프로세스 인덱스 카운터 리셋됨")
    
    def show_process_log(self, model_id: str, stream_type: str = "stdout") -> bool:
        """
        # Open a new terminal window to monitor the log of a running process
        # model_id: ID of the model to monitor
        # stream_type: Type of stream to monitor ("stdout" or "stderr")
        # Returns: True if successful, False otherwise
        # 
        # 참고: 단일 스트림만 표시하는 기존 메서드입니다.
        # 두 스트림을 모두 표시하려면 show_combined_logs 메서드를 사용하세요.
        """
        with self.lock:
            if model_id not in self.processes:
                logger.error(f"모델 {model_id}를 찾을 수 없습니다.")
                return False
            
            if stream_type not in ["stdout", "stderr"]:
                logger.error(f"유효하지 않은 스트림 유형: {stream_type}. 'stdout' 또는 'stderr'를 사용하세요.")
                return False
            
            # 로그 파일 경로 가져오기
            log_key = f"{stream_type}_log"
            if log_key not in self.processes[model_id]:
                logger.error(f"모델 {model_id}에 대한 {stream_type} 로그 파일 정보가 없습니다.")
                return False
            
            log_file = self.processes[model_id][log_key]
            
            try:
                import platform
                if platform.system() == "Linux":
                    monitor_command = f"gnome-terminal --title=\"모델 {model_id} {stream_type} 로그\" -- bash -c 'echo \"출력 로그 모니터링 중... ({model_id})\" && tail -f {log_file}; read -p \"Enter 키를 누르면 종료됩니다...\"'"
                    subprocess.Popen(monitor_command, shell=True)
                    logger.info(f"모델 {model_id}의 {stream_type} 로그 모니터링 터미널이 열렸습니다.")
                elif platform.system() == "Windows":
                    monitor_command = f"start \"모델 {model_id} {stream_type} 로그\" cmd /k \"echo 출력 로그 모니터링 중... ({model_id}) && tail -f {log_file}\""
                    subprocess.Popen(monitor_command, shell=True)
                else:
                    logger.warning(f"현재 플랫폼({platform.system()})에서는 별도 터미널 모니터링을 지원하지 않습니다.")
                    return False
                
                return True
            except Exception as e:
                logger.error(f"로그 모니터링 터미널을 열지 못했습니다: {e}")
                return False
    
    def show_combined_logs(self, model_id: str) -> bool:
        """
        # Open a new terminal window to monitor both stdout and stderr logs of a process
        # model_id: ID of the model to monitor
        # Returns: True if successful, False otherwise
        """
        with self.lock:
            if model_id not in self.processes:
                logger.error(f"모델 {model_id}를 찾을 수 없습니다.")
                return False
            
            # 로그 파일 경로 가져오기
            if "stdout_log" not in self.processes[model_id] or "stderr_log" not in self.processes[model_id]:
                logger.error(f"모델 {model_id}에 대한 로그 파일 정보가 없습니다.")
                return False
            
            stdout_log = self.processes[model_id]["stdout_log"]
            stderr_log = self.processes[model_id]["stderr_log"]
            
            # 명령어 및 GPU 정보 가져오기
            command = self.processes[model_id].get("command", "Unknown command")
            gpu_id = self.processes[model_id].get("gpu_id", "Unknown GPU")
            if isinstance(gpu_id, list):
                gpu_info = ",".join(str(g) for g in gpu_id)
            else:
                gpu_info = str(gpu_id)
            
            try:
                import platform
                if platform.system() == "Linux":
                    # 로그 표시를 위한 터미널 창 열기
                    terminal_title = f"모델 {model_id} 로그"
                    monitor_command = (
                        f"gnome-terminal --title=\"{terminal_title}\" -- bash -c '"
                        f"echo \"===========================\"; "
                        f"echo \"모델 ID: {model_id}\"; "
                        f"echo \"명령어: {command}\"; "
                        f"echo \"GPU: {gpu_info}\"; "
                        f"echo \"로그 파일: {stdout_log}, {stderr_log}\"; "
                        f"echo \"===========================\"; "
                        f"echo \"\"; "
                        
                        # stdout과 stderr 모두 표시 (grep 필터링 제거)
                        f"echo \"stdout 출력:\" && "
                        f"tail -n 5 {stdout_log} && "
                        f"echo \"\"; "
                        f"echo \"stderr 출력:\" && "
                        f"tail -n 5 {stderr_log} && "
                        f"echo \"\"; "
                        f"echo \"실시간 stdout과 stderr 모니터링 중...\"; "
                        f"echo \"\"; "
                        
                        # 두 파일 동시에 테일링 (grep 필터 제거)
                        f"tail -f {stdout_log} {stderr_log} || "
                        # 백업 방법으로 두 개의 tail 명령어를 백그라운드와 포그라운드로 실행
                        f"(tail -f {stdout_log} & tail -f {stderr_log}); "
                        
                        f"echo \"\"; "
                        f"echo \"프로세스가 종료되었습니다. 이 창은 수동으로 닫을 때까지 유지됩니다.\"; "
                        f"echo \"종료하려면 이 창을 닫으세요.\"; "
                        f"read -r -d \"\" "  # 무한정 대기 (창이 닫히지 않도록)
                        f"'"
                    )
                    subprocess.Popen(monitor_command, shell=True)
                    logger.info(f"모델 {model_id}의 통합 로그 모니터링 터미널이 열렸습니다.")
                    return True
                    
                elif platform.system() == "Windows":
                    # Windows용 명령 (명령 프롬프트가 자동으로 닫히지 않음)
                    terminal_title = f"모델 {model_id} 로그"
                    monitor_command = (
                        f"start \"{terminal_title}\" cmd /k \"" 
                        f"echo =========================== && "
                        f"echo 모델 ID: {model_id} && "
                        f"echo 명령어: {command} && "
                        f"echo GPU: {gpu_info} && "
                        f"echo 로그 파일: {stdout_log}, {stderr_log} && "
                        f"echo =========================== && "
                        f"echo. && "
                        
                        # Windows에서 stdout과 stderr을 개별적으로 먼저 보여주기
                        f"echo 현재 stdout 내용: && "
                        f"type {stdout_log} && echo. && "
                        f"echo 현재 stderr 내용: && "
                        f"type {stderr_log} && echo. && "
                        
                        f"echo 실시간 로그 모니터링 중... && "
                        
                        # Windows에서 두 파일을 더 확실하게 모니터링 (PowerShell 활용)
                        f"powershell -Command \"$stdoutWatcher = New-Object System.IO.FileSystemWatcher; "
                        f"$stdoutWatcher.Path = [System.IO.Path]::GetDirectoryName('{stdout_log}'); "
                        f"$stdoutWatcher.Filter = [System.IO.Path]::GetFileName('{stdout_log}'); "
                        f"$stdoutWatcher.EnableRaisingEvents = $true; "
                        f"$stderrWatcher = New-Object System.IO.FileSystemWatcher; "
                        f"$stderrWatcher.Path = [System.IO.Path]::GetDirectoryName('{stderr_log}'); "
                        f"$stderrWatcher.Filter = [System.IO.Path]::GetFileName('{stderr_log}'); "
                        f"$stderrWatcher.EnableRaisingEvents = $true; "
                        f"while($true) {{ "
                        f"  Get-Content -Path '{stdout_log}' -Tail 1 -Wait | ForEach-Object {{ Write-Host \\\"[STDOUT] $_\\\" }}; "
                        f"  Start-Sleep -Milliseconds 100; "
                        f"  Get-Content -Path '{stderr_log}' -Tail 1 -Wait | ForEach-Object {{ Write-Host \\\"[STDERR] $_\\\" -ForegroundColor Red }}; "
                        f"  Start-Sleep -Milliseconds 100; "
                        f"}}\" "
                        
                        # 실패한 경우 단순한 백업 명령어
                        f"|| (echo 실시간 모니터링 실패, 단순 파일 모니터링으로 전환합니다... && "
                        f"powershell -Command \"while($true) {{ Get-Content -Path '{stdout_log}','{stderr_log}' -Tail 1; Start-Sleep -Seconds 1 }}\")"
                        f"\""
                    )
                    subprocess.Popen(monitor_command, shell=True)
                    logger.info(f"모델 {model_id}의 통합 로그 모니터링 터미널이 열렸습니다.")
                    return True
                    
                else:
                    logger.warning(f"현재 플랫폼({platform.system()})에서는 별도 터미널 모니터링을 지원하지 않습니다.")
                    return False
                
            except Exception as e:
                logger.error(f"로그 모니터링 터미널을 열지 못했습니다: {e}")
                return False
    
    def show_all_process_logs(self) -> None:
        """
        # Open terminal windows to monitor all running processes
        # Uses combined logs (stdout + stderr)
        """
        with self.lock:
            running_models = [model_id for model_id, info in self.processes.items() 
                            if self.is_process_running(model_id)]
            
            if not running_models:
                logger.info("현재 실행 중인 프로세스가 없습니다.")
                return
            
            for model_id in running_models:
                # 통합 로그 모니터링
                self.show_combined_logs(model_id)
    
    def get_run_name(self, model_id: str) -> Optional[str]:
        """
        # Get run name for a model
        # model_id: ID of the model
        # Returns: Run name or None if not found
        """
        with self.lock:
            if model_id not in self.processes:
                return None
            
            # 프로세스 정보에서 run_name 가져오기
            return self.processes[model_id].get("run_name") 