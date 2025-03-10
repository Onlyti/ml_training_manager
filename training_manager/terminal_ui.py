import os
import sys
import time
import threading
import logging
import curses
from typing import Dict, List, Any, Optional, Callable
from collections import deque

logger = logging.getLogger(__name__)

class TerminalUI:
    """
    # Terminal UI for training manager
    """
    def __init__(self, 
                 get_models_callback: Callable[[], Dict[str, Any]],
                 stop_training_callback: Callable[[str], bool],
                 exit_callback: Callable[[], None],
                 show_log_callback: Callable[[str], bool] = None,
                 show_all_logs_callback: Callable[[], None] = None):
        """
        # Initialize the terminal UI
        # get_models_callback: Function to get model list
        # stop_training_callback: Function to stop training for a model
        # exit_callback: Function to call on exit
        # show_log_callback: Function to show log for a specific model
        # show_all_logs_callback: Function to show logs for all running models
        """
        self.get_models_callback = get_models_callback
        self.stop_training_callback = stop_training_callback
        self.exit_callback = exit_callback
        self.show_log_callback = show_log_callback
        self.show_all_logs_callback = show_all_logs_callback
        
        self.screen = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.logs = deque(maxlen=100)  # Store up to 100 log messages
        self.selected_index = 0  # Selected model index
        self.command_buffer = ""  # Current command being typed
        self.status_update_interval = 1.0  # Status update interval in seconds
        
        self.color_pairs = {
            "info": 1,
            "success": 2,
            "warning": 3,
            "error": 4,
            "highlight": 5
        }
        
        # Command history
        self.command_history = []
        self.max_history = 10
        
        # Status log
        self.status_log = []
        self.max_log_entries = 100
        
        logger.info("터미널 UI 초기화 완료")
    
    def start(self):
        """
        # Start the terminal UI in a separate thread
        """
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_ui)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("터미널 UI 시작됨")
    
    def stop(self):
        """
        # Stop the terminal UI
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        logger.info("터미널 UI 중지됨")
    
    def _run_ui(self):
        """
        # Main UI thread function
        """
        try:
            # Initialize curses
            curses.wrapper(self._main_loop)
        except Exception as e:
            logger.error(f"터미널 UI 실행 중 오류 발생: {e}")
            self.running = False
    
    def _main_loop(self, stdscr):
        """
        # Main UI loop with curses
        """
        self.screen = stdscr
        
        # Set up colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)  # Success
        curses.init_pair(2, curses.COLOR_RED, -1)    # Error
        curses.init_pair(3, curses.COLOR_YELLOW, -1) # Warning
        curses.init_pair(4, curses.COLOR_BLUE, -1)   # Info
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected
        
        # Hide cursor
        curses.curs_set(0)
        
        # Enable key input
        self.screen.keypad(True)
        self.screen.timeout(int(self.status_update_interval * 1000))
        
        # Main loop
        while self.running:
            # Clear screen
            self.screen.clear()
            
            # Get terminal size
            height, width = self.screen.getmaxyx()
            
            # Draw UI
            self._draw_header(width)
            
            # Get model list
            model_list = self._get_formatted_model_list()
            
            # Calculate layout
            list_height = min(len(model_list), height - 10)
            log_height = min(10, height - list_height - 5)
            
            # Draw model list
            self._draw_model_list(model_list, 2, list_height)
            
            # Draw status log
            self._draw_status_log(list_height + 3, log_height, width)
            
            # Draw command line
            self._draw_command_line(height - 2, width)
            
            # Refresh the screen
            self.screen.refresh()
            
            # Handle key input
            self._handle_input()
    
    def _draw_header(self, width: int):
        """
        # Draw header section
        """
        title = "ML Training Manager"
        self.screen.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
        
        # Draw separator
        self.screen.addstr(1, 0, "=" * width)
    
    def _get_formatted_model_list(self) -> List[Dict[str, Any]]:
        """
        # Get and format the model list for display
        """
        try:
            models = self.get_models_callback()
            
            # Format for display
            formatted_list = []
            for model_id, model in models.items():
                # Create a display-friendly version of the model info
                display_info = {
                    "id": model_id,
                    "name": model.get("name", "Unknown"),
                    "status": model.get("TrainingCheck", ""),
                    "runtime": model.get("runtime", 0),
                    "pid": model.get("pid", ""),
                    "run_id": model.get("run_id", ""),
                    "progress": model.get("progress", 0.0),
                    "loss": model.get("loss", "N/A"),
                    "raw_data": model
                }
                
                formatted_list.append(display_info)
            
            return formatted_list
        except Exception as e:
            logger.error(f"모델 목록 가져오기 중 오류 발생: {e}")
            self.add_log(f"모델 목록 가져오기 중 오류 발생: {e}", "error")
            return []
    
    def _draw_model_list(self, model_list: List[Dict[str, Any]], start_y: int, height: int):
        """
        # Draw the model list section
        """
        if not model_list:
            self.screen.addstr(start_y, 2, "모델 목록이 비어 있습니다.")
            return
        
        # Header
        self.screen.addstr(start_y, 2, "ID", curses.A_BOLD)
        self.screen.addstr(start_y, 10, "이름", curses.A_BOLD)
        self.screen.addstr(start_y, 35, "상태", curses.A_BOLD)
        self.screen.addstr(start_y, 47, "실행 시간", curses.A_BOLD)
        self.screen.addstr(start_y, 60, "WandB Run", curses.A_BOLD)
        
        # Adjust selection index if needed
        if self.selected_index >= len(model_list):
            self.selected_index = len(model_list) - 1
        
        # Calculate start and end indices for scrolling
        total_models = len(model_list)
        visible_models = min(height - 1, total_models)
        
        if total_models <= visible_models:
            start_idx = 0
            end_idx = total_models
        else:
            # Center the selected item
            half_height = visible_models // 2
            start_idx = max(0, self.selected_index - half_height)
            end_idx = min(total_models, start_idx + visible_models)
            
            # Adjust if we're near the end
            if end_idx == total_models:
                start_idx = max(0, total_models - visible_models)
        
        # Draw visible items
        for i, idx in enumerate(range(start_idx, end_idx)):
            model = model_list[idx]
            row = start_y + i + 1
            
            # Determine text color based on status
            attr = curses.A_NORMAL
            if model["status"] == "Done":
                attr |= curses.color_pair(1)  # Success/Green
            elif model["status"] == "Crash":
                attr |= curses.color_pair(2)  # Error/Red
            elif model["status"] == "Training":
                attr |= curses.color_pair(3)  # Warning/Yellow
            
            # Highlight the selected row
            if idx == self.selected_index:
                self.screen.attron(curses.color_pair(5) | curses.A_BOLD)
            else:
                self.screen.attron(attr)
            
            # Clear line first
            _, width = self.screen.getmaxyx()
            self.screen.addstr(row, 1, " " * (width - 2))
            
            # Draw the model info
            self.screen.addstr(row, 2, f"{model['id'][:7]}")
            name = model["name"][:23] + "..." if len(model["name"]) > 25 else model["name"]
            self.screen.addstr(row, 10, f"{name}")
            self.screen.addstr(row, 35, f"{model['status']}")
            
            # Runtime formatting
            runtime = model.get("runtime", 0)
            if isinstance(runtime, (int, float)):
                hours, remainder = divmod(runtime, 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            else:
                runtime_str = "N/A"
            
            self.screen.addstr(row, 47, runtime_str)
            
            # WandB Run ID
            run_id = model.get("run_id", "")
            if run_id:
                run_id_short = run_id[:12] + "..." if len(run_id) > 15 else run_id
                self.screen.addstr(row, 60, run_id_short)
            
            # Reset attributes
            if idx == self.selected_index:
                self.screen.attroff(curses.color_pair(5) | curses.A_BOLD)
            else:
                self.screen.attroff(attr)
    
    def _draw_status_log(self, start_y: int, height: int, width: int):
        """
        # Draw the status log section
        """
        self.screen.addstr(start_y, 0, "-" * width)
        self.screen.addstr(start_y + 1, 2, "상태 로그", curses.A_BOLD)
        
        # Show the most recent log entries
        log_start = max(0, len(self.status_log) - height + 1)
        for i, idx in enumerate(range(log_start, len(self.status_log))):
            log_entry = self.status_log[idx]
            row = start_y + 2 + i
            
            # Set color based on log level
            attr = curses.A_NORMAL
            if log_entry["level"] == "error":
                attr |= curses.color_pair(2)
            elif log_entry["level"] == "warning":
                attr |= curses.color_pair(3)
            elif log_entry["level"] == "success":
                attr |= curses.color_pair(1)
            else:
                attr |= curses.color_pair(4)
            
            # Format timestamp
            timestamp = log_entry["timestamp"].strftime("%H:%M:%S")
            
            # Draw log entry
            self.screen.addstr(row, 2, f"[{timestamp}] ", attr)
            message = log_entry["message"]
            max_msg_len = width - 15
            if len(message) > max_msg_len:
                message = message[:max_msg_len - 3] + "..."
            self.screen.addstr(row, 13, message, attr)
    
    def _draw_command_line(self, y: int, width: int):
        """
        # Draw the command input line
        """
        self.screen.addstr(y - 1, 0, "-" * width)
        
        # Command help
        help_text = "q:종료 | s:중지 | r:새로고침 | l:로그보기 | a:모든로그 | ↑/↓:선택"
        self.screen.addstr(y - 1, 2, help_text, curses.A_BOLD)
        
        # Command prompt
        self.screen.addstr(y, 2, "> ")
    
    def _handle_input(self):
        """
        # Handle user input
        """
        try:
            key = self.screen.getch()
            
            if key == curses.ERR:  # No input (timeout)
                return
            
            if key == ord('q'):  # Quit
                self.add_log("프로그램 종료 요청", "info")
                self.running = False
                self.exit_callback()
            
            elif key == ord('r'):  # Refresh
                self.add_log("화면 새로고침", "info")
                # The screen will refresh on the next loop iteration
            
            elif key == ord('s'):  # Stop training
                model_list = self._get_formatted_model_list()
                if model_list and 0 <= self.selected_index < len(model_list):
                    model = model_list[self.selected_index]
                    self.add_log(f"모델 {model['id']} 학습 중지 요청", "warning")
                    success = self.stop_training_callback(model['id'])
                    if success:
                        self.add_log(f"모델 {model['id']} 학습 중지 성공", "success")
                    else:
                        self.add_log(f"모델 {model['id']} 학습 중지 실패", "error")
            
            elif key == ord('l'):  # Show log for selected model
                if self.show_log_callback:
                    model_list = self._get_formatted_model_list()
                    if model_list and 0 <= self.selected_index < len(model_list):
                        model = model_list[self.selected_index]
                        self.add_log(f"모델 {model['id']} 로그 창 열기 요청", "info")
                        success = self.show_log_callback(model['id'])
                        if success:
                            self.add_log(f"모델 {model['id']} 로그 창이 열렸습니다.", "success")
                        else:
                            self.add_log(f"모델 {model['id']} 로그 창을 열지 못했습니다.", "error")
                else:
                    self.add_log("로그 모니터링 기능이 활성화되지 않았습니다.", "warning")
            
            elif key == ord('a'):  # Show all logs
                if self.show_all_logs_callback:
                    self.add_log("모든 실행 중인 프로세스의 로그 창 열기 요청", "info")
                    self.show_all_logs_callback()
                    self.add_log("로그 창이 열렸습니다.", "success")
                else:
                    self.add_log("로그 모니터링 기능이 활성화되지 않았습니다.", "warning")
            
            elif key == curses.KEY_UP:  # Move selection up
                model_list = self._get_formatted_model_list()
                if model_list:
                    self.selected_index = max(0, self.selected_index - 1)
            
            elif key == curses.KEY_DOWN:  # Move selection down
                model_list = self._get_formatted_model_list()
                if model_list:
                    self.selected_index = min(len(model_list) - 1, self.selected_index + 1)
            
            # Add more commands as needed
            
        except Exception as e:
            logger.error(f"사용자 입력 처리 중 오류 발생: {e}")
            self.add_log(f"사용자 입력 처리 중 오류 발생: {e}", "error")
    
    def add_log(self, message: str, level: str = "info"):
        """
        # Add an entry to the status log
        # level: 'info', 'warning', 'error', 'success'
        """
        from datetime import datetime
        
        self.status_log.append({
            "timestamp": datetime.now(),
            "message": message,
            "level": level
        })
        
        # Keep log size limited
        if len(self.status_log) > self.max_log_entries:
            self.status_log = self.status_log[-self.max_log_entries:]
        
        # Also log to the logger
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

def run_simple_terminal_ui(get_status_callback: Callable[[], Dict[str, Any]]):
    """
    # Run a simple text-based UI without curses
    # Use this when curses is not available
    """
    running = True
    
    def print_status():
        os.system('cls' if os.name == 'nt' else 'clear')
        print("===== ML Training Manager =====")
        print()
        
        status = get_status_callback()
        print(f"학습 대기 중: {status.get('waiting', 0)}")
        print(f"학습 중: {status.get('training', 0)}")
        print(f"학습 완료: {status.get('done', 0)}")
        print(f"학습 실패: {status.get('crashed', 0)}")
        print()
        
        if 'current' in status and status['current']:
            print("현재 학습 중인 모델:")
            for model_id, model in status['current'].items():
                print(f"  - {model_id}: {model.get('name', 'Unknown')}")
                print(f"    상태: {model.get('status', 'Unknown')}")
                print(f"    실행 시간: {model.get('runtime', 0):.2f} 초")
                print()
        
        print("명령어: q(종료), r(새로고침)")
    
    while running:
        print_status()
        
        try:
            cmd = input("> ")
            if cmd.lower() == 'q':
                running = False
            elif cmd.lower() == 'r':
                pass  # Just refresh
            else:
                print(f"알 수 없는 명령어: {cmd}")
                time.sleep(1)
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print(f"오류 발생: {e}")
            time.sleep(2)
        
        time.sleep(0.5) 