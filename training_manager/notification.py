import logging
import smtplib
import os
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
import platform
import subprocess

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    # Notification manager for training events
    """
    def __init__(self, enable_email: bool = False, enable_desktop: bool = True, enable_sound: bool = True):
        """
        # Initialize notification manager
        # enable_email: Whether to enable email notifications
        # enable_desktop: Whether to enable desktop notifications
        # enable_sound: Whether to enable sound notifications
        """
        self.enable_email = enable_email
        self.enable_desktop = enable_desktop
        self.enable_sound = enable_sound
        
        # Email configuration
        self.email_config = {
            "smtp_server": os.environ.get("NOTIFY_SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.environ.get("NOTIFY_SMTP_PORT", "587")),
            "username": os.environ.get("NOTIFY_EMAIL_USERNAME", ""),
            "password": os.environ.get("NOTIFY_EMAIL_PASSWORD", ""),
            "from_addr": os.environ.get("NOTIFY_FROM_ADDR", ""),
            "to_addr": os.environ.get("NOTIFY_TO_ADDR", ""),
        }
        
        # Sound configuration
        self.sound_config = {
            "success_sound": os.environ.get("NOTIFY_SUCCESS_SOUND", ""),
            "error_sound": os.environ.get("NOTIFY_ERROR_SOUND", ""),
        }
        
        logger.info("알림 매니저 초기화 완료")
    
    def notify_training_started(self, model_id: str, model_name: str):
        """
        # Send notification that training has started
        """
        subject = f"학습 시작: {model_name} ({model_id})"
        message = f"모델 {model_name} (ID: {model_id})의 학습이 시작되었습니다."
        
        if self.enable_email:
            self._send_email(subject, message)
        
        if self.enable_desktop:
            self._send_desktop_notification(subject, message)
        
        logger.info(f"학습 시작 알림 전송: {model_id}")
    
    def notify_training_completed(self, model_id: str, model_name: str, runtime_hours: float, wandb_url: Optional[str] = None):
        """
        # Send notification that training has completed successfully
        """
        subject = f"학습 완료: {model_name} ({model_id})"
        message = f"모델 {model_name} (ID: {model_id})의 학습이 완료되었습니다.\n"
        message += f"학습 시간: {runtime_hours:.2f} 시간\n"
        
        if wandb_url:
            message += f"WandB URL: {wandb_url}\n"
        
        if self.enable_email:
            self._send_email(subject, message)
        
        if self.enable_desktop:
            self._send_desktop_notification(subject, message)
        
        if self.enable_sound:
            self._play_sound("success")
        
        logger.info(f"학습 완료 알림 전송: {model_id}")
    
    def notify_training_crashed(self, model_id: str, model_name: str, error_msg: str, runtime_hours: float):
        """
        # Send notification that training has crashed
        """
        subject = f"학습 중단 (에러): {model_name} ({model_id})"
        message = f"모델 {model_name} (ID: {model_id})의 학습이 중단되었습니다.\n"
        message += f"학습 시간: {runtime_hours:.2f} 시간\n"
        message += f"오류 메시지: {error_msg}\n"
        
        if self.enable_email:
            self._send_email(subject, message)
        
        if self.enable_desktop:
            self._send_desktop_notification(subject, message, is_error=True)
        
        if self.enable_sound:
            self._play_sound("error")
        
        logger.info(f"학습 중단 알림 전송: {model_id}")
    
    def notify_all_training_completed(self, total_models: int, runtime_hours: float):
        """
        # Send notification that all training has completed
        """
        subject = f"모든 학습 완료 ({total_models}개 모델)"
        message = f"총 {total_models}개 모델의 학습이 모두 완료되었습니다.\n"
        message += f"전체 소요 시간: {runtime_hours:.2f} 시간\n"
        
        if self.enable_email:
            self._send_email(subject, message)
        
        if self.enable_desktop:
            self._send_desktop_notification(subject, message)
        
        if self.enable_sound:
            self._play_sound("success")
        
        logger.info("모든 학습 완료 알림 전송")
    
    def _send_email(self, subject: str, message: str):
        """
        # Send email notification
        """
        if not all([
            self.email_config["smtp_server"],
            self.email_config["username"],
            self.email_config["password"],
            self.email_config["from_addr"],
            self.email_config["to_addr"]
        ]):
            logger.warning("이메일 설정이 완료되지 않아 이메일을 보낼 수 없습니다.")
            return
        
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_config["from_addr"]
            msg["To"] = self.email_config["to_addr"]
            msg["Subject"] = subject
            
            msg.attach(MIMEText(message, "plain"))
            
            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"]) as server:
                server.starttls()
                server.login(self.email_config["username"], self.email_config["password"])
                server.send_message(msg)
            
            logger.info(f"이메일 알림 전송 완료: {subject}")
        except Exception as e:
            logger.error(f"이메일 전송 중 오류 발생: {e}")
    
    def _send_desktop_notification(self, title: str, message: str, is_error: bool = False):
        """
        # Send desktop notification (platform-specific)
        """
        try:
            system = platform.system()
            
            if system == "Windows":
                # Using Windows toast notification
                try:
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(title, message, duration=10, threaded=True)
                except ImportError:
                    # Fallback using powershell
                    msg = message.replace("'", "''")
                    title = title.replace("'", "''")
                    ps_cmd = f'powershell -command "& {{Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; $notify = New-Object System.Windows.Forms.NotifyIcon; $notify.Icon = [System.Drawing.SystemIcons]::Information; $notify.Visible = $true; $notify.ShowBalloonTip(0, \'{title}\', \'{msg}\', [System.Windows.Forms.ToolTipIcon]::None)}}"'
                    subprocess.run(ps_cmd, shell=True)
            
            elif system == "Darwin":  # macOS
                # Using AppleScript
                cmd = f'''
                osascript -e 'display notification "{message}" with title "{title}"'
                '''
                subprocess.run(cmd, shell=True)
            
            elif system == "Linux":
                # Using notify-send (requires libnotify-bin)
                cmd = f'notify-send "{title}" "{message}"'
                subprocess.run(cmd, shell=True)
            
            logger.info(f"데스크톱 알림 전송 완료: {title}")
        except Exception as e:
            logger.error(f"데스크톱 알림 전송 중 오류 발생: {e}")
    
    def _play_sound(self, sound_type: str):
        """
        # Play notification sound (platform-specific)
        """
        try:
            system = platform.system()
            
            # Use custom sound if specified
            sound_file = None
            if sound_type == "success" and self.sound_config["success_sound"]:
                sound_file = self.sound_config["success_sound"]
            elif sound_type == "error" and self.sound_config["error_sound"]:
                sound_file = self.sound_config["error_sound"]
            
            if sound_file and os.path.exists(sound_file):
                # Play custom sound file
                if system == "Windows":
                    cmd = f'powershell -c "(New-Object Media.SoundPlayer \'{sound_file}\').PlaySync()"'
                    subprocess.run(cmd, shell=True)
                elif system == "Darwin":  # macOS
                    cmd = f'afplay "{sound_file}"'
                    subprocess.run(cmd, shell=True)
                elif system == "Linux":
                    cmd = f'paplay "{sound_file}" || aplay "{sound_file}"'
                    subprocess.run(cmd, shell=True)
            else:
                # Use system sounds
                if system == "Windows":
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION if sound_type == "error" else winsound.MB_OK)
                elif system == "Darwin":  # macOS
                    cmd = 'afplay /System/Library/Sounds/Glass.aiff'
                    subprocess.run(cmd, shell=True)
                elif system == "Linux":
                    cmd = 'paplay /usr/share/sounds/freedesktop/stereo/complete.oga || aplay /usr/share/sounds/freedesktop/stereo/complete.oga'
                    subprocess.run(cmd, shell=True)
            
            logger.info(f"알림 사운드 재생 완료: {sound_type}")
        except Exception as e:
            logger.error(f"알림 사운드 재생 중 오류 발생: {e}")
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, password: str, from_addr: str, to_addr: str):
        """
        # Configure email settings
        """
        self.email_config["smtp_server"] = smtp_server
        self.email_config["smtp_port"] = smtp_port
        self.email_config["username"] = username
        self.email_config["password"] = password
        self.email_config["from_addr"] = from_addr
        self.email_config["to_addr"] = to_addr
        self.enable_email = True
        logger.info("이메일 설정이 업데이트되었습니다.")
    
    def configure_sound(self, success_sound: str = "", error_sound: str = ""):
        """
        # Configure sound settings
        """
        if success_sound:
            self.sound_config["success_sound"] = success_sound
        if error_sound:
            self.sound_config["error_sound"] = error_sound
        logger.info("사운드 설정이 업데이트되었습니다.") 