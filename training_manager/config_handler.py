import os
import configparser
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class ConfigHandler:
    """
    # Configuration handler for ML Training Manager
    # Handles INI configuration files
    """
    def __init__(self, config_file: str = None):
        """
        # Initialize configuration handler
        # config_file: Path to INI configuration file (optional)
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        
        # Set default values
        self._set_defaults()
        
        # Load configuration file if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
            logger.info(f"설정 파일 로드 완료: {config_file}")
        else:
            logger.info("기본 설정 사용")
    
    def _set_defaults(self):
        """
        # Set default configuration values
        """
        # General settings
        self.config['general'] = {
            'csv_file': '',           # Path to ML_Experiment_Table.csv
            'check_interval': '30',   # Check interval in seconds
            'max_training_process': '1',    # Maximum concurrent training processes
            'log_dir': 'logs',        # Directory for logs
            'process_gpu_mapping': '', # Process to GPU mapping (process0=0,process1=1,...)
            'auto_continue': 'true',  # Automatically continue to next model
        }
        
        # WandB settings
        self.config['wandb'] = {
            'entity': '',            # WandB entity (username or team)
            'project': 'Controller-Imitator-Multi-Final',  # WandB project name
            'api_key': '',           # WandB API key (optional)
        }
        
        # Email notification settings
        self.config['email'] = {
            'enable': 'false',       # Enable email notifications
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': '587',
            'username': '',
            'password': '',
            'from_addr': '',
            'to_addr': '',
        }
        
        # Notification settings
        self.config['notification'] = {
            'enable_desktop': 'true',    # Enable desktop notifications
            'enable_sound': 'true',      # Enable sound notifications
            'success_sound': '',         # Path to success sound file
            'error_sound': '',           # Path to error sound file
        }
        
        # Environment setup settings
        self.config['environment'] = {
            'setup_script': '',          # Script to run before training (e.g., conda activation)
            'use_conda': 'false',        # Whether to use conda environment
            'conda_env': '',             # Conda environment name
            'env_vars': '',              # Additional environment variables (key1=value1,key2=value2)
        }
        
        # GPU settings
        self.config['gpu'] = {
            'enable_gpu_assignment': 'true',     # Enable GPU assignment
            'default_gpu': '0',                  # Default GPU ID
            'gpu_list': '0,1,2,3',               # Available GPUs (comma-separated)
            'use_process_order': 'true',         # Use process order for GPU assignment instead of model ID
            'allow_multi_gpu': 'true',           # Allow multiple GPUs for a single process
        }
    
    def load_config(self, config_file: str):
        """
        # Load configuration from file
        # config_file: Path to INI configuration file
        """
        try:
            if not os.path.exists(config_file):
                logger.warning(f"설정 파일을 찾을 수 없음: {config_file}")
                return False
            
            self.config.read(config_file)
            self.config_file = config_file
            return True
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return False
    
    def save_config(self, config_file: str = None):
        """
        # Save current configuration to file
        # config_file: Path to save configuration (default: original file)
        """
        if not config_file:
            config_file = self.config_file
        
        if not config_file:
            logger.warning("저장할 설정 파일 경로가 지정되지 않았습니다.")
            return False
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            with open(config_file, 'w') as f:
                self.config.write(f)
            logger.info(f"설정 파일 저장 완료: {config_file}")
            return True
        except Exception as e:
            logger.error(f"설정 파일 저장 중 오류 발생: {e}")
            return False
    
    def get(self, section: str, option: str, fallback: Any = None) -> str:
        """
        # Get configuration value
        # section: INI section name
        # option: Option name
        # fallback: Default value if not found
        """
        try:
            return self.config.get(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def getint(self, section: str, option: str, fallback: int = 0) -> int:
        """
        # Get integer configuration value
        """
        try:
            return self.config.getint(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def getfloat(self, section: str, option: str, fallback: float = 0.0) -> float:
        """
        # Get float configuration value
        """
        try:
            return self.config.getfloat(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        """
        # Get boolean configuration value
        """
        try:
            return self.config.getboolean(section, option, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def get_list(self, section: str, option: str, fallback: List = None) -> List:
        """
        # Get list configuration value (comma-separated)
        """
        if fallback is None:
            fallback = []
        
        try:
            value = self.config.get(section, option, fallback="")
            if not value:
                return fallback
            return [item.strip() for item in value.split(',')]
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_dict(self, section: str, option: str, fallback: Dict = None) -> Dict:
        """
        # Get dictionary configuration value (comma-separated key=value pairs)
        """
        if fallback is None:
            fallback = {}
        
        try:
            value = self.config.get(section, option, fallback="")
            if not value:
                return fallback
            
            result = {}
            pairs = value.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, val = pair.strip().split('=', 1)
                    result[key.strip()] = val.strip()
            
            return result
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_csv_file_path(self) -> str:
        """
        # Get CSV file path from configuration
        """
        return self.get('general', 'csv_file', '')
    
    def is_auto_continue_enabled(self) -> bool:
        """
        # Check if auto-continue to next model is enabled
        """
        return self.getboolean('general', 'auto_continue', True)
    
    def get_process_gpu_mapping(self) -> Dict[int, Union[str, List[str]]]:
        """
        # Get process to GPU mapping (process index -> GPU ID or list of GPU IDs)
        # Returns: Dictionary mapping process indices to GPU IDs
        """
        mapping_dict = {}
        
        # Get mapping from general/process_gpu_mapping
        mapping_str = self.get('general', 'process_gpu_mapping', '')
        if mapping_str:
            pairs = mapping_str.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, val = pair.strip().split('=', 1)
                    try:
                        # Extract process number from 'process0', 'process1', etc.
                        if key.startswith('process'):
                            process_idx = int(key[7:])
                            
                            # Check if multiple GPUs are specified with '+'
                            if '+' in val:
                                mapping_dict[process_idx] = [gpu_id.strip() for gpu_id in val.split('+')]
                            else:
                                mapping_dict[process_idx] = val.strip()
                    except (ValueError, IndexError):
                        continue
        
        return mapping_dict
    
    def assign_gpu_to_process_index(self, process_index: int) -> Union[str, List[str]]:
        """
        # Assign GPU to a process based on its index
        # process_index: Index of the process (0, 1, 2, ...)
        # Returns: Assigned GPU ID or list of GPU IDs
        """
        if not self.getboolean('gpu', 'enable_gpu_assignment', True):
            return ""
        
        # Check process-specific GPU mapping
        process_mapping = self.get_process_gpu_mapping()
        if process_index in process_mapping:
            return process_mapping[process_index]
        
        # Get available GPUs
        available_gpus = self.get_list('gpu', 'gpu_list', ['0'])
        
        # If no available GPUs, use default
        if not available_gpus:
            return self.get('gpu', 'default_gpu', '0')
        
        # Check if multi-GPU is allowed
        if self.getboolean('gpu', 'allow_multi_gpu', True) and len(available_gpus) > 1:
            # Assign all GPUs to process 0, and single GPUs to others (round-robin)
            if process_index == 0:
                return available_gpus
            else:
                # For other processes, assign individual GPUs in round-robin fashion
                gpu_index = (process_index - 1) % len(available_gpus)
                return available_gpus[gpu_index]
        else:
            # Assign GPU based on process index (round-robin)
            gpu_index = process_index % len(available_gpus)
            return available_gpus[gpu_index]
    
    def get_environment_setup(self) -> Dict[str, Any]:
        """
        # Get environment setup configuration
        # Returns: Dictionary with environment setup details
        """
        setup_script = self.get('environment', 'setup_script', '')
        use_conda = self.getboolean('environment', 'use_conda', False)
        conda_env = self.get('environment', 'conda_env', '')
        env_vars = self.get_dict('environment', 'env_vars', {})
        
        return {
            'setup_script': setup_script,
            'use_conda': use_conda,
            'conda_env': conda_env,
            'env_vars': env_vars
        }
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """
        # Get WandB configuration
        # Returns: Dictionary with WandB configuration
        """
        entity = self.get('wandb', 'entity', '')
        project = self.get('wandb', 'project', 'Controller-Imitator-Multi-Final')
        api_key = self.get('wandb', 'api_key', '')
        
        return {
            'entity': entity,
            'project': project,
            'api_key': api_key
        }
    
    def get_email_config(self) -> Dict[str, Any]:
        """
        # Get email notification configuration
        # Returns: Dictionary with email configuration
        """
        enable = self.getboolean('email', 'enable', False)
        smtp_server = self.get('email', 'smtp_server', 'smtp.gmail.com')
        smtp_port = self.getint('email', 'smtp_port', 587)
        username = self.get('email', 'username', '')
        password = self.get('email', 'password', '')
        from_addr = self.get('email', 'from_addr', '')
        to_addr = self.get('email', 'to_addr', '')
        
        return {
            'enable': enable,
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_addr': from_addr,
            'to_addr': to_addr
        }

    def create_default_config(self, config_file: str):
        """
        # Create a default configuration file if it doesn't exist
        # config_file: Path to INI configuration file
        """
        if os.path.exists(config_file):
            logger.warning(f"설정 파일이 이미 존재합니다: {config_file}")
            return False
        
        return self.save_config(config_file) 