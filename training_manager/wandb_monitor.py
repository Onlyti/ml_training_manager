import wandb
import logging
from typing import Dict, List, Any, Optional
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WandbMonitor:
    """
    # Monitor training runs with Weights & Biases
    """
    def __init__(self, entity: str, project_name: str):
        """
        # Initialize WandB monitor
        # entity: Username or team name
        # project_name: WandB project name
        """
        self.entity = entity
        self.project_name = project_name
        self.api = wandb.Api()
        logger.info(f"WandB 모니터 초기화: 엔티티={entity}, 프로젝트={project_name}")
    
    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """
        # Get the status of a specific run
        # run_id: WandB run ID
        """
        try:
            run = self.api.run(f"{self.entity}/{self.project_name}/{run_id}")
            
            # Extract relevant information
            status = {
                "id": run.id,
                "name": run.name,
                "state": run.state,  # 'running', 'finished', 'crashed', etc.
                "created_at": run.created_at,
                "heartbeat_at": getattr(run, "heartbeat_at", None),
                "runtime": getattr(run, "runtime", 0),
                "summary": {},
            }
            
            # Extract summary metrics
            if hasattr(run, "summary"):
                for key, value in run.summary._json_dict.items():
                    if isinstance(value, (int, float, str)):
                        status["summary"][key] = value
            
            return status
        except Exception as e:
            logger.error(f"WandB 실행 상태 확인 중 오류 발생: {e}")
            return {"id": run_id, "state": "unknown", "error": str(e)}
    
    def is_run_finished(self, run_id: str) -> bool:
        """
        # Check if a run is finished ('finished', 'failed', 'crashed', etc.)
        """
        status = self.get_run_status(run_id)
        return status["state"] in ["finished", "failed", "crashed"]
    
    def is_run_crashed(self, run_id: str) -> bool:
        """
        # Check if a run has crashed or failed
        """
        status = self.get_run_status(run_id)
        return status["state"] in ["crashed", "failed"]
    
    def is_run_stalled(self, run_id: str, timeout_minutes: int = 30) -> bool:
        """
        # Check if a run is stalled (no heartbeat for a specified time)
        """
        status = self.get_run_status(run_id)
        
        if status["state"] != "running":
            return False
        
        if not status.get("heartbeat_at"):
            # If no heartbeat info, check created time
            return False
        
        try:
            heartbeat_time = datetime.strptime(status["heartbeat_at"], "%Y-%m-%dT%H:%M:%S")
            current_time = datetime.now().replace(microsecond=0)
            time_diff = current_time - heartbeat_time
            
            return time_diff > timedelta(minutes=timeout_minutes)
        except Exception as e:
            logger.error(f"스톨 상태 확인 중 오류 발생: {e}")
            return False
    
    def get_run_metrics(self, run_id: str, keys: List[str] = None) -> Dict[str, Any]:
        """
        # Get specific metrics from a run
        """
        try:
            run = self.api.run(f"{self.entity}/{self.project_name}/{run_id}")
            history = run.scan_history(keys=keys)
            
            metrics = {}
            for row in history:
                for key, value in row.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            
            return metrics
        except Exception as e:
            logger.error(f"WandB 메트릭 가져오기 중 오류 발생: {e}")
            return {}
    
    def get_latest_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        # Get the latest runs from the project
        """
        try:
            runs = self.api.runs(f"{self.entity}/{self.project_name}", per_page=limit)
            
            latest_runs = []
            for run in runs:
                latest_runs.append({
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                })
                
                if len(latest_runs) >= limit:
                    break
                    
            return latest_runs
        except Exception as e:
            logger.error(f"최신 실행 목록 가져오기 중 오류 발생: {e}")
            return []
    
    def extract_output_info(self, run_id: str) -> Dict[str, Any]:
        """
        # Extract information about output files from the run
        # This can help identify weight files
        """
        try:
            run = self.api.run(f"{self.entity}/{self.project_name}/{run_id}")
            files = run.files()
            
            output_info = {
                "weight_files": [],
                "log_files": [],
                "other_files": []
            }
            
            for file in files:
                if file.name.endswith(('.pt', '.pth', '.h5', '.keras', '.model', '.weights')):
                    output_info["weight_files"].append(file.name)
                elif file.name.endswith(('.log', '.txt')):
                    output_info["log_files"].append(file.name)
                else:
                    output_info["other_files"].append(file.name)
            
            return output_info
        except Exception as e:
            logger.error(f"출력 정보 추출 중 오류 발생: {e}")
            return {"weight_files": [], "log_files": [], "other_files": []}
    
    def get_run_name(self, run_id: str) -> Optional[str]:
        """
        # Get the name of a run
        # run_id: WandB run ID
        # Returns: Name of the run or None if an error occurs
        """
        try:
            run = self.api.run(f"{self.entity}/{self.project_name}/{run_id}")
            return run.name
        except Exception as e:
            logger.error(f"WandB Run 이름 가져오기 중 오류 발생: {e}")
            return None
    
    def get_output_dir(self, run_id: str) -> Optional[str]:
        """
        # Get the output directory of a run
        # run_id: WandB run ID
        # Returns: Path to the output directory or None if an error occurs
        """
        try:
            run = self.api.run(f"{self.entity}/{self.project_name}/{run_id}")
            # WandB 런은 일반적으로 config.yaml 파일에 출력 디렉토리 정보를 저장
            config = run.config
            
            # 일반적인 출력 디렉토리 키 검사
            output_dir_keys = ['output_dir', 'save_dir', 'checkpoint_dir', 'model_dir', 'log_dir']
            
            for key in output_dir_keys:
                if key in config:
                    return config[key]
            
            # 직접적인 키가 없으면 추론 시도
            for key in config.keys():
                if 'dir' in key.lower() or 'path' in key.lower() or 'save' in key.lower() or 'output' in key.lower():
                    if isinstance(config[key], str):
                        return config[key]
            
            # 다른 방법: 파일 경로에서 추론
            files = run.files()
            for file in files:
                if file.name.endswith(('.pt', '.pth')):
                    # 파일 경로에서 디렉토리 추출
                    path_parts = file.name.split('/')
                    if len(path_parts) > 1:
                        # 마지막 부분(파일 이름)을 제외한 경로 반환
                        return '/'.join(path_parts[:-1])
            
            # 마지막 수단: run 이름 사용
            return run.name
        except Exception as e:
            logger.error(f"WandB 출력 디렉토리 가져오기 중 오류 발생: {e}")
            return None 