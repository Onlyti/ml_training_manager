import os
import pandas as pd
import csv
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CSVHandler:
    """
    # CSV file handler for ML experiment table
    """
    def __init__(self, csv_file_path: str):
        """
        # Initialize CSV handler
        # csv_file_path: Path to ML_Experiment_Table.csv
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.reload()
    
    def reload(self) -> None:
        """
        # Reload CSV file data
        """
        if os.path.exists(self.csv_file_path):
            try:
                # 비어있는 값을 None으로 처리 
                self.df = pd.read_csv(self.csv_file_path)
                logger.info(f"CSV 파일 로드 완료: {self.csv_file_path}")
            except Exception as e:
                logger.error(f"CSV 파일 로드 중 오류 발생: {e}")
                raise
        else:
            logger.error(f"CSV 파일을 찾을 수 없음: {self.csv_file_path}")
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없음: {self.csv_file_path}")
    
    def get_untrained_models(self) -> pd.DataFrame:
        """
        # Get models that haven't been trained yet (TrainingCheck is empty)
        """
        return self.df[self.df['TrainingCheck'].isna() | (self.df['TrainingCheck'] == '')]
    
    def get_models_in_training(self) -> pd.DataFrame:
        """
        # Get models that are currently training (TrainingCheck is 'Training')
        """
        return self.df[self.df['TrainingCheck'] == 'Training']
    
    def get_trained_models(self) -> pd.DataFrame:
        """
        # Get models that have been trained (TrainingCheck is 'Done')
        """
        return self.df[self.df['TrainingCheck'] == 'Done']
    
    def get_crashed_models(self) -> pd.DataFrame:
        """
        # Get models that crashed during training (TrainingCheck is 'Crash')
        """
        return self.df[self.df['TrainingCheck'] == 'Crash']
    
    def get_model_by_id(self, model_id: str) -> Optional[pd.Series]:
        """
        # Get model data by ID
        """
        model = self.df[self.df['ID'] == model_id]
        if model.empty:
            return None
        return model.iloc[0]
    
    def update_model_status(self, model_id: str, status: str) -> bool:
        """
        # Update TrainingCheck status for a model
        # status: 'Training', 'Done', or 'Crash'
        """
        return self.update_value(model_id, 'TrainingCheck', status)
    
    def update_weight_file(self, model_id: str, weight_file: str) -> bool:
        """
        # Update WeightFile for a model
        """
        return self.update_value(model_id, 'WeightFile', weight_file)
    
    def update_value(self, model_id: str, column: str, value: Any) -> bool:
        """
        # Update a value in a specific column for a model
        # model_id: Model ID to update
        # column: Column name to update
        # value: New value to set
        """
        try:
            self.reload()  # Reload to get the latest data
            idx = self.df[self.df['ID'] == model_id].index
            if idx.empty:
                logger.error(f"모델 ID를 찾을 수 없음: {model_id}")
                return False
            
            # Check if column exists
            if column not in self.df.columns:
                logger.error(f"열을 찾을 수 없음: {column}")
                return False
            
            # Update the value (None 값도 처리 가능)
            self.df.loc[idx, column] = value
            
            # Save the updated DataFrame to CSV
            self.df.to_csv(self.csv_file_path, index=False, na_rep='')  # 비어있는 값은 빈 문자열로 저장
            logger.info(f"모델 {model_id}의 {column} 값을 '{value}'로 업데이트했습니다.")
            return True
        except Exception as e:
            logger.error(f"값 업데이트 중 오류 발생: {e}")
            return False
    
    def get_training_command(self, model_id: str) -> Optional[str]:
        """
        # Get training command for a model
        """
        model = self.get_model_by_id(model_id)
        if model is None:
            return None
        
        # 비어있는 값인 경우 None 반환
        command = model.get('TrainingCommand')
        if pd.isna(command):
            return None
        
        return command

    def get_all_models(self) -> pd.DataFrame:
        """
        # Get all models in the CSV file
        """
        return self.df 

    def is_empty_value(self, model_id: str, column: str) -> bool:
        """
        # Check if a value in a specific column is empty (NaN, None, or empty string)
        """
        model = self.get_model_by_id(model_id)
        if model is None:
            return True
        
        value = model.get(column)
        if pd.isna(value) or value == '':
            return True
        
        return False 