#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import datetime

class CommandHistoryManager:
    """
    생성된 명령어를 CSV 파일로 관리하는 클래스
    """
    def __init__(self, csv_file_path=None):
        """
        명령어 히스토리 관리자 초기화
        
        Args:
            csv_file_path (str, optional): CSV 파일 경로. 기본값은 None.
        """
        if csv_file_path:
            self.csv_file_path = csv_file_path
        else:
            # 기본 경로는 현재 모듈과 같은 폴더의 command_history.csv
            self.csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'command_history.csv')
        
        # 파일이 없으면 생성
        if not os.path.exists(self.csv_file_path):
            self._create_empty_csv()
    
    def _create_empty_csv(self):
        """
        비어있는 CSV 파일 생성
        """
        try:
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'timestamp', 'description', 'command', 'exit_code', 'output'])
            return True
        except Exception as e:
            print(f"CSV 파일 생성 오류: {e}")
            return False
    
    def add_command(self, command, description="", exit_code=None, output=""):
        """
        명령어를 히스토리에 추가
        
        Args:
            command (str): 실행된 명령어
            description (str, optional): 명령어 설명
            exit_code (int, optional): 명령어 실행 결과 코드
            output (str, optional): 명령어 실행 출력
            
        Returns:
            int: 추가된 명령어 ID
        """
        try:
            # 기존 명령어 불러오기
            commands = self.get_all_commands()
            
            # 새 ID 생성
            command_id = 1
            if commands:
                command_id = max([int(cmd.get('id', 0)) for cmd in commands]) + 1
            
            # 현재 시간
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 새 행 추가
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([command_id, timestamp, description, command, exit_code, output])
            
            return command_id
        
        except Exception as e:
            print(f"명령어 추가 오류: {e}")
            return None
    
    def get_all_commands(self):
        """
        모든 명령어 히스토리 가져오기
        
        Returns:
            list: 명령어 목록 (딕셔너리 형태)
        """
        try:
            if not os.path.exists(self.csv_file_path):
                return []
            
            commands = []
            with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    commands.append(row)
            
            return commands
        
        except Exception as e:
            print(f"명령어 조회 오류: {e}")
            return []
    
    def get_command_by_id(self, command_id):
        """
        특정 ID의 명령어 가져오기
        
        Args:
            command_id (int): 명령어 ID
            
        Returns:
            dict: 명령어 정보
        """
        try:
            commands = self.get_all_commands()
            for cmd in commands:
                if int(cmd.get('id', 0)) == int(command_id):
                    return cmd
            
            return None
        
        except Exception as e:
            print(f"명령어 조회 오류: {e}")
            return None
    
    def update_command_result(self, command_id, exit_code, output):
        """
        명령어 실행 결과 업데이트
        
        Args:
            command_id (int): 명령어 ID
            exit_code (int): 실행 결과 코드
            output (str): 명령어 실행 출력
            
        Returns:
            bool: 성공 여부
        """
        try:
            commands = self.get_all_commands()
            updated = False
            
            # 임시 파일에 업데이트된 내용 저장
            temp_file = self.csv_file_path + '.temp'
            with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'timestamp', 'description', 'command', 'exit_code', 'output'])
                
                for cmd in commands:
                    row = [cmd['id'], cmd['timestamp'], cmd['description'], cmd['command']]
                    
                    if int(cmd['id']) == int(command_id):
                        row.append(exit_code)
                        row.append(output)
                        updated = True
                    else:
                        row.append(cmd['exit_code'])
                        row.append(cmd['output'])
                    
                    writer.writerow(row)
            
            # 임시 파일을 원래 파일로 대체
            if updated:
                os.replace(temp_file, self.csv_file_path)
                return True
            else:
                os.remove(temp_file)
                return False
        
        except Exception as e:
            print(f"명령어 결과 업데이트 오류: {e}")
            return False
    
    def delete_command(self, command_id):
        """
        명령어 삭제
        
        Args:
            command_id (int): 명령어 ID
            
        Returns:
            bool: 성공 여부
        """
        try:
            commands = self.get_all_commands()
            deleted = False
            
            # 임시 파일에 삭제된 행을 제외한 내용 저장
            temp_file = self.csv_file_path + '.temp'
            with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'timestamp', 'description', 'command', 'exit_code', 'output'])
                
                for cmd in commands:
                    if int(cmd['id']) != int(command_id):
                        writer.writerow([cmd['id'], cmd['timestamp'], cmd['description'], 
                                       cmd['command'], cmd['exit_code'], cmd['output']])
                    else:
                        deleted = True
            
            # 임시 파일을 원래 파일로 대체
            if deleted:
                os.replace(temp_file, self.csv_file_path)
                return True
            else:
                os.remove(temp_file)
                return False
        
        except Exception as e:
            print(f"명령어 삭제 오류: {e}")
            return False 