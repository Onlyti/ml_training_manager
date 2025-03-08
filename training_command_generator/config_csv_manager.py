#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
from datetime import datetime
import collections

class ConfigCSVManager:
    """
    모델 구성 설정을 CSV 파일로 관리하는 클래스
    """
    def __init__(self, csv_file_path=None):
        """
        CSV 관리자 초기화
        
        Args:
            csv_file_path (str, optional): CSV 파일 경로. 기본값은 None.
        """
        if csv_file_path:
            self.csv_file_path = csv_file_path
        else:
            # 기본 경로는 현재 모듈과 같은 폴더의 model_configs.csv
            self.csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_configs.csv')
    
    def _create_empty_csv(self):
        """
        비어있는 CSV 파일 생성
        """
        try:
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['idx'])  # 첫 번째 열은 구성 인덱스
            return True
        except Exception as e:
            print(f"CSV 파일 생성 오류: {e}")
            return False
    
    def capture_current_config(self, checkboxes, env_variables, custom_configs, pre_commands, config_name=None):
        """
        현재 UI의 구성을 캡처하여 CSV에 저장
        
        Args:
            checkboxes (dict): 섹션별 체크박스 정보 또는 체크박스 구성 데이터
            env_variables (list): 환경 변수 정보
            custom_configs (list): 사용자 정의 설정 정보
            pre_commands (list): 사전 명령어 정보
            config_name (str, optional): 구성 이름. 지정하지 않으면 자동 번호 부여
            
        Returns:
            tuple: (성공 여부, 구성 인덱스)
        """
        try:
            # 전달된 데이터의 구조 확인
            is_proper_format = True
            
            try:
                # 체크박스 구성이 이미 적절한 형식(TrainingCommandGenerator에서 전처리됨)인지 확인
                for section, params in checkboxes.items():
                    if not isinstance(params, dict):
                        is_proper_format = False
                        break
                    for param, value in params.items():
                        if not isinstance(value, dict) or 'checked' not in value:
                            is_proper_format = False
                            break
            except Exception:
                is_proper_format = False
            
            # 데이터가 이미 적절한 형식인 경우
            if is_proper_format:
                print("이미 처리된 체크박스 구성 데이터 사용")
                checkboxes_config = checkboxes
            # 적절한 형식이 아닌 경우 변환 시도
            else:
                print("체크박스 데이터 형식 변환 필요")
                checkboxes_config = {}
                
                for section_name, section_data in checkboxes.items():
                    if section_name not in checkboxes_config:
                        checkboxes_config[section_name] = {}
                    
                    # 리스트 형식인 경우 [(display_name, value, checkbox), ...]
                    if isinstance(section_data, list):
                        for item in section_data:
                            if len(item) >= 3:
                                display_name, value, checkbox = item[0], item[1], item[2]
                                try:
                                    if hasattr(checkbox, 'isChecked'):
                                        is_checked = checkbox.isChecked()
                                    else:
                                        is_checked = False
                                        print(f"경고: '{section_name}.{display_name}'은(는) isChecked 메서드가 없음")
                                except Exception as e:
                                    is_checked = False
                                    print(f"체크박스 상태 확인 오류: {e}")
                                
                                checkboxes_config[section_name][display_name] = {
                                    'display_name': display_name,
                                    'checked': is_checked,
                                    'value': value if is_checked else "0"
                                }
                    # 딕셔너리 형식인 경우 {param_name: checkbox, ...}
                    elif isinstance(section_data, dict):
                        for param_name, checkbox in section_data.items():
                            try:
                                if hasattr(checkbox, 'isChecked'):
                                    is_checked = checkbox.isChecked()
                                else:
                                    # 이미 딕셔너리 형식인 경우 ('checked' 키가 있는지 확인)
                                    if isinstance(checkbox, dict) and 'checked' in checkbox:
                                        is_checked = checkbox['checked']
                                    else:
                                        is_checked = False
                                        print(f"경고: '{section_name}.{param_name}'은(는) isChecked 메서드가 없고 'checked' 키도 없음")
                            except Exception as e:
                                is_checked = False
                                print(f"체크박스 상태 확인 오류: {e}")
                            
                            value = "1" if is_checked else "0"
                            # 값이 있는 경우 가져오기
                            if isinstance(checkbox, dict) and 'value' in checkbox:
                                if checkbox['value'] and checkbox['value'] != "0":
                                    value = checkbox['value']
                            
                            checkboxes_config[section_name][param_name] = {
                                'display_name': param_name,
                                'checked': is_checked,
                                'value': value
                            }
                    else:
                        print(f"경고: 지원되지 않는 체크박스 데이터 형식: {type(section_data)}")
            
            # 데이터 유효성 확인
            print(f"처리된 체크박스 구성: {checkboxes_config}")
            
            # 새 메서드 호출
            config_idx = self.save_config_to_csv(
                checkboxes_config, env_variables, custom_configs, pre_commands, config_name)
                
            return (config_idx is not None, config_idx)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"구성 캡처 오류: {e}")
            return (False, None)
    
    def _transform_csv_format(self):
        """
        CSV 파일을 요청한 형식으로 변환
        - 첫 번째 행: 섹션 이름 (반복됨)
        - 두 번째 행: 파라미터 이름
        - 세 번째 행 이후: 값
        """
        try:
            # 저장된 CSV 파일 읽기
            df = pd.read_csv(self.csv_file_path)
            
            # 인덱스 열을 정수형으로 변환 (소수점은 버림)
            df['idx'] = df['idx'].apply(lambda x: int(float(x)) if pd.notnull(x) else 0)
            
            # 섹션별로 컬럼 그룹화
            section_columns = collections.defaultdict(list)
            for col in df.columns:
                if col == 'idx':
                    continue
                
                if '.' in col:
                    section, param = col.split('.', 1)
                    section_columns[section].append((col, param))
            
            # 새 CSV 파일 작성
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 첫 번째 행: 섹션 이름
                header_row1 = ['idx']
                
                # 두 번째 행: 파라미터 이름
                header_row2 = ['idx']
                
                # 각 섹션의 컬럼 추가
                for section, columns in section_columns.items():
                    for original_col, param in columns:
                        header_row1.append(section)
                        header_row2.append(param)
                
                # 헤더 행 작성
                writer.writerow(header_row1)
                writer.writerow(header_row2)
                
                # 데이터 행 작성
                for _, row in df.iterrows():
                    data_row = [int(row['idx'])]  # 인덱스를 정수로 변환하여 저장
                    
                    for section, columns in section_columns.items():
                        for original_col, _ in columns:
                            value = row.get(original_col, '')
                            data_row.append(value)
                    
                    writer.writerow(data_row)
            
            return True
        
        except Exception as e:
            print(f"CSV 변환 오류: {e}")
            return False
    
    def load_config_from_csv(self, config_name):
        """
        CSV 파일에서 특정 구성 로드
        
        Args:
            config_name (str): 로드할 구성 이름(또는 인덱스)
            
        Returns:
            tuple: (checkboxes_config, env_variables, custom_configs, pre_commands) 또는 None
        """
        try:
            # CSV 파일 읽기
            if not os.path.exists(self.csv_file_path):
                return None
            
            # CSV 파일 읽기 (일반 csv 모듈 사용)
            with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 3:  # 최소 3행 필요 (섹션, 파라미터, 데이터)
                    return None
                
                # 각 행 파싱
                section_row = rows[0]  # 첫 번째 행: 섹션
                param_row = rows[1]    # 두 번째 행: 파라미터
                
                # 선택된 구성 인덱스 찾기
                try:
                    # 문자열에 '.' 포함된 경우(예: '1.0') 먼저 float로 변환 후 int로 변환
                    if config_name.replace('.', '', 1).isdigit() and '.' in config_name:
                        config_idx = int(float(config_name))
                    else:
                        config_idx = int(config_name)
                except ValueError:
                    print(f"구성 번호 변환 오류: {config_name}은(는) 유효한 숫자가 아닙니다.")
                    return None
                
                # 해당 구성 데이터 찾기
                data_row = None
                for row in rows[2:]:  # 세 번째 행부터 데이터
                    if len(row) > 0 and row[0]:
                        try:
                            # 행의 첫 번째 열도 같은 방식으로 변환 시도
                            row_idx = row[0]
                            if row_idx.replace('.', '', 1).isdigit() and '.' in row_idx:
                                row_idx = int(float(row_idx))
                            else:
                                row_idx = int(row_idx)
                                
                            if row_idx == config_idx:
                                data_row = row
                                break
                        except ValueError:
                            # 숫자로 변환할 수 없는 경우 무시
                            continue
                
                if data_row is None:
                    return None
                
                # 결과 데이터 준비
                checkboxes_config = {}
                env_variables = []
                custom_configs = []
                pre_commands = []
                
                # 디버그용
                print(f"로드된 구성 {config_idx}의 데이터 행: {data_row}")
                
                # 각 열의 데이터 처리
                for i in range(1, len(section_row)):
                    if i >= len(data_row):
                        continue
                        
                    section = section_row[i]
                    param = param_row[i]
                    value = data_row[i]
                    
                    # 값이 없는 경우 스킵 (빈 문자열, 'nan', 등)
                    if not value or value.lower() == 'nan':
                        continue
                    
                    # 디버그용
                    print(f"처리 중: 섹션={section}, 파라미터={param}, 값={value}")
                    
                    # 빈 섹션 처리
                    if param == "_empty_":
                        if section not in checkboxes_config:
                            checkboxes_config[section] = {}
                        # 빈 섹션임을 나타내는 특수 파라미터
                        checkboxes_config[section]["ON"] = {
                            'display_name': "ON",
                            'checked': True,  # 항상 체크됨
                            'value': ""
                        }
                        continue
                    
                    # 섹션 별로 데이터 처리
                    if section == 'ENV_VARIABLES':
                        env_variables.append({
                            'name': param,
                            'value': value,
                            'enabled': True
                        })
                    
                    elif section == 'CUSTOM_CONFIG':
                        custom_configs.append({
                            'param': param,
                            'value': value,
                            'enabled': True
                        })
                    
                    elif section == 'PRE_COMMANDS':
                        cmd_idx = int(param.replace('cmd_', '')) - 1
                        while len(pre_commands) <= cmd_idx:
                            pre_commands.append(None)
                        
                        pre_commands[cmd_idx] = {
                            'command': value,
                            'description': f"명령어 {cmd_idx+1}",
                            'enabled': True
                        }
                    
                    else:
                        # 일반 체크박스 설정
                        if section not in checkboxes_config:
                            checkboxes_config[section] = {}
                        
                        # 값이 0/1 또는 True/False 등으로 저장된 경우 불리언 값으로 변환
                        is_checked = False  # 기본값은 체크 해제됨
                        
                        # "1", "true", "True" 등만 체크된 것으로 판단
                        str_value = str(value).lower().strip()
                        if str_value in ('1', 'true', 'yes', 'y', 'on'):
                            is_checked = True
                        
                        checkboxes_config[section][param] = {
                            'display_name': param,
                            'checked': is_checked,
                            'value': value
                        }
                        
                        # 디버그용
                        print(f"체크박스 설정: 섹션={section}, 파라미터={param}, 체크={is_checked}, 값={value}")
                
                # None 항목 제거
                pre_commands = [cmd for cmd in pre_commands if cmd is not None]
                
                # 디버그용
                print(f"로드된 체크박스 설정: {checkboxes_config}")
                
                return checkboxes_config, env_variables, custom_configs, pre_commands
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"구성 로드 오류: {e}")
            return None
    
    def get_available_configs(self):
        """
        사용 가능한 구성 목록 반환
        
        Returns:
            list: 구성 인덱스 목록
        """
        try:
            if not os.path.exists(self.csv_file_path):
                return []
            
            # CSV 파일 읽기
            with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 3:  # 최소 3행 필요
                    return []
                
                # 인덱스 추출 (세 번째 행부터)
                configs = []
                for row in rows[2:]:
                    if row and row[0]:
                        configs.append(row[0])
                
                return configs
        
        except Exception as e:
            print(f"구성 목록 조회 오류: {e}")
            return []
    
    def delete_config(self, config_name):
        """
        CSV 파일에서 특정 구성 삭제
        
        Args:
            config_name (str): 삭제할 구성 인덱스
            
        Returns:
            bool: 성공 여부
        """
        try:
            if not os.path.exists(self.csv_file_path):
                return False
            
            # CSV 파일 읽기
            with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 3:  # 최소 3행 필요
                    return False
                
                # 선택된 구성 인덱스 찾기
                try:
                    # 문자열에 '.' 포함된 경우(예: '1.0') 먼저 float로 변환 후 int로 변환
                    if config_name.replace('.', '', 1).isdigit() and '.' in config_name:
                        config_idx = int(float(config_name))
                    else:
                        config_idx = int(config_name)
                except ValueError:
                    print(f"구성 번호 변환 오류: {config_name}은(는) 유효한 숫자가 아닙니다.")
                    return False
                
                # 해당 구성 행 제외하고 다시 쓰기
                new_rows = [rows[0], rows[1]]  # 헤더 2행은 유지
                
                for row in rows[2:]:
                    if not row or not row[0]:
                        continue
                    
                    try:
                        # 행의 첫 번째 열도 같은 방식으로 변환 시도
                        row_idx = row[0]
                        if row_idx.replace('.', '', 1).isdigit() and '.' in row_idx:
                            row_idx = int(float(row_idx))
                        else:
                            row_idx = int(row_idx)
                            
                        if row_idx != config_idx:
                            new_rows.append(row)
                    except ValueError:
                        # 숫자로 변환할 수 없는 경우 행 유지
                        new_rows.append(row)
                
                # CSV 파일 다시 쓰기
                with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(new_rows)
                
                return True
        
        except Exception as e:
            print(f"구성 삭제 오류: {e}")
            return False
    
    def rename_config(self, old_name, new_name):
        """
        이 형식에서는 이름 변경 대신 인덱스 변경만 가능
        
        Args:
            old_name (str): 기존 구성 인덱스
            new_name (str): 새 구성 인덱스
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 새 이름이 숫자인지 확인
            try:
                # 점이 포함된 경우 부동 소수점으로 간주하고 정수로 변환
                if new_name.replace('.', '', 1).isdigit() and '.' in new_name:
                    new_idx = int(float(new_name))
                else:
                    new_idx = int(new_name)
            except ValueError:
                print(f"새 구성 번호 변환 오류: {new_name}은(는) 유효한 숫자가 아닙니다.")
                return False
            
            # 기존 이름이 숫자인지 확인
            try:
                # 점이 포함된 경우 부동 소수점으로 간주하고 정수로 변환
                if old_name.replace('.', '', 1).isdigit() and '.' in old_name:
                    old_idx = int(float(old_name))
                else:
                    old_idx = int(old_name)
            except ValueError:
                print(f"기존 구성 번호 변환 오류: {old_name}은(는) 유효한 숫자가 아닙니다.")
                return False
            
            if not os.path.exists(self.csv_file_path):
                return False
            
            # CSV 파일 읽기
            with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 3:  # 최소 3행 필요
                    return False
                
                # 인덱스 변경
                changed = False
                for i in range(2, len(rows)):
                    if not rows[i] or not rows[i][0]:
                        continue
                        
                    try:
                        # 행의 첫 번째 열 값을 가져와 변환
                        row_idx_str = rows[i][0]
                        # 점이 포함된 경우 부동 소수점으로 간주하고 정수로 변환
                        if row_idx_str.replace('.', '', 1).isdigit() and '.' in row_idx_str:
                            row_idx = int(float(row_idx_str))
                        else:
                            row_idx = int(row_idx_str)
                            
                        if row_idx == old_idx:
                            rows[i][0] = str(new_idx)
                            changed = True
                    except ValueError:
                        # 숫자로 변환할 수 없는 경우 무시
                        continue
                
                # CSV 파일 다시 쓰기
                if changed:
                    with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    return True
                else:
                    return False
        
        except Exception as e:
            print(f"구성 이름 변경 오류: {e}")
            return False
    
    def save_config_to_csv(self, checkboxes_config, env_variables, custom_configs, pre_commands, config_name=None):
        """
        구성을 CSV 파일에 저장
        
        Args:
            checkboxes_config (dict): 체크박스 구성 데이터
            env_variables (list): 환경 변수 설정
            custom_configs (list): 사용자 정의 설정
            pre_commands (list): 사전 실행 명령어
            config_name (str, optional): 구성 이름. 기본값은 None (자동 번호 부여)
            
        Returns:
            str: 저장된 구성 인덱스
        """
        try:
            print(f"저장할 체크박스 구성: {checkboxes_config}")
            
            # CSV 데이터 준비
            section_row = ["Index"]
            param_row = ["Name" if config_name else ""]
            data_row = []
            
            # 다음 사용할 인덱스 구하기
            next_idx = self._get_next_config_idx()
            
            # 인덱스 및 이름 설정
            if config_name:
                data_row.append(config_name)
            else:
                data_row.append(str(next_idx))
            
            # 각 섹션별 데이터 처리
            for section, params in checkboxes_config.items():
                # 섹션 처리
                for param, config in params.items():
                    section_row.append(section)
                    param_row.append(param)
                    
                    # 체크박스 상태에 따라 값 설정
                    if config.get('checked', False):
                        data_row.append("1")  # 체크된 경우 1로 저장
                    else:
                        data_row.append("0")  # 체크 해제된 경우 0으로 저장
            
            # 환경 변수 처리
            for env_var in env_variables:
                if env_var.get('enabled', True):  # 활성화된 환경 변수만 저장
                    section_row.append("ENV_VARIABLES")
                    param_row.append(env_var['name'])
                    data_row.append(env_var['value'])
            
            # 사용자 정의 설정 처리
            for config in custom_configs:
                if config.get('enabled', True):  # 활성화된 설정만 저장
                    section_row.append("CUSTOM_CONFIG")
                    param_row.append(config['param'])
                    data_row.append(config['value'])
            
            # 사전 실행 명령어 처리
            for i, cmd in enumerate(pre_commands):
                if cmd and cmd.get('enabled', True):  # 활성화된 명령어만 저장
                    section_row.append("PRE_COMMANDS")
                    param_row.append(f"cmd_{i+1}")
                    data_row.append(cmd['command'])
            
            # 기존 CSV 파일 읽기 (없으면 새로 생성)
            rows = []
            if os.path.exists(self.csv_file_path):
                with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    if len(rows) >= 2:  # 헤더가 2줄 이상 있으면
                        existing_section_row = rows[0]
                        existing_param_row = rows[1]
                        
                        # 섹션과 파라미터 행 업데이트 (기존 값과 병합)
                        for i in range(1, len(section_row)):
                            if i < len(existing_section_row):
                                # 기존 위치에 덮어쓰기 (이미 존재하면)
                                existing_section_row[i] = section_row[i]
                                existing_param_row[i] = param_row[i]
                            else:
                                # 새로운 위치에 추가 (존재하지 않으면)
                                existing_section_row.append(section_row[i])
                                existing_param_row.append(param_row[i])
                                
                        section_row = existing_section_row
                        param_row = existing_param_row
            
            # 행 길이 맞추기
            max_len = max(len(section_row), len(param_row), len(data_row))
            section_row.extend([''] * (max_len - len(section_row)))
            param_row.extend([''] * (max_len - len(param_row)))
            data_row.extend([''] * (max_len - len(data_row)))
            
            # 저장할 행 생성
            new_rows = [section_row, param_row]
            
            # 기존 데이터 행 가져오기 (있다면)
            if len(rows) > 2:
                new_rows.extend(rows[2:])
            
            # 기존에 이미 같은 이름의 구성이 있는지 검사
            config_idx = data_row[0]
            replaced = False
            
            for i in range(2, len(new_rows)):
                if len(new_rows[i]) > 0 and str(new_rows[i][0]) == str(config_idx):
                    # 기존 구성 값 교체
                    new_rows[i] = data_row
                    replaced = True
                    break
            
            # 새 구성 추가 (기존에 없는 경우)
            if not replaced:
                new_rows.append(data_row)
            
            # CSV 파일 작성
            with open(self.csv_file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(new_rows)
            
            return config_idx
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"구성 저장 오류: {e}")
            return None
    
    def _get_next_config_idx(self):
        """
        CSV 파일에서 다음에 사용할 구성 인덱스를 가져옴
        
        Returns:
            int: 다음 사용할 구성 인덱스
        """
        try:
            if not os.path.exists(self.csv_file_path):
                return 1
                
            # CSV 파일 읽기
            with open(self.csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 3:  # 최소 3행 필요 (헤더 2행 + 데이터 1행)
                    return 1
                
                # 인덱스 추출 (세 번째 행부터)
                indices = []
                for row in rows[2:]:
                    if row and row[0]:
                        try:
                            # 숫자로 변환 시도
                            if row[0].replace('.', '', 1).isdigit():
                                if '.' in row[0]:
                                    indices.append(int(float(row[0])))
                                else:
                                    indices.append(int(row[0]))
                        except (ValueError, TypeError):
                            # 숫자가 아닌 경우 무시
                            pass
                
                # 가장 큰 인덱스 + 1 반환
                if indices:
                    return max(indices) + 1
                else:
                    return 1
                    
        except Exception as e:
            print(f"다음 구성 인덱스 가져오기 오류: {e}")
            return 1 