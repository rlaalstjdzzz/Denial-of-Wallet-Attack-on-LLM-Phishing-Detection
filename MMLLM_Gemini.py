import json
import glob
import time
from tqdm import tqdm
import os
from MMLLM_Common import *
from google.api_core.exceptions import ResourceExhausted, InternalServerError, TooManyRequests, ServerError, BadRequest
from google import genai
from google.genai import types

class MMLLM_Gemini:
    def __init__(self, str_api_key:str):
        self.str_api_key = str_api_key
        self.str_phase1_system_msg: dict        
        self.str_phase1_res_format: dict
        self.str_phase2_system_msg: dict        
        
        self.client = genai.Client(api_key=str_api_key)

        self.str_model = "gemini-2.5-flash-lite"
        return    

    def load_prompt_text(self, input_mode:InputMode):        
        str_phase1_prompt_path = dict_system_prompt_path.get(input_mode)
        assert str_phase1_prompt_path is not None, f"Unknown Input mode {input_mode}"

        with open(str_phase1_prompt_path, encoding='utf-8') as f_read:
            str_phase1_system_prompt = f_read.read()        
            self.str_phase1_system_msg = str_phase1_system_prompt

        str_phase1_response_prompt_path = dict_response_prompt_path.get(input_mode)
        assert str_phase1_response_prompt_path is not None, f"Unknown Input mode {input_mode}"

        with open(str_phase1_response_prompt_path, encoding='utf-8') as f_read:
            str_res_format = f_read.read()
            self.str_phase1_res_format = '\n\n' + str_res_format

        str_phase2_system_prompt_path = dict_response_prompt_path.get(Phase2Mode.Phase2)
        assert str_phase2_system_prompt_path is not None, f"Unknown Input mode {Phase2Mode.Phase2}"

        with open(str_phase2_system_prompt_path, encoding='utf-8') as f_read:
            str_phase2_system_prompt = f_read.read()
            self.str_phase2_system_msg = str_phase2_system_prompt
        return

    def create_identification_prompt(self, input_mode:InputMode, encoded_image, html_content):                
        # 최신 SDK도 리스트 형태의 contents([텍스트, 이미지, 텍스트])를 자동으로 처리합니다.
        str_resource_msg = "Here are the provided resources: "
        list_phase1_system_msg = [self.str_phase1_system_msg, str_resource_msg]

        if input_mode == InputMode.SS:
           list_phase1_system_msg.append(encoded_image)
        elif input_mode == InputMode.HTML:
           list_phase1_system_msg.append(html_content)
        elif input_mode == InputMode.BOTH:
            list_phase1_system_msg.append(html_content)
            list_phase1_system_msg.append(encoded_image)
        
        list_phase1_system_msg.append(self.str_phase1_res_format)

        return list_phase1_system_msg

    def create_brandcheck_prompt(self, str_groundtruth:str, str_prediction:str):                
        str_phase2_data = f"Ground Truth: \"{str_groundtruth}\"\n\"Prediction:\"{str_prediction}\""
        list_phase2_system_msg = [self.str_phase2_system_msg, str_phase2_data]
        return list_phase2_system_msg
    

    def phase1_brand_identification(self, input_dataset:InputDataset):
        str_dataset = input_dataset.value
        list_data_dir = glob.glob(f'{str_input_dir_base}/{str_dataset}/*/*/')
        list_data_dir.sort()

        for str_data_dir in tqdm(list_data_dir, desc=f'{str_dataset}'):
            str_data_dir = str_data_dir.replace('\\', '/')
            list_prop = str_data_dir.split('/')
            str_ss_path = str_data_dir + '/screenshot_aft.png'
            str_html_path = str_data_dir + '/add_info.json'

            str_hash = list_prop[-2]
            str_brand = list_prop[-3]

            if not os.path.exists(str_ss_path) or not os.path.exists(str_html_path):
                continue
        
            image = None
            try:
                image = crop_encode_image_PIL(str_ss_path)
            except Exception as e:
                print(f'[Warning] Image processing failed for {str_data_dir}. Skipping. Error: {e}')
                continue
            
            with open(str_html_path, encoding='utf-8') as f_read:
                dict_html_info = json.load(f_read)
                # original_html_content = dict_html_info['html_brand_info']
                str_html_info = dict_html_info['html_brand_info']
                str_url = dict_html_info['Url']

            for input_mode in InputMode:
                self.load_prompt_text(input_mode) 

                str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Gemini', input_mode.value, str_brand)
                if not os.path.exists(str_output_dir):
                    os.makedirs(str_output_dir)

                str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                
                if os.path.exists(str_output_file_path):
                    tqdm.write(f'[Skipping] Output already exists for hash: {str_hash}')
                    continue
                
                list_model_prompt = self.create_identification_prompt(input_mode, image, str_html_info)
                
                # --- [변경 3] API 호출 및 Thinking Config 적용 ---
                try:
                    response = self.client.models.generate_content(
                        model=self.str_model,
                        contents=list_model_prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=-1)
                        )
                    )
                    
                except ValueError:
                    print(f'[Warning] Value Error for {str_hash}. Skipping.')
                    continue
                except ResourceExhausted:
                    print(f'[Warning] {str_hash} made Quota error')
                    time.sleep(60)            
                    continue
                except InternalServerError:                    
                    print(f'[Warning] {str_hash} InternalServerError')
                    time.sleep(60)            
                    continue
                except BadRequest:
                    print(f'[Warning] {str_hash} BadRequest')
                    continue
                except Exception as e: # 기타 모든 에러 잡기 (새 SDK 마이그레이션 시 안전장치)
                    print(f'[Warning] Unhandled error for {str_hash}: {e}')
                    continue
                
                if not hasattr(response, 'text') or not response.text:
                     # 텍스트 생성이 안 된 경우 (Block 등)
                     dict_res_data = format_model_response(str_hash, '', True, False)
                else:                    
                    str_res = response.text                    
                    try:
                        dict_res_data = format_model_response(str_hash, str_res)                
                    except:
                        dict_res_data = format_model_response(str_hash, 'Safety Error', False, True)                        
                        pass
                    dict_res_data['Url'] = str_url

                # --- 토큰 사용량 추출 ---
                if response.usage_metadata:
                    dict_res_data['prompt_token_count'] = response.usage_metadata.prompt_token_count
                    
                    if hasattr(response.usage_metadata, 'thoughts_token_count'):
                         dict_res_data['thoughts_token_count'] = response.usage_metadata.thoughts_token_count
                    else:
                        dict_res_data['thoughts_token_count'] = 0
                        
                    dict_res_data['candidates_token_count'] = response.usage_metadata.candidates_token_count
                    dict_res_data['total_token_count'] = response.usage_metadata.total_token_count    
                
                str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                with open(str_output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(dict_res_data, f, indent=4, ensure_ascii=False)
        return

    def phase2_phishing_classification(self, input_dataset:InputDataset):
        str_dataset = input_dataset.value
        
        str_output_summary_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase2_Gemini')
        os.makedirs(str_output_summary_dir, exist_ok=True)
        
        str_output_summary_path = os.path.join(str_output_summary_dir, "Phase2_Res_Summary.csv")
        
        if not os.path.exists(str_output_summary_path):
            with open(str_output_summary_path, 'w', encoding='utf-8') as f_summary:
                str_phase2_res_summary_hdr = f'Dataset,InputMode,Brand,Hash,Phase1Pred,Phase2Matched\n'
                f_summary.write(str_phase2_res_summary_hdr)

        search_all_brands_path = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Gemini', '*', '*')
    
        # glob.glob으로 모든 브랜드 경로를 찾고, 경로의 마지막 요소(브랜드 이름)만 추출하여 set으로 만듭니다.
        all_brand_paths = glob.glob(search_all_brands_path)
        
        # 경로에서 브랜드 이름만 추출하여 중복을 제거한 고유 목록(set)을 만듭니다.
        unique_brands = set()
        for path in all_brand_paths:
            # 경로가 '.../Phase1_Gemini/mode/brand' 형태임을 가정하고 브랜드 이름을 추출합니다.
            path = os.path.normpath(path)
            parts = path.split(os.sep)
            
            # 마지막 두 요소가 mode와 brand여야 합니다.
            if len(parts) >= 2:
                brand_name = parts[-1]
                input_mode_dir = parts[-2]
                
                # mode 디렉토리 이름이 InputMode의 값(html, ss, both)인지 확인하여 불필요한 폴더를 거릅니다.
                if input_mode_dir in [m.value for m in InputMode]:
                    unique_brands.add(brand_name)
        
        # 고유 브랜드 목록을 정렬하여 루프에 사용합니다.
        list_brands = sorted(list(unique_brands))
        
        if not list_brands:
            print("경고: Phase 1 결과에서 유효한 브랜드 폴더가 발견되지 않았습니다. Phase 1 실행을 확인해주세요.")
            return

        for str_brand_dir in tqdm(list_brands, desc=f'Phase2 Processing for {str_dataset}'):
            str_brand = os.path.basename(os.path.normpath(str_brand_dir))
            for input_mode in InputMode:
                str_input_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Gemini', input_mode.value, str_brand)
                list_input_path = glob.glob(f'{str_input_dir}/*.json')
                list_input_path.sort()
                
                str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase2_Gemini', input_mode.value, str_brand)
                os.makedirs(str_output_dir, exist_ok=True)

                for str_input_path in list_input_path:
                    str_hash = os.path.basename(str_input_path).replace('.json', '')

                    str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json") 
                    
                    if os.path.exists(str_output_file_path):
                        tqdm.write(f'[Skipping Phase 2] Output already exists for hash: {str_hash} in {input_mode.value}')
                        continue
                    
                    try:
                        with open(str_input_path, encoding='utf-8') as f_read:
                            dict_data = json.load(f_read)
                            str_phase1_pred = dict_data['Brand']
                            b_phase1_error = dict_data['Error']
                            str_url = dict_data['Url']
                    except Exception as e:
                        print(f'[Warning] Broken Phase 1 result: {str_input_path}. Error: {e}')
                        continue
                    
                    if b_phase1_error:
                        continue

                    # Gemini
                    list_model_prompt = self.create_brandcheck_prompt(str_url, str_phase1_pred)
                    
                    # --- Phase 2 API 호출에도 Config 적용 ---
                    try:
                        response = self.client.models.generate_content(
                            model=self.str_model,
                            contents=list_model_prompt,
                            config=types.GenerateContentConfig(
                                # thinking_config=types.ThinkingConfig(thinking_budget=-1)
                            )
                        )
                    except ResourceExhausted:
                        print(f'[Warning] {str_hash} made Quota error')
                        time.sleep(60)
                        continue
                    except InternalServerError:
                        print(f'[Warning] {str_hash} InternalServerError')
                        time.sleep(60)
                        continue
                    except BadRequest:
                        print(f'[Warning] {str_hash} BadRequest')
                        continue
                    except Exception as e:
                        print(f'[Warning] Phase 2 Error: {e}')
                        continue
                    
                    if not hasattr(response, 'text') or not response.text:
                        dict_res_data = format_phase2_response('', False, True)
                    else:
                        str_res = response.text
                        try:
                            dict_res_data = format_phase2_response(str_res, False, False)
                        except:
                            dict_res_data = format_phase2_response(str_res, False, True)
                            pass
                    
                    if response.usage_metadata:
                        dict_res_data['prompt_token_count'] = response.usage_metadata.prompt_token_count
                        # Token field name check
                        if hasattr(response.usage_metadata, 'thoughts_token_count'):
                            dict_res_data['thoughts_token_count'] = response.usage_metadata.thoughts_token_count
                        else:
                            dict_res_data['thoughts_token_count'] = 0
                        dict_res_data['candidates_token_count'] = response.usage_metadata.candidates_token_count
                        dict_res_data['total_token_count'] = response.usage_metadata.total_token_count
                    
                    str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                    with open(str_output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(dict_res_data, f, indent=4, ensure_ascii=False)

                    str_phase2_res_summary = f'{str_dataset},{input_mode.value},{str_brand},{str_hash},"{str_phase1_pred}",{dict_res_data["BrandMatched"]}\n'
                    with open(str_output_summary_path, 'a', encoding='utf-8') as f_summary:
                        f_summary.write(str_phase2_res_summary)
        return