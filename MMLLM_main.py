from MMLLM_Common import InputDataset
from MMLLM_Gemini import MMLLM_Gemini

if __name__ == "__main__":
    gemini_exp = MMLLM_Gemini('API-KEY')
    gemini_exp.phase1_brand_identification(InputDataset.MMLLM_Benign)
    gemini_exp.phase2_phishing_classification(InputDataset.MMLLM_Benign)
    gemini_exp.phase1_brand_identification(InputDataset.MMLLM_Phishing)
    gemini_exp.phase2_phishing_classification(InputDataset.MMLLM_Phishing)