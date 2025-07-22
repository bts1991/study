from transformers import MarianMTModel, MarianTokenizer

# 영어 → 한국어: Helsinki-NLP/opus-mt-en-ko
# 한국어 → 영어: Helsinki-NLP/opus-mt-ko-en
# 자동 언어 감지 + 번역: facebook/nllb-200-distilled-600M

# 사용할 모델명 지정
model_name = "Helsinki-NLP/opus-mt-en-ko"  # 영어 → 한국어
# model_name = "Helsinki-NLP/opus-mt-ko-en" # 한국어 → 영어

# 토크나이저와 모델 불러오기
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 번역할 문장
src_text = "I am very happy to learn machine learning with Hugging Face!"

# 토큰화 및 모델 입력
tokens = tokenizer.prepare_seq2seq_batch([src_text], return_tensors="pt")

# 번역 실행
translated = model.generate(**tokens)

# 출력 해석
result = tokenizer.decode(translated[0], skip_special_tokens=True)
print("번역 결과:", result)
