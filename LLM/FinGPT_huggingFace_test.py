from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# FinGPT 모델 로드
model_name = "AI4Finance/FinGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 텍스트 생성 파이프라인 설정
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 프롬프트 입력
prompt = "What is the impact of interest rate hikes on stock market sentiment?"

# 결과 생성
outputs = generator(prompt, max_length=200, do_sample=True, temperature=0.7)

# 출력
print(outputs[0]['generated_text'])