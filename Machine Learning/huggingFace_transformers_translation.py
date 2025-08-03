from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# # 입력 문장
# text = "I've been waiting for a HuggingFace course my whole life."

# # 감성 분석 실행
# result = classifier(text)

# # 결과 출력
# print(result)

# 입력 문장
text2 = ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!", "Sometimes, I do workout."]

# 감성 분석 실행
result2 = classifier(text2)

# 결과 출력
print(result2)


from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library.",
    candidate_labels=["education", "politics", "business"],
)

print(result)


from transformers import pipeline

generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")

print(result)


from transformers import pipeline

generator = pipeline("text-generation", model = 'gpt2')
result = generator("In this course, we will teach you how to", 
                   max_length=50, 
                   num_return_sequences=3, 
                   temperature=0.7, 
                   top_p=0.9)

print(result)