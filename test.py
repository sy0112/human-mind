import pandas as pd
import time
from openai import OpenAI
from docx import Document

# 🔑 OpenAI API 키 입력
client = OpenAI(api_key="")

# CSV 파일 불러오기
file_paths = ["225.csv"]
all_data = pd.concat([pd.read_csv(path) for path in file_paths], ignore_index=True)

# question_id에서 숫자만 추출
all_data['question_num'] = all_data['question_id'].str.extract('(\d+)').astype(int)

def analyze_section(qa_pairs):
    """
    섹션 단위로 분석 요약 (GPT가 섹션 주제를 스스로 도출)
    """
    formatted = "\n".join([
        f"Q: {q}\nA: {a}" for q, a in qa_pairs
    ])

    prompt = f"""
당신은 인터뷰 데이터를 분석하는 전문가입니다.
다음은 하나의 섹션에 속하는 질문과 답변 모음입니다.
아래 데이터를 바탕으로 먼저 이 질문들이 다루는 주제를 스스로 정하고,
그 주제를 섹션 제목으로 제시하세요.  
그 후, 응답자들의 공통적인 패턴, 주요 키워드, 긍정적/부정적 인식 등을 종합하여 
약 5~8문장 정도의 줄글 요약을 작성하세요.

[데이터]
{formatted}

[출력 형식]
- 섹션 제목
- 줄글 요약
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        return None

# Word 문서 생성
doc = Document()
doc.add_heading("포토이즘 인터뷰 분석 보고서", level=0)

# session별 그룹핑 (session 컬럼 있다고 가정)
if "session" not in all_data.columns:
    raise ValueError("CSV에 'session' 컬럼이 필요합니다.")

for session_id, group in all_data.groupby("session"):
    qa_pairs = list(zip(group['question_text'], group['answer']))

    if not qa_pairs:
        continue

    # GPT 분석
    result = analyze_section(qa_pairs)
    if result:
        doc.add_heading(f"세션 {session_id}", level=1)
        doc.add_paragraph(result)

    time.sleep(2)  # 안전한 API 호출 간격

# Word 파일 저장
output_path = "interview_session_report2.docx"
doc.save(output_path)

print(f"✅ 분석 완료! Word 보고서가 '{output_path}' 파일로 생성되었습니다.")
