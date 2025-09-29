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

# 섹션 매핑 (4문항씩)
section_map = {
    1: "포토이즘 이용 경험",
    2: "포토이즘 사용 동기",
    3: "포토이즘 개선 사항",
    4: "포토이즘 전반적인 인식"
}

def analyze_section(section_name, qa_pairs):
    """
    섹션 단위로 분석 요약 (줄글)
    """
    formatted = "\n".join([
        f"Q: {q}\nA: {a}" for q, a in qa_pairs
    ])

    prompt = f"""
당신은 인터뷰 데이터를 분석하는 전문가입니다.
다음은 '{section_name}' 섹션의 질문과 답변 모음입니다.
응답자들의 공통적인 패턴, 주요 키워드, 긍정적/부정적 인식 등을 종합하여 
섹션별로 하나의 보고서 단락(줄글)로 작성하세요.

[데이터]
{formatted}

[보고서 작성 형식]
- 섹션 제목
- 줄글 요약 (약 5~8문장)
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

# 섹션별 분석 추가
for section_idx, section_name in section_map.items():
    # 해당 섹션에 속하는 질문 필터링
    section_rows = all_data[
        (all_data['question_num'] > (section_idx-1)*4) &
        (all_data['question_num'] <= section_idx*4)
    ]
    qa_pairs = list(zip(section_rows['question_text'], section_rows['answer']))

    if not qa_pairs:
        continue

    # GPT 분석
    result = analyze_section(section_name, qa_pairs)
    if result:
        doc.add_heading(section_name, level=1)
        doc.add_paragraph(result)

    time.sleep(2)  # 안전한 API 호출 간격

# Word 파일 저장
output_path = "interview_section_report.docx"
doc.save(output_path)

print(f"✅ 분석 완료! Word 보고서가 '{output_path}' 파일로 생성되었습니다.")
