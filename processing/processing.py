import pandas as pd
import openai
import time
import json
import os
import glob
import re
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")

input_folder = "input"
output_folder = "output"
analysis_folder = "analysis"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(analysis_folder, exist_ok=True)

# 설정
MAX_RETRIES = 3
SLEEP_BETWEEN_CHUNKS = 1
MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 1000
ENCODER = tiktoken.encoding_for_model(MODEL_NAME)

def count_tokens(text):
    return len(ENCODER.encode(str(text)))

def extract_json(text):
    """문자열에서 JSON 배열만 안전하게 추출"""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        print("JSON 배열 추출 실패, 원본 반환")
        return None
    json_text = match.group(0)
    
    # 작은따옴표 -> 큰따옴표
    json_text = json_text.replace("'", '"')
    # 콜론 뒤 공백 제거
    json_text = re.sub(r'":\s*"', '":"', json_text)
    json_text = json_text.replace("\n", "")
    
    try:
        data = json.loads(json_text)
        if not isinstance(data, list):
            print("JSON이 리스트가 아님, 원본 반환")
            return None
        return data
    except Exception as e:
        print("JSON 파싱 실패:", e)
        return None

def ai_preprocess_chunk(chunk, chunk_index, total_chunks):
    """청크 전처리, 실패 시 재시도, 길이 불일치 시 재귀 처리"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answers_str = json.dumps(chunk, ensure_ascii=False)
            prompt = (
                "다음 설문 답변 리스트를 데이터 분석용으로 전처리하세요.\n"
                "조건:\n"
                "1. 일반 영어 단어는 자연스러운 한국어로 번역 (예: complicated -> 복잡한)\n"
                "2. 전문 용어, 약어, 브랜드명, 제품 이름, 기술 용어, UI/UX 관련 단어 등은 그대로 유지\n"
                "3. 의미 없는 감탄사, '음', '어', '...', 중복/불필요 표현 제거\n"
                "4. 문장 단순화, 핵심 의미만 유지\n"
                "5. 연속 공백과 특수문자 제거\n"
                "6. CSV용 한 줄 문자열(answer_processed 컬럼용)으로 출력\n"
                "7. 절대 원문 그대로 반환하지 마세요\n"
                "8. 출력은 반드시 JSON 배열 표준(JSON RFC) 형식, 큰따옴표(\")만 사용\n"
                "9. 추가 설명, 작은따옴표 등 다른 문구 없이 순수 JSON 배열만 반환\n\n"
                "입력 리스트: " + answers_str
            )

            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "당신은 데이터 전처리 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            text = response.choices[0].message.content.strip()
            data = extract_json(text)

            if data is None:
                # 안전하게 원본 반환
                return chunk

            chunk_processed = []
            for item in data:
                if isinstance(item, dict) and 'answer_processed' in item:
                    chunk_processed.append(item['answer_processed'])
                else:
                    # 구조가 올바르지 않으면 원본 사용
                    chunk_processed.append(str(item))

            # 길이 불일치 시 재귀 분할
            if len(chunk_processed) != len(chunk):
                if len(chunk) > 1:
                    mid = len(chunk) // 2
                    first_half = ai_preprocess_chunk(chunk[:mid], chunk_index, total_chunks)
                    second_half = ai_preprocess_chunk(chunk[mid:], chunk_index, total_chunks)
                    return first_half + second_half
                else:
                    print(f"[청크 {chunk_index}/{total_chunks}] 단일 답변 처리 실패, 원본 사용")
                    return chunk

            return chunk_processed

        except Exception as e:
            print(f"[청크 {chunk_index}/{total_chunks}][시도 {attempt}/{MAX_RETRIES}] 실패: {e}")
            time.sleep(SLEEP_BETWEEN_CHUNKS)

    print(f"[청크 {chunk_index}/{total_chunks}] 모든 시도 실패, 원본 사용")
    return list(chunk)

def split_into_token_safe_chunks(answers):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for ans in answers:
        ans_tokens = count_tokens(ans)
        if current_tokens + ans_tokens > MAX_TOKENS and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [ans]
            current_tokens = ans_tokens
        else:
            current_chunk.append(ans)
            current_tokens += ans_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def ai_preprocess_batch(answers):
    processed_answers = []
    chunks = split_into_token_safe_chunks(answers)
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        print(f"처리 중: 청크 {i}/{total_chunks} ({len(chunk)}개 답변)")
        chunk_processed = ai_preprocess_chunk(chunk, i, total_chunks)
        processed_answers.extend(chunk_processed)
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    return processed_answers

# CSV 처리
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

for file_path in csv_files:
    print(f"\n=== 처리 시작: {file_path} ===")
    df = pd.read_csv(file_path)

    if 'answer' not in df.columns:
        print(f"경고: 'answer' 컬럼 없음, 스킵 {file_path}")
        continue

    all_answers = df['answer'].tolist()
    processed_answers = ai_preprocess_batch(all_answers)
    df['answer_processed'] = processed_answers

    # 전처리 완료 저장
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_processed.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"전처리 저장 완료: {output_path}")

    # 분석용 CSV 저장 (question + answer_processed)
    if 'question' in df.columns:
        analysis_df = df[['question', 'answer_processed']].copy()
        analysis_path = os.path.join(analysis_folder, f"{base_name}_analysis.csv")
        analysis_df.to_csv(analysis_path, index=False, encoding='utf-8-sig')
        print(f"분석용 저장 완료: {analysis_path}")

print("\n모든 파일 전처리 및 분석용 CSV 생성 완료!")
