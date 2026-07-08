import os

# 1. 제외할 무거운 폴더 및 데이터 파일 설정
IGNORE_DIRS = {'.git', 'venv', 'env', 'node_modules', '__pycache__', 'dist', 'build'}
# 2. 분석에 필요한 소스코드 확장자 설정 (필요시 추가 가능)
ALLOWED_EXTENSIONS = {'.py', '.html', '.css', '.js', '.json', '.md', '.txt', '.sh'}

with open('project_code_summary.txt', 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('.'):
        # 무시할 디렉토리 필터링
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ALLOWED_EXTENSIONS:
                filepath = os.path.join(root, file)
                
                # 파일 구분선 및 경로 작성
                outfile.write(f"\n{'='*60}\n")
                outfile.write(f"File Path: {filepath}\n")
                outfile.write(f"{'='*60}\n")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read() + "\n")
                except Exception as e:
                    outfile.write(f"Error reading file (maybe not a text file): {e}\n")

print("✅ 'project_code_summary.txt' 파일 생성이 완료되었습니다! 이 파일을 업로드해 주십시오.")