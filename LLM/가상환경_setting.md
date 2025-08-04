가상환경 생성
python -m venv myenv


가상환경 활성화
.\myenv\Scripts\Activate.ps1


보안 정책 오류가 나올 수도 있어요. 그 경우
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process


pip 업그레이드
python.exe -m pip install --upgrade pip


NumPy를 다운그레이드
pip install numpy==1.23.5


gym 설치
pip install gym


matplotlib 설치
pip install matplotlib