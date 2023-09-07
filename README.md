# ZX-inference
Inference scripts for text2sql 

[Step 0: Git repo clone & model checkpoint download]
```
git clone https://github.com/deokhk/ZX-inference.git
cd ZX-inference
mkdir models 
```
Link (https://drive.google.com/file/d/1P4YRYr0YKNSza7t9jqCjK9BVg5SiLaD5/view?usp=sharing) 에서 text2sql_mt0_ckpt.zip download한 후, models directiory에 저장
```
gdown "https://drive.google.com/uc?id=1P4YRYr0YKNSza7t9jqCjK9BVg5SiLaD5"
unzip -d text2sql_mt0_ckpt text2sql_mt0_ckpt.zip
```
models directory에서 unzip

[Step 1: 환경 설치]
```
conda create -n ZX python=3.8.5
conda activate ZX 
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
python nltk_downloader.py
```
[Step 2: 테스트]

```
sh ./scripts/inference.sh    <- 해당 command는 ZX-inference의 root directory에서 실행해야 함.
```
