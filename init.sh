pip uninstall --yes torch torchvision torchtext torchaudio 
yes | pip install torch==1.8.1
yes | pip install transformers==4.15.0 tokenizers==0.10.3 jieba rouge tqdm pandas bert4torch torchvision==0.9.1
pip show torch