pip install --editable ./
pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
pip install sentencepiece==0.1.96
pip install numpy==1.19.5
pip install faiss-gpu
conda install -c conda-forge cudatoolkit-dev
# Please revise the following line according to your torch version and cuda version
# The supported torch and cuda version could be found in https://github.com/rusty1s/pytorch_scatter
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.1+cu101.html
pip install sacrebleu==1.5.1
pip install SoundFile==0.10.3.post1