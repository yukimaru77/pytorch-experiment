# Nvidia CUDAの12.1.0-devel-ubuntu18.04をベースイメージにする
FROM nvidia/cuda:12.1.0-devel-ubuntu18.04

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
sudo \
wget \
graphviz

# 作業ディレクトリを/optに指定
WORKDIR /opt

# Anacondaをダウンロードし/opt/anaconda3にインストール
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh && \
sh Anaconda3-2022.10-Linux-x86_64.sh -b -p /opt/anaconda3 && \
rm -f Anaconda3-2022.10-Linux-x86_64.sh

# PATHを設定
ENV PATH /opt/anaconda3/bin:$PATH

# 必要なPythonパッケージをpipでインストール
RUN pip install --upgrade pip && pip install \
torch torchvision torchaudio torchviz torchinfo graphviz japanize-matplotlib


# 拡張機能を入れようとしたら依存関係で色々エラーが出たのでバージョン管理したライブラリ群
RUN pip install pyqt5==5.15.4 pyqtwebengine==5.15.4 \
ruamel-yaml \
clyent==1.2.1 \
nbformat==5.4.0 \
nbformat==5.4.0

# JupyterLabの日本語化
# 変数の中身を確認するためのライブラリ
# タブ補完を提供するためのライブラリ
# 対話的なWebベースのウィジェットを作成するためのライブラリ
# LSP(Language Server Protocol)を実装するためのライブラリ
# LSP(Language Server Protocol)を実装するためのライブラリ
RUN pip install jupyterlab-language-pack-ja-JP \ 
lckr-jupyterlab-variableinspector \ 
jupyterlab_tabnine \ 
ipywidgets \ 
jupyterlab-lsp \ 
'python-lsp-server[all]' 

#pip install jupyterlab-language-pack-ja-JP lckr-jupyterlab-variableinspector jupyterlab_nvdashboard ipywidgets black yapf isort jupyterlab_code_formatter jupyterlab_tabnine jupyterlab-lsp 'python-lsp-server[all]' pprintpp jupyterlab-spreadsheet-editor ipympl opencv-python
