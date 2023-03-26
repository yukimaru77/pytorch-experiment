# jupyterLab+pytorch+Docker

JupyterLabを触ったことがなかったため、まずはとりあえずdocker上で動かしたい。それだけのためのディレクトリです。

## 構成要素

* Dockerfile
* docker-compose.yml
* workディレクトリ

workディレクトリとはコンテナにマウントされるディレクトリです。

## 実行方法

docker-compose.ymlがあるディレクトリで以下のコマンドを実行する。
```
docker compose up
```
なお、docker-compose.ymlを変更した場合は、変更を更新するために
```
docker compose up --build
```
とする。