import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from tqdm import tqdm
import inspect
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Any, Tuple, Union, List, Dict, Optional

# def generate_boxed_string(func):
#     def wrapper(*args, **kwargs):
#         width = 50
#         border_line = "━" * (width + 2)
#         padding_line = " " * (width + 2)
#         box_lines = [f"┏{border_line}┓"]
#         box_lines.append(f"┃{padding_line}┃")
#         print("\n".join(box_lines))
#         func(*args, **kwargs)
#         box_lines = [f"┃{padding_line}┃"]
#         box_lines.append(f"┗{border_line}┛")
#         print("\n".join(box_lines))
#     return wrapper

def _wrap_text_by_width(text: str, width: int = 60) -> list[str]: #50文字ごとまたは改行コードごとに区切ってリストを返す。公開する気はないのでアンダーバーを先頭につけている。

    result = []
    current_line = ""
    current_width = 0

    for ch in text:
        ch_width = 2 if ord(ch) >= 128 else 1 #日本語とか全角は2文字としてカウントする。
        if (current_width + ch_width > width):
            result.append(current_line)
            current_line = ""
            current_width = 0

        if (ch == "\n"): #改行文字があったら、その行を追加して次の行に移る。
            result.append(current_line)
            current_line = ""
            current_width = 0
        else:
            current_line += ch

        current_width += ch_width

    if current_line: #最後の行は50文字以下でも追加する。
        result.append(current_line)

    return result

# 公開する関数の一覧や説明を表示する関数
def show_function(functions=None):
    width = 50
    border_line = "━" * (width + 2)
    padding_line = " " * (width + 2)
    all_functions = [torch_seed, fit, evaluate_history, show_images_labels]
    if functions is None:
        functions = all_functions
    
    if(type(functions)!=list):
        functions = [functions]
    
    for func in functions:
        if func not in all_functions:
            try:
                print(f"{func.__name__} はこのライブラリに存在しません。")
            except:
                print(f"{func} はこのライブラリに存在しません。")
            continue

        function_info = [f"関数名: {func.__name__}\n"]
        function_info.extend(_wrap_text_by_width("説明: " + func.__doc__))
        function_info.append("引数: " + str(inspect.signature(func)))
        
        box_lines = [f"┏{border_line}┓"]
        box_lines.append(f"┃{padding_line}┃")
        print("\n".join(box_lines)+"\n")
        print("\n".join(function_info)+"\n")
        box_lines = [f"┃{padding_line}┃"]
        box_lines.append(f"┗{border_line}┛")
        print("\n".join(box_lines))


# PyTorchの乱数を固定する関数
def torch_seed(seed = 3):
    """
    PyTorchの乱数を固定する関数。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    print('torch seed set')
    return None




# 学習用関数
def fit(net, optimizer, criterion, num_epochs,
        train_loader, test_loader, device,
        history ):
    """
    学習を行う関数。
    """
    base_epoch = len(history)
    for epoch in tqdm(range(base_epoch,base_epoch+num_epochs)):
        # データ数用変数の初期化
        n_train,n_test = 0,0
        # 損失用変数の初期化
        train_loss,test_loss = 0,0
        # 正解数(精度)用変数の初期化
        train_acc,test_acc= 0,0

        # 訓練モードに設定
        net.train()

        # 訓練データのループ
        for inputs,labels in train_loader:
            
            # データ数のカウント
            n_train += len(labels)

            # GPUへデータを送る
            inputs,labels = inputs.to(device),labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 順伝播
            outputs = net(inputs)

            # 損失の計算
            loss = criterion(outputs,labels)

            # 逆伝播
            loss.backward()

            # パラメータの更新
            optimizer.step()

            # 予測ラベルの取得
            _,predicted = torch.max(outputs.data,1)

            # 損失の計算(lossは平均値なのでデータ数でかけてもどす)
            train_loss += loss.item()*len(labels)

            # 正解数の計算
            train_acc += (predicted == labels).sum().item()
        
        # 検証モードに設定
        net.eval()
        
        for inputs_test, labels_test in test_loader:
            # データ数のカウント
            n_test += len(labels_test)

            # GPUへデータを送る
            inputs_test,labels_test = inputs_test.to(device),labels_test.to(device)

            # 順伝播
            outputs_test = net(inputs_test)

            # 損失の計算
            loss_test = criterion(outputs_test,labels_test)

            # 予測ラベルの取得
            _,predicted_test = torch.max(outputs_test.data,1)

            # 損失の計算(lossは平均値なのでデータ数でかけてもどす)
            test_loss += loss_test.item()*len(labels_test)

            # 正解数の計算
            test_acc += (predicted_test == labels_test).sum().item()
        
        # 損失の計算
        train_loss /= n_train
        test_loss /= n_test

        # 正解率の計算
        train_acc /= n_train
        test_acc /= n_test

        # ログの出力
        print(f'epoch:{epoch+1}/{base_epoch+num_epochs},train_loss:{train_loss:.4f},test_loss:{test_loss:.4f},train_acc:{train_acc:.4f},test_acc:{test_acc:.4f}')

        # ログの保存
        history = np.vstack((history,np.array([epoch+1,train_loss,train_acc,test_loss,test_acc])))

    return history


# 学習ログ解析
def evaluate_history(history):
    """
    学習ログを解析する関数
    """
    #損失と精度の確認
    print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
    print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='訓練')
    plt.plot(history[:,0], history[:,3], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    plt.show()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='訓練')
    plt.plot(history[:,0], history[:,4], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('精度')
    plt.title('学習曲線(精度)')
    plt.legend()
    plt.show()

# イメージとラベル表示
def show_images_labels(loader, classes, net, device: torch.device) -> None:
    """ 
    画像とラベルを表示する関数
    """

    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
      # デバイスの割り当て
      inputs = images.to(device)
      labels = labels.to(device)

      # 予測計算
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]
      #images = images.to('cpu')

    # 最初のn_size個の表示
    plt.figure(figsize=(30, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 正解かどうかで色分けをする
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
          ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1)/2
        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()
