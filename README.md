# marksheet
マーク式試験実施用のモジュールと使用例です。

# インストール
1. 必要なライブラリ等は，ライブラリ管理ツールpixiを使用して一括でインストールできます。
1. ブラウザでpixi.shへアクセスし，説明に従ってインストールください。
1. 本リポジトリ上でターミナルを開き，「pixi shell」コマンドを実行してください。必要なライブラリが全てインストールされます。

# サンプル動作
1. sample_script.ipynbを実行してください。
1. files/resultに，各グリッドのマークが塗られていればtrue，そうでなければfalseが書き込まれたcsvが出力されます。
1. 加えて，Trueと判断された箇所が赤丸で示された画像も出力されます。

# マークシートの作成
1. sample_sheet.xlsxがサンプルです。
1. 同一サイズの長方形セルで作成されたシートが対応します。
1. デザイン・印刷は青インクで行ってください（画像処理上，青を透明化してマークを検出します。）


