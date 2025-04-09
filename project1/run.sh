#!/bin/bash

echo "欢迎使用中文文本纠错系统"
echo "正在检查环境依赖..."

# 检查 pip 是否存在
if ! command -v pip &> /dev/null
then
    echo "❌ 未找到 pip，请先安装 Python 和 pip。"
    exit 1
fi

# 安装/检查依赖项
echo "使用 requirements.txt 安装依赖..."
pip install -r requirements.txt

# 环境检查
echo "验证环境中依赖是否有冲突..."
pip check

echo "✅ 环境检查完毕！你现在可以运行主程序了，例如："

echo "请选择你要运行的模型: "
echo "1. Rule-based"
echo "2. Statistical Ngram"
echo "3. Statistical ML"
echo "4. Ensemble Learning"
echo "5. Online Ensemble Learning"
echo "6. Neural Network"
echo "7. Neural Network"

read -p "请输入对应编号 (1-6): " model_id

case $model_id in
  1)
    echo "[Running Rule-based model]"
    python3 main.py --method rule --analyze 0
    ;;
  2)
    echo "[Running Statistical Ngram model]"
    python3 main.py --method statistical --analyze 0 --statistical_method ngram --statistical_optparam 0
    ;;
  3)
    echo "[Running Statistical ML model]"
    python3 main.py --method statistical --analyze 0 --statistical_method ml
    ;;
  4)
    echo "[Running Ensemble Learning model]"
    python3 main.py --method ensemble --analyze 0
    ;;
  5)
    echo "[Running Online Ensemble Learning model]"
    python3 main.py --method ol --analyze 0
    ;;
  6)
    echo "[Running Neural Network model]"
    python3 main.py --method nn --analyze 0
    ;;
  7)
    echo "[Running Neural Network Pretrained Bert model]"
    python3 main.py --method nnpre --analyze 0
    ;;
  *)
    echo "无效输入，请输入 1 到 7 的数字。"
    ;;
esac
