#!/bin/bash

echo "欢迎大模型推理系统，基于Kimi"
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

echo "请选择你要运行的方法: "
echo "1. Api生成"
echo "2. Prompt大师"
echo "3. Logic辅助推理"
echo "4. 多智能体（法官开庭!)"
echo "注意事项，你可以在多智能体里面使用‘--loop+数字’ 来选择开庭辩论的次数。(默认为1)"

read -p "请输入对应编号 (1-4): " model_id

case $model_id in
  1)
    echo "[Api生成]"
    python3 main.py --method 0 -n 0
    ;;
  2)
    echo "[Prompt大师]"
    python3 main.py --method 1 -n 0
    ;;
  3)
    echo "[Logic辅助推理]"
    python3 main.py --method 2 -n 0
    ;;
  4)
    echo "[多智能体（法官开庭!)]"
    python3 main.py --method 3 -n 0 --loop 1
    ;;
  *)
    echo "无效输入，请输入 1 到 4 的数字。"
    ;;
esac
