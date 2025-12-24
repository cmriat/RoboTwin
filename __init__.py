import os
import sys
from pathlib import Path

# RoboTwin 根目录
_ROBOTWIN_ROOT = Path(__file__).resolve().parent

# 切换工作目录，使 ./assets 等相对路径能正确解析
os.chdir(_ROBOTWIN_ROOT)

# 将 RoboTwin 加入模块搜索路径，使 `from envs.xxx import` 能正常工作
if str(_ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOTWIN_ROOT))
