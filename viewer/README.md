# 实验导图与表格

这里是同一套实验维护源生成的独立查看页，只包含本仓库的HiddenDetect实验记录。没有个人论文阅读库、PDF、模型或原始输入文本。

在仓库根目录运行：

```sh
python viewer/build.py
python -m http.server 8000
```

打开 `http://localhost:8000/viewer/`。也可直接用浏览器打开 `viewer/experiments/detail.html?id=hiddendetect`。

维护源在`viewer/templates/`，项目侧实验设计输入在`viewer/docs/`，实验数据来源仍是仓库`results/`与`docs/`。`viewer/build.py`负责路径适配，保留同树收放、全屏、拖动和鼠标中心缩放；不要只改生成HTML。质心与CFG映射由`docs/experiment-numbering.json`统一提供。

真实项目中的论文按钮进入对应精读页；这个公开研究快照不包含个人精读库，因此按钮指向官方论文入口。表格中的“查看实验记录”链接指向仓库本地报告。未执行方案仍是待验证，历史结果保留原运行与比较边界。
