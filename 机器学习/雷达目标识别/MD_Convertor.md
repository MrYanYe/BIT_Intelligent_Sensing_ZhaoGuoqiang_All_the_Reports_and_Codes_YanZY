---
title: Markdown编辑环境搭建
author: wuqx1999
date: 2023年9月21日
# 指定汉字字体，如果缺少中文显示将不正常
CJKmainfont: 方正苏新诗柳楷简体-yolan
   latex 选项
fontsize: 12pt
linkcolor: blue
urlcolor: green
citecolor: cyan
filecolor: magenta
toccolor: red
geometry: margin=0.3in
papersize: A4
documentclass: article

# pandoc设置
output:
   word_document
   # path: 你想保存的路径，默认与.md文档同文件
# 打印背景色
# 保存文件时自动生成
# export_on_save:
#   pandoc: true
---









卷 C 实现细节

卷积神经网络：需要说明的是，对于 OpenSARShip，CNN 的输出维度为 3（如图 1 所示）；对于 MSTAR，CNN 的输出维度为 10（在图 1 中，第三个全连接层为 \(128\times10\)）。训练时，将训练集中的图像作为 CNN 的输入，进行前向传播，得到一个长度为 3（OpenSARShip）或 10（MSTAR）的输出向量。向量的每个分量表示该图像属于某一类别的概率。随后使用随机梯度下降（SGD）最小化式 (1)。初始学习率设为 0.001，每两轮训练将学习率减半，每五轮进行一次验证。当验证准确率不再上升并出现下降趋势时，采用早停策略。

验证阶段，将验证集的图像输入 CNN，样本的类别预测由下式给出：  
\[
\hat{y}_i=\arg\max_c P_\theta\bigl(y_i=c\mid x_i\bigr).
\]
测试阶段同样使用测试集图像作为输入，按式 (7) 的方法预测类别标签。

度量学习：在每个训练 episode 中，支持集 \(S\) 由整个训练集构成，查询集 \(Q\) 则从训练集中随机抽取样本。对于 OpenSARShip，取 \(n=16\)、\(N=3\)；对于 MSTAR，取 \(n=16\)、\(N=10\)。得到 \(S\) 和 \(Q\) 后，先按式 (3) 计算 \(S\) 的原型（prototypes），然后用 SGD 最小化式 (6)。OpenSARShip 的初始学习率设为 \(10^{-5}\)，MSTAR 的初始学习率设为 \(10^{-4}\)。两套数据集均每 1000 个 episode 将学习率减半，每 100 个 episode 进行一次验证。当验证准确率不再上升并出现下降趋势时，同样采用早停策略。

验证时，支持集 \(S\) 保持不变，查询集 \(Q\) 按序从验证集中取出。对特征向量的类别预测由下式给出：  
\[
\hat{y}_j=\arg\max_c r_{j,c}.
\]
测试时同样令 \(S\) 不变，按序从测试集中取出 \(Q\)，并用式 (8) 的方法预测类别标签。

我可以把这段翻译整理成论文排版格式，方便直接替换到稿件中。