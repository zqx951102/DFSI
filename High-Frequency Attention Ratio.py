
import re
import matplotlib.pyplot as plt

# 读取 .log 文件内容
with open("/Users/zhangqixian/PycharmProjects/pycode/paper/att_weights-1.log", "r") as f:
    log_text = f.read()

# 正则提取 High attention ratio 数值
ratios = [float(m.group(1)) for m in re.finditer(r'High attention ratio: ([0-9.]+)', log_text)]
epochs = list(range(1, len(ratios) + 1))

# 设置字体大小
title_fontsize = 18
label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 14

# 绘制 Attention 曲线图
plt.figure(figsize=(10, 5))
plt.plot(epochs, ratios, marker='o', label='High-Frequency Attention Ratio')
plt.axhline(0.5, color='gray', linestyle='--', label='Equal Ratio')

plt.title("High-Frequency Attention Ratio", fontsize=title_fontsize)
plt.xlabel("Iteration", fontsize=label_fontsize)
plt.ylabel("High Attention Ratio", fontsize=label_fontsize)

plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.grid(True)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()

# 保存为 PDF 文件
plt.savefig("./high_attention_ratio_plot.pdf", format="pdf")
plt.show()

