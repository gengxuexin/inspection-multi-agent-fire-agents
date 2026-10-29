"""把精排覆盖率从 tf 饱和递增改成 idf 超线性加成。

★ 为什么第一版方向错了（这个必须写进代码注释里）：
  第一版想"命中次数多说明更相关"，改成 idf*(1+ln(1+tf))。
  实测反而更糟：法条文本里重复的往往是虚词。
  查"建筑消防设施包括哪些"时，排在前面的噪声片段是
      tf={'消防设施':1, '包括':7, '建筑':4}
  光靠"包括"重复 7 次就把分刷上去，
  而真正的答案片段只命中 1 个词、tf=1，排名从 212 掉到 565。

  —— 重复次数不是相关性的证据。稀有词命中一次，
  远比常见词重复十几次更有信息量。
"""
import pathlib

NEW = '''if token_set:
            default_idf = self._IDF_MIN
            weights = {t: self._idf.get(t, default_idf) for t in token_set}
            # 归一化基准必须用同一个映射，否则分子多出 ln 项、分母还是线性
            # idf 和，覆盖率会整体超过 1，把 W_OVERLAP 的量纲撑破
            # （分数看似变高，但相对排序被压缩，等于白改）。
            total_w = sum(w * w for w in weights.values()) or 1.0
        else:
            weights, total_w = {}, 1.0

        # 查询里疑问句/连接词的占比。命中这些词的片段多半在讲程序性内容
        # 而不是答案，需要给覆盖率打折。查"建筑消防设施包括哪些"时，
        # "包括"在噪声片段里出现 7 次，不打折它就能刷分。
        cue_ratio = (
            sum(1 for t in token_set if t in self._QUERY_CUES) / len(token_set)
            if token_set else 0.0
        )

        for d in docs:
            score = self._score(law_keyword, article, d)
            text = d.page_content

            # 词面覆盖率（IDF 超线性加成）
            #
            # ★ 这里踩了两次坑才找对方向，完整记录以免再犯：
            #
            # 【坑 1】原来是 `if t in text` —— bool 命中，命中一次和五次同权。
            #   后果：覆盖率迅速饱和。查"电线电缆类产品"时 10 个候选的 ovl
            #   全顶到同一个值，BM25 把 P4 标准条目排到第 1 的判断
            #   在精排这关被抹平，P4 目录切分等于白做。
            #   教训：**召回侧的改进会被精排侧的判据设计抵消。**
            #   改分块前必须问："切出来的更好片段，精排会不会把它排上去？"
            #
            # 【坑 2】第一版改成 tf 饱和递增 idf*(1+ln(1+tf))，
            #   想奖励"命中多次"。结果更糟：法条文本里重复的往往是虚词。
            #   实测"建筑消防设施包括哪些"的噪声片段
            #       tf={'消防设施':1, '包括':7, '建筑':4}
            #   光靠"包括"重复 7 次就刷到 0.85 分，
            #   答案片段只命中 1 词 tf=1，排名 212 → 565（大幅后退）。
            #   —— 重复次数不是相关性的证据。
            #
            # 【现在】只按 idf 计权，稀有词用 idf^2 做超线性加成：
            #   - 稀有词命中一次就能压过所有常见词的命中总和；
            #   - 常见词重复不再有额外收益，"包括"刷 7 次也没用；
            #   - 不依赖 tf，也就无需长度归一化 —— 若按 tf 除以文本长度，
            #     会出现"短片段更容易满分"的副作用，
            #     那样89 字的 P4 目录条目会被长段落比下去（正好和目标相反）。
            if token_set:
                got = sum(w * w for t, w in weights.items() if t in text)
                cov = got / total_w
                if cue_ratio > 0:
                    cov *= (1.0 - self._CUE_DISCOUNT * cue_ratio)
                score += self.W_OVERLAP * cov
'''

# 定位起点：total_w = sum(w * (1.0 + math.log1p
start_marker = "            total_w = sum(w * (1.0 + math.log1p(self._MAX_TF)) for w in weights.values()) or 1.0"
end_marker = "                score += self.W_OVERLAP * (got / total_w)"

p = pathlib.Path("agent1/hybrid.py")
s = p.read_text(encoding="utf-8")

i = s.find(start_marker)
if i < 0:
    raise SystemExit("未找到起点标记")
# 往前回溯到 "if token_set:" 那一行
head = s.rfind("if token_set:", 0, i)
if head < 0:
    raise SystemExit("未找到 if token_set:")
j = s.find(end_marker, i)
if j < 0:
    raise SystemExit("未找到终点标记")
j += len(end_marker)

p.write_text(s[:head] + NEW + s[j:], encoding="utf-8")
print("已替换精排覆盖率实现")