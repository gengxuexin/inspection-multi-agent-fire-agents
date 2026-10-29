"""修正 compare_with_baseline 的调用签名不匹配。

现象：install 步骤按旧签名接的
    compare_with_baseline(eval_data, report, args.baseline)
而实际定义是
    compare_with_baseline(details, baseline_path, noise_floor)

不修的话只要加 --baseline 就TypeError。
"""
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
s = p.read_text(encoding="utf-8")

old = "        rc = compare_with_baseline(eval_data, report, args.baseline)"
new = ("        rc = compare_with_baseline(\n"
       "            eval_data, args.baseline,\n"
       "            noise_floor=args.noise_floor)")

if old in s:
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    print("✓ 已修正调用签名")
else:
    print("· 调用处未找到（可能已修或结构不同），跳过")

# 顺手确认 _noise_floor 字段写入用的是 args.noise_floor
if '"_noise_floor"] = _NOISE_FLOOR' in s:
    s = s.replace('report["_noise_floor"] = _NOISE_FLOOR',
                  'report["_noise_floor"] = args.noise_floor')
    p.write_text(s, encoding="utf-8")
    print("✓ _noise_floor 改为取命令行参数")