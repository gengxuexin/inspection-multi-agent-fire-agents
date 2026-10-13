#!/bin/bash
# 消防智能执法系统 - 演示用例
# 用法：./demo.sh [1-5]

BASE=~/fire_agents
DEMO=~/fire_agents/demo

# 准备测试文件
mkdir -p $DEMO

echo "=========================================="
echo "  消防智能执法系统 - 演示"
echo "=========================================="
echo

# ---------- 用例 1：法规查询 ----------
#!/bin/bash
cd ~/fire_agent_v2/fire_agents
source ~/.venv/bin/activate
unset DEEPSEEK_API_KEY

TOKEN=$(curl -s -X POST http://localhost:8000/auth/token -H "Content-Type: application/json" -d '{"sub":"张三","dept":"消防支队","role":"officer","clearance":2}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
echo "Token: ${TOKEN:0:20}..."

curl -s http://localhost:8000/dispatch -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"query":"消防法第六十条的处罚标准","conversation_id":"demo-1"}' | python -m json.tool

case_1() {
    echo "【演示 1】法规智能问答"
    echo "问题：消防法第六十条内容是什么？"
    echo "---"
    curl -s -X POST http://127.0.0.1:8000/dispatch \
      -H "Content-Type: application/json" \
      -d '{"query": "消防法第六十条内容是什么", "force_kind": "rag"}' \
      --max-time 300 | python3 -c "
import sys, json
d = json.load(sys.stdin)
inner = d.get('result', {})
if isinstance(inner, dict) and 'result' in inner:
    inner = inner['result']
ans = inner.get('answer', '')
print(ans[:600])
"
}

# ---------- 用例 2：文书生成 ----------
case_2() {
    echo "【演示 2】文书自动生成"
    echo "任务：生成一份立案审批表"
    echo "---"
    curl -s -X POST http://127.0.0.1:8000/dispatch \
      -H "Content-Type: application/json" \
      -d '{"query": "帮我生成一份立案审批表", "force_kind": "doc"}' \
      --max-time 600 | python3 -c "
import sys, json
d = json.load(sys.stdin)
inner = d.get('result', {})
if isinstance(inner, dict) and 'result' in inner:
    inner = inner['result']
files = inner.get('files', [])
for f in files:
    print(f\"✅ 已生成：{f.get('docx')}\")
    print(f\"   大小：{f.get('size_bytes')} 字节\")
"
}

# ---------- 用例 3：图片分析 ----------
case_3() {
    echo "【演示 3】现场照片分析（VLM + YOLO 融合）"
    echo "输入：一张消防检查照片"
    echo "---"
    IMG="$DEMO/demo_image.jpg"
    if [ ! -f "$IMG" ]; then
        echo "⚠️ 请先准备 $IMG"
        return
    fi
    curl -s -X POST http://127.0.0.1:8000/dispatch_file \
      -F "file=@$IMG" \
      -F "query=看看这张图有没有消防隐患" \
      -F "force_kind=vision" \
      --max-time 900 | python3 -c "
import sys, json
d = json.load(sys.stdin)
inner = d.get('result', {}).get('result', {})
yolo = inner.get('yolo', {}).get('counts', {})
ocr = inner.get('ocr', {}).get('ocr_texts', [])
judgment = inner.get('judgment', {})
print(f\"【YOLO】{yolo}\")
print(f\"【OCR】{ocr}\")
print(f\"【结论】{judgment.get('overall_conclusion', '')}\")
for a in judgment.get('alerts', []):
    print(f\"⚠️ [{a.get('level')}] {a.get('title')}: {a.get('description', '')}\")
"
}

# ---------- 用例 4：语音问答 ----------
case_4() {
    echo "【演示 4】语音智能问答"
    echo "输入：一段录音"
    echo "---"
    AUDIO="$DEMO/demo_voice.mp3"
    if [ ! -f "$AUDIO" ]; then
        # 自动生成
        python3 -c "
import asyncio
from edge_tts import Communicate
async def gen():
    c = Communicate('消防法第六十条规定了什么', voice='zh-CN-XiaoxiaoNeural')
    await c.save('$AUDIO')
asyncio.run(gen())
"
    fi
    curl -s -X POST http://127.0.0.1:8000/dispatch_file \
      -F "file=@$AUDIO" \
      -F "query=听一下这段录音" \
      -F "force_kind=voice" \
      --max-time 600 | python3 -c "
import sys, json
d = json.load(sys.stdin)
inner = d.get('result', {}).get('result', {})
if isinstance(inner, dict) and 'result' in inner:
    inner = inner['result']
print(f\"【识别】{inner.get('user_text', '')}\")
print(f\"【回答】{inner.get('answer_text', '')[:300]}\")
print(f\"【语音】{inner.get('audio_path', '')}\")
"
}

# ---------- 用例 5：多模态综合 ----------
case_5() {
    echo "【演示 5】多模态综合（图片+录音 → 报告+文书）"
    echo "输入：现场照片 + 询问录音"
    echo "---"
    IMG="$DEMO/demo_image.jpg"
    AUDIO="$DEMO/demo_voice.mp3"
    TPL="$BASE/agent1/templates/立案审批表.docx"

    [ ! -f "$IMG" ] && echo "⚠️ 缺 $IMG" && return
    [ ! -f "$AUDIO" ] && echo "⚠️ 缺 $AUDIO" && return
    [ ! -f "$TPL" ] && echo "⚠️ 缺 $TPL" && return

    curl -s -X POST http://127.0.0.1:8000/dispatch_multimodal \
      -F "files=@$IMG" \
      -F "files=@$AUDIO" \
      -F "template_file=@$TPL" \
      -F "query=综合分析并生成文书" \
      --max-time 1800 | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('【案情摘要】')
print(d.get('case_summary', '')[:400])
print()
print('【生成文书】', d.get('docx', '（未生成）'))
print()
print('【处理统计】', d.get('stats', {}))
print('【耗时】', d.get('elapsed_sec'), '秒')
"
}

# ---------- 主入口 ----------
case "${1:-all}" in
    1) case_1 ;;
    2) case_2 ;;
    3) case_3 ;;
    4) case_4 ;;
    5) case_5 ;;
    all)
        case_1; echo; read -p "按回车继续..." _
        case_2; echo; read -p "按回车继续..." _
        case_3; echo; read -p "按回车继续..." _
        case_4; echo; read -p "按回车继续..." _
        case_5
        ;;
    *) echo "用法：$0 [1-5|all]" ;;
esac
