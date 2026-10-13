"""
logger = logging.getLogger("fire_agents.agent1")

docx → pdf 转换器
优先用 docx2pdf（依赖本机 Office），失败则用 LibreOffice
"""
import os
import subprocess
import logging


def docx_to_pdf(docx_path: str, pdf_path: str = None) -> str:
    """
    把 docx 转成 pdf，返回 pdf 路径（失败返回 ""）
    """
    if pdf_path is None:
        pdf_path = os.path.splitext(docx_path)[0] + ".pdf"

    if not os.path.exists(docx_path):
        logger.info(f"    ⚠️ 源文件不存在: {docx_path}")
        return ""

    # 方法 1：docx2pdf（依赖本机 Office，转换保真好）
    try:
        from docx2pdf import convert
        convert(docx_path, pdf_path)
        if os.path.exists(pdf_path):
            return pdf_path
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"    docx2pdf 失败: {e}")

    # 方法 2：LibreOffice headless
    # Windows 常见路径
    candidates = [
        "/usr/bin/soffice",
        "/usr/bin/libreoffice",
        "soffice",
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    soffice = None
    for c in candidates:
        if c == "soffice" or os.path.exists(c):
            soffice = c
            break

    if soffice:
        try:
            out_dir = os.path.dirname(pdf_path) or "."
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf",
                 "--outdir", out_dir, docx_path],
                check=True, capture_output=True, timeout=120,
            )
            # LibreOffice 生成的文件名基于 docx 文件名
            generated = os.path.join(
                out_dir,
                os.path.splitext(os.path.basename(docx_path))[0] + ".pdf"
            )
            if os.path.exists(generated):
                # 如果名字不同，重命名
                if generated != pdf_path:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    os.rename(generated, pdf_path)
                return pdf_path
        except Exception as e:
            logger.error(f"    LibreOffice 失败: {e}")

    logger.info("    ❌ 未找到 docx→pdf 转换工具。请安装：")
    logger.info("       pip install docx2pdf    （需要本机安装 Microsoft Word）")
    logger.info("       或安装 LibreOffice: https://www.libreoffice.org/")
    return ""


def batch_convert(output_dir: str, keep_docx: bool = True):
    """批量把 output 目录下的 .docx 全部转成 pdf"""
    if not os.path.exists(output_dir):
        return
    docx_files = [f for f in os.listdir(output_dir) if f.endswith(".docx")]
    for f in docx_files:
        docx_path = os.path.join(output_dir, f)
        logger.info(f"    📄 转换: {f}")
        pdf_path = docx_to_pdf(docx_path)
        if pdf_path:
            logger.info(f"       ✅ → {os.path.basename(pdf_path)}")
            if not keep_docx:
                os.remove(docx_path)