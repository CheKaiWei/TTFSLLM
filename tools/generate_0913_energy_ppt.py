from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path(__file__).resolve().parents[1] / "0913_TTFS_HardwareSim_Energy.pptx"

BG = "#0B1220"
PANEL = "#111C2F"
CARD = "#17243A"
CARD_2 = "#1A2D46"
GRID = "#29405D"
WHITE = "#F5F8FC"
MUTED = "#A9B8CC"
CYAN = "#45D7C6"
BLUE = "#5AA9FF"
AMBER = "#F4C866"
RED = "#FF7F82"
GREEN = "#7BD88F"


def rgb(value):
    value = value.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def shape_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(color)
    shape.line.fill.background()


def add_box(slide, x, y, w, h, color=CARD, radius=True, line=None):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape_fill(shape, color)
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(0.8)
    return shape


def add_line(slide, x1, y1, x2, y2, color=GRID, width=1.0, dash=None):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    if dash:
        line.line.dash_style = dash
    return line


def add_text(slide, x, y, w, h, text, size=14, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, font="Noto Sans CJK SC", margin=0.04,
             valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    paragraphs = text.split("\n")
    for idx, line in enumerate(paragraphs):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(0)
        p.space_before = Pt(0)
        for run in p.runs:
            run.font.name = font
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = rgb(color)
    return box


def add_label(slide, x, y, text, color=CYAN):
    add_text(slide, x, y, 3.8, 0.25, text.upper(), size=9, color=color, bold=True)


def add_number(slide, x, y, number, color):
    circle = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x), Inches(y), Inches(0.32), Inches(0.32)
    )
    shape_fill(circle, color)
    add_text(slide, x, y + 0.005, 0.32, 0.30, str(number), size=11, color=BG,
             bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin=0)


def add_metric_row(slide, x, y, w, label, value, color=WHITE):
    add_text(slide, x, y, w * 0.67, 0.25, label, size=10, color=MUTED)
    add_text(slide, x + w * 0.67, y, w * 0.33, 0.25, value, size=10, color=color,
             bold=True, align=PP_ALIGN.RIGHT)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    background.fill.solid()
    background.fill.fore_color.rgb = rgb(BG)

    # Header
    add_label(slide, 0.42, 0.22, "TTFS / HARDWARESIM AUDIT", CYAN)
    add_text(
        slide, 0.42, 0.50, 12.4, 0.55,
        "TTFS 的能耗优势：来自事件执行，不来自时间码量化",
        size=25, color=WHITE, bold=True,
    )
    add_text(
        slide, 0.44, 1.08, 12.0, 0.25,
        "原文理想机制  |  HardwareSim 可核算指标  |  当前 Qwen2.5 代码证据边界",
        size=10, color=MUTED,
    )
    add_line(slide, 0.42, 1.37, 12.90, 1.37, GRID, 0.8)

    # Three main columns
    add_box(slide, 0.42, 1.58, 3.98, 4.18, PANEL, line=GRID)
    add_box(slide, 4.56, 1.58, 4.20, 4.18, PANEL, line=GRID)
    add_box(slide, 8.92, 1.58, 3.98, 4.18, PANEL, line=RED)

    # Left: original mechanism
    add_label(slide, 0.68, 1.82, "01  原文机制：为什么“可能”省能", CYAN)
    add_text(slide, 0.68, 2.10, 3.35, 0.34,
             "单脉冲 + 二值事件 + 突触累加", size=17, color=WHITE, bold=True)

    rows = [
        ("1", "最多一个脉冲 / 神经元", "N_TTFS ≤ N"),
        ("2", "脉冲触发权重累加", "s_j=1:  u_i += w_ij"),
        ("3", "相对 rate coding 的事件压缩", "N_TTFS / N_rate ≈ 1/(T·r)"),
    ]
    for idx, (num, title, formula) in enumerate(rows):
        yy = 2.67 + idx * 0.73
        add_number(slide, 0.70, yy, num, CYAN)
        add_text(slide, 1.14, yy - 0.01, 2.95, 0.25, title, size=11, color=WHITE, bold=True)
        add_text(slide, 1.14, yy + 0.28, 2.98, 0.22, formula, size=10, color=CYAN, font="DejaVu Sans")
        if idx < 2:
            add_line(slide, 0.86, yy + 0.36, 0.86, yy + 0.68, GRID, 1.1)

    add_box(slide, 0.68, 4.90, 3.44, 0.58, CARD_2)
    add_text(slide, 0.84, 5.03, 3.12, 0.28,
             "必要前提：时间戳直接驱动突触/膜电位，不能先还原 x。",
             size=10, color=AMBER, bold=True)

    # Center: metric model
    add_label(slide, 4.82, 1.82, "02  HardwareSim：应核算完整成本", AMBER)
    add_text(slide, 4.82, 2.10, 3.55, 0.34,
             "AC/MAC 只是算术项，不是总能耗", size=17, color=WHITE, bold=True)

    add_box(slide, 4.82, 2.65, 3.68, 0.88, CARD_2)
    add_text(slide, 5.02, 2.80, 3.30, 0.25,
             "E_arith = N_MAC·e_MAC + N_AC·e_AC + N_kernel·e_kernel",
             size=10, color=WHITE, font="DejaVu Sans")
    add_text(slide, 5.02, 3.15, 3.30, 0.25,
             "E_total = E_arith + E_mem + E_route + E_time + E_sync",
             size=10, color=CYAN, font="DejaVu Sans")

    add_text(slide, 4.82, 3.78, 3.55, 0.24,
             "TTFSFormer 原文采用的理想化账本", size=10, color=MUTED, bold=True)
    add_box(slide, 4.82, 4.05, 3.68, 0.60, CARD)
    add_text(slide, 5.02, 4.17, 3.28, 0.23,
             "E_SNN / E_ANN = OP_SNN·0.9 / (OP_ANN·4.6)",
             size=10, color=AMBER, font="DejaVu Sans")
    add_text(slide, 5.02, 4.42, 3.28, 0.16,
             "0.9 pJ AC  |  4.6 pJ MAC  |  estimated, not measured",
             size=8, color=MUTED)

    add_metric_row(slide, 4.84, 4.92, 3.55, "时间戳控制", "bits / compare", BLUE)
    add_metric_row(slide, 4.84, 5.20, 3.55, "状态与 KV 访存", "bytes / token", BLUE)
    add_metric_row(slide, 4.84, 5.48, 3.55, "事件路由与同步", "events / barrier", BLUE)

    # Right: current Qwen audit
    add_label(slide, 9.18, 1.82, "03  当前 Qwen2.5：实现审计", RED)
    add_text(slide, 9.18, 2.10, 3.40, 0.34,
             "当前没有端到端节能证据", size=17, color=WHITE, bold=True)
    add_box(slide, 9.18, 2.62, 3.42, 0.86, CARD_2)
    add_text(slide, 9.36, 2.77, 3.06, 0.24,
             "time tensor  →  decode  →  dense op",
             size=11, color=WHITE, font="DejaVu Sans", bold=True)
    add_text(slide, 9.36, 3.12, 3.06, 0.22,
             "→  encode  →  next boundary",
             size=10, color=RED, font="DejaVu Sans")

    audit = [
        "QK / AV：dense torch.matmul",
        "RMSNorm / SwiGLU：dense 实值算子",
        "KV cache：同形状 time tensor，全量 decode",
        "Embedding / LM Head：默认仍是 dense",
        "没有 event router、膜电位状态、early stop",
    ]
    for idx, item in enumerate(audit):
        yy = 3.78 + idx * 0.34
        add_text(slide, 9.22, yy, 0.18, 0.22, "×", size=13, color=RED, bold=True)
        add_text(slide, 9.48, yy, 3.03, 0.22, item, size=9.5, color=WHITE)

    add_box(slide, 9.18, 5.27, 3.42, 0.28, RED)
    add_text(slide, 9.28, 5.29, 3.20, 0.20,
             "结论：这是 functional simulator，不是 event-native SNN",
             size=8.6, color=BG, bold=True, align=PP_ALIGN.CENTER)

    # Bottom coarse timing strip
    add_box(slide, 0.42, 5.98, 12.48, 1.02, PANEL, line=GRID)
    add_label(slide, 0.68, 6.17, "04  粗粒度时间量化：K=10", AMBER)
    add_text(slide, 0.68, 6.43, 2.20, 0.30,
             "0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0",
             size=9.5, color=WHITE, font="DejaVu Sans")
    add_line(slide, 0.76, 6.76, 3.96, 6.76, GRID, 2.0)
    for idx in range(11):
        xx = 0.76 + idx * (3.20 / 10)
        color = CYAN if idx in (0, 5, 10) else BLUE
        circle = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.OVAL, Inches(xx - 0.045), Inches(6.715), Inches(0.09), Inches(0.09)
        )
        shape_fill(circle, color)

    add_text(slide, 4.30, 6.13, 3.78, 0.63,
             "11 个端点 = 10 个区间\nΔx=(b−a)/10；只降低 timestamp/control 位宽",
             size=10.5, color=WHITE, bold=True)
    add_text(slide, 8.25, 6.13, 4.28, 0.63,
             "宽度 318.6 → 步长 31.86，误差上界 15.93\nK↓ ≠ events↓；推荐：K_i + residual event + silence threshold",
             size=9.3, color=AMBER, bold=True)

    add_text(slide, 0.45, 7.17, 12.35, 0.16,
             "Source: TTFSFormer §5.2 / Appendix C Table 3; TTFS_LM README; docs/0913_能耗机制与Qwen方案审计.md",
             size=7.5, color=MUTED, font="DejaVu Sans")

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
