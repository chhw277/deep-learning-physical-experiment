# -*- coding: utf-8 -*-
"""生成竞赛答辩PPT：基于深度学习的低成本单次压缩全斯托克斯偏振成像"""

from pptx import Presentation
from pptx.util import Inches, Pt, Cm, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# Paths
BASE = r"D:\courser workshop\deep rearch(physical rearch)"
FIG_DIR = os.path.join(BASE, "figures")
OUTPUT = os.path.join(BASE, "presentation.pptx")

# Create presentation (widescreen 16:9)
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color scheme
TITLE_BG = RGBColor(0x1B, 0x3A, 0x5C)  # Dark blue
CONTENT_BG = RGBColor(0xFF, 0xFF, 0xFF)  # White
ACCENT = RGBColor(0x2E, 0x86, 0xAB)  # Teal accent
TEXT_DARK = RGBColor(0x1A, 0x1A, 0x1A)
TEXT_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HIGHLIGHT = RGBColor(0xE8, 0x3E, 0x3E)  # Red for key numbers

def add_bg_shape(slide, color):
    """Add full-slide background rectangle"""
    left = top = 0
    width = prs.slide_width
    height = prs.slide_height
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    # Send to back
    sp = shape._element
    sp.getparent().remove(sp)
    slide.shapes._spTree.insert(2, sp)

def set_text(tf, text, size=18, bold=False, color=TEXT_DARK, align=PP_ALIGN.LEFT):
    """Set text in a text frame"""
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return p

def add_para(tf, text, size=16, bold=False, color=TEXT_DARK, indent=0, space_before=Pt(6)):
    """Add a paragraph to text frame"""
    p = tf.add_paragraph()
    p.space_before = space_before
    p.level = indent
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return p

def make_title_slide():
    """Slide 1: Title"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    add_bg_shape(slide, TITLE_BG)

    # Title
    txBox = slide.shapes.add_textbox(Inches(1), Inches(1.8), Inches(11.3), Inches(2.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    set_text(tf, "基于深度学习的低成本单次压缩\n全斯托克斯偏振成像研究",
             size=36, bold=True, color=TEXT_WHITE, align=PP_ALIGN.CENTER)

    # Subtitle
    txBox2 = slide.shapes.add_textbox(Inches(1), Inches(4.5), Inches(11.3), Inches(1.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    set_text(tf2, "第十二届全国大学生物理实验竞赛（自选类题目1）",
             size=22, color=RGBColor(0xAA, 0xD4, 0xE6), align=PP_ALIGN.CENTER)
    add_para(tf2, "Deep Learning-based Full-Stokes Compressed Polarization Imaging",
             size=14, color=RGBColor(0x88, 0xBB, 0xCC), space_before=Pt(12))
    tf2.paragraphs[-1].alignment = PP_ALIGN.CENTER

    # Notes
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        "各位评委老师好，我汇报的题目是《基于深度学习的低成本单次压缩全斯托克斯偏振成像研究》。"
        "本工作围绕Fan等人2023年在Optics Express发表的DL-FSCPHI方法，"
        "实现了系统成本从约34万元降至1629元的极端简化，同时保持核心成像性能。"
    )

def make_background_slide():
    """Slide 2: Research Background"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Section title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    txb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(8), Inches(0.7))
    set_text(txb.text_frame, "研究背景与问题", size=26, bold=True, color=TEXT_WHITE)

    # Content
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.5), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    set_text(tf, "全斯托克斯偏振成像的重要性", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 应用领域：环境监测、生物医学、材料表征、遥感探测", size=16)
    add_para(tf, "• Stokes参数 [S0, S1, S2, S3] 完整描述光的偏振态", size=16)
    add_para(tf, "", size=10)
    add_para(tf, "传统方法的瓶颈", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 光学系统复杂，需要LCTF、电动旋转台等精密器件", size=16)
    add_para(tf, "• 系统成本高昂：约 34 万元", size=16, bold=True, color=HIGHLIGHT)
    add_para(tf, "• 多次测量效率低，无法实现单帧成像", size=16)
    add_para(tf, "", size=10)
    add_para(tf, "教学困境", size=20, bold=True, color=ACCENT)
    add_para(tf, '• 传统仪器"黑匣子"设计 → 学生无法理解内部物理机理', size=16)
    add_para(tf, "• 高成本阻碍大规模教学部署", size=16)

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        "全斯托克斯偏振成像在多个前沿领域具有重要应用。"
        "然而传统系统面临三大瓶颈：结构复杂、成本高昂、测量效率低。"
        "Fan等人2023年在Optics Express提出DL-FSCPHI方法实现单帧成像，"
        "但其系统成本约34万元，无法用于本科教学。"
        "此外，传统教学仪器的黑匣子设计导致学生无法深入理解物理原理。"
    )

def make_innovation_slide():
    """Slide 3: Core Innovations"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    txb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(8), Inches(0.7))
    set_text(txb.text_frame, "核心创新：三项系统简化", size=26, bold=True, color=TEXT_WHITE)

    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.5), Inches(5.8))
    tf = txBox.text_frame
    tf.word_wrap = True

    set_text(tf, "❶ 光源简化", size=20, bold=True, color=ACCENT)
    add_para(tf, "532nm单色激光模块 (77元)  ←  宽带光源+LCTF (~34万元)", size=16)
    add_para(tf, "物理依据：压缩比4:1仅由QWP+LP穆勒矩阵决定，与光谱带宽无关", size=14, color=RGBColor(0x55,0x55,0x55))
    add_para(tf, "", size=8)
    add_para(tf, "❷ 偏振控制简化", size=20, bold=True, color=ACCENT)
    add_para(tf, "手动旋转架 (173元/个)  ←  高精度电动旋转台 (~2万元)", size=16)
    add_para(tf, "", size=8)
    add_para(tf, "❸ 探测器简化", size=20, bold=True, color=ACCENT)
    add_para(tf, "普通工业相机 (380元)  ←  科学级CMOS探测器", size=16)
    add_para(tf, "", size=12)

    # Highlight box
    highlight_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                           Inches(2.5), Inches(5.5), Inches(8), Inches(1.2))
    highlight_box.fill.solid()
    highlight_box.fill.fore_color.rgb = RGBColor(0xFF, 0xF3, 0xF3)
    highlight_box.line.color.rgb = HIGHLIGHT
    htf = highlight_box.text_frame
    htf.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_text(htf, "总成本：1,629 元（原系统的 0.5%），降幅超 99.4%",
             size=22, bold=True, color=HIGHLIGHT, align=PP_ALIGN.CENTER)

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        "我们的核心创新是三项系统简化。"
        "第一，光源简化：用77元的532nm激光模块替代价值约34万的宽带光源+LCTF组合。"
        "物理自洽性在于，4:1压缩比仅由QWP和LP的穆勒矩阵决定，与光谱带宽无关。"
        "第二，用手动旋转架替代电动旋转台。第三，用普通工业相机替代科学级探测器。"
        "最终系统总成本仅1629元，降幅超过99.4%。"
    )

def make_principle_slide():
    """Slide 4: Physical Principle"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    txb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(8), Inches(0.7))
    set_text(txb.text_frame, "物理原理：光学编码 + 深度学习解码", size=26, bold=True, color=TEXT_WHITE)

    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(5.5), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    set_text(tf, "偏振压缩感知", size=20, bold=True, color=ACCENT)
    add_para(tf, "• QWP + LP 固定角度 → 4维Stokes压缩为1维强度", size=15)
    add_para(tf, "• 压缩比 4:1", size=15)
    add_para(tf, "• 测量方程：", size=15)
    add_para(tf, "  I = w₀·S₀ + w₁·S₁ + w₂·S₂ + w₃·S₃", size=15, bold=True)
    add_para(tf, "", size=8)
    add_para(tf, "物理意义", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 穆勒矩阵 = 压缩感知测量矩阵", size=15)
    add_para(tf, "• 深度学习网络 = 非线性逆映射算子", size=15)
    add_para(tf, "", size=8)
    add_para(tf, "DL-M1 网络结构 (~200参数)", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 第1层：Conv 1×1 (1→4) + ReLU", size=15)
    add_para(tf, "  → 压缩解耦（逐像素线性变换）", size=13, color=RGBColor(0x66,0x66,0x66))
    add_para(tf, "• 第2层：Conv 7×7 (4→4)", size=15)
    add_para(tf, "  → 空间增强（上下文建模）", size=13, color=RGBColor(0x66,0x66,0x66))
    add_para(tf, "• 损失：MSE | 优化器：Adam (lr=0.1)", size=15)
    add_para(tf, "• 训练：30 epoch, ~3min (GTX 1650)", size=15)

    # Add loss curve figure on the right
    fig1_path = os.path.join(FIG_DIR, "fig1_loss_curve.png")
    if os.path.exists(fig1_path):
        slide.shapes.add_picture(fig1_path, Inches(7.0), Inches(1.5), Inches(5.8), Inches(4.5))

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        '核心原理是"光学编码+深度学习解码"范式。'
        "QWP和LP以固定角度组合，其穆勒矩阵将4维Stokes信息压缩为1维强度测量，压缩比4:1。"
        "这本质上是偏振域的压缩感知。"
        "深度学习网络DL-M1作为非线性逆映射算子，仅需约200个参数："
        "第一层1×1卷积实现逐像素解耦，第二层7×7卷积引入空间上下文。"
        "右图是训练损失曲线，可以看到约30个epoch即收敛，训练仅需3分钟。"
    )

def make_experiment_slide():
    """Slide 5: Experimental Setup"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    txb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(8), Inches(0.7))
    set_text(txb.text_frame, "实验方案", size=26, bold=True, color=TEXT_WHITE)

    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.5), Inches(5.8))
    tf = txBox.text_frame
    tf.word_wrap = True

    set_text(tf, "光路设计", size=20, bold=True, color=ACCENT)
    add_para(tf, "532nm激光 → 目标 → QWP(手动旋转架) → LP(固定) → 工业相机", size=16, bold=True)
    add_para(tf, "", size=8)
    add_para(tf, "数据采集策略", size=20, bold=True, color=ACCENT)
    add_para(tf, "真值采集（五角度法）：", size=16, bold=True)
    add_para(tf, "  QWP: 0°, 22.5°, 45°, 67.5°, 90° | LP固定45°", size=15)
    add_para(tf, "压缩采集（单帧）：", size=16, bold=True)
    add_para(tf, "  QWP固定114°（最优编码角） | LP固定0°", size=15)
    add_para(tf, "", size=8)
    add_para(tf, "数据集", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 12种不同偏振特性目标", size=16)
    add_para(tf, "• 训练集：10对 | 测试集：2对", size=16)
    add_para(tf, "", size=8)
    add_para(tf, "关键参数", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 空间分辨率：256×256像素", size=16)
    add_para(tf, "• 角度精度：±1° | 功率稳定性：<2%（预热后）", size=16)

    # Add alignment check figure
    fig4_path = os.path.join(FIG_DIR, "fig4_align_check.png")
    if os.path.exists(fig4_path):
        slide.shapes.add_picture(fig4_path, Inches(8.5), Inches(4.2), Inches(4.3), Inches(3.0))

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        "实验光路为：532nm激光照射目标，依次经过QWP和LP后由工业相机采集。"
        "真值采集采用五角度法：QWP旋转5个角度，LP固定45°，联立求解完整Stokes参数。"
        "压缩采集仅需单帧：QWP固定114°、LP固定0°即可。"
        "114°是通过穆勒矩阵条件数优化确定的最优编码角。"
        "共制备12种偏振目标，按10:2划分训练测试集。"
        "右下图展示了系统对齐检查结果。"
    )

def make_results_slide():
    """Slide 6: Results - Quantitative"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    txb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(8), Inches(0.7))
    set_text(txb.text_frame, "实验结果：定量评估", size=26, bold=True, color=TEXT_WHITE)

    # Left: metrics table
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(6.0), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    set_text(tf, "重建质量（PSNR / SSIM）", size=20, bold=True, color=ACCENT)
    add_para(tf, "", size=6)
    add_para(tf, "通道      PSNR(dB)    SSIM", size=16, bold=True)
    add_para(tf, "─────────────────────────────", size=14)
    add_para(tf, "S0         38.2         1.00   ← 总光强，最易重建", size=15)
    add_para(tf, "S3         29.7         0.92", size=15)
    add_para(tf, "S2         27.8         0.87", size=15)
    add_para(tf, "S1         22.3         0.81   ← 误差传播效应", size=15)
    add_para(tf, "─────────────────────────────", size=14)
    add_para(tf, "平均       29.5         0.91", size=16, bold=True, color=HIGHLIGHT)
    add_para(tf, "", size=10)
    add_para(tf, "关键验证", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 与原文DL-FSCPHI结果差异 < 0.5 dB", size=15)
    add_para(tf, "• 鲁棒性：114°与27°编码角PSNR差值仅0.4dB", size=15)
    add_para(tf, "• 推理速度：0.02秒/帧（实时）", size=15)

    # Right: PSNR/SSIM comparison figure
    fig2_path = os.path.join(FIG_DIR, "fig2_psnr_ssim_compare.png")
    if os.path.exists(fig2_path):
        slide.shapes.add_picture(fig2_path, Inches(7.2), Inches(1.3), Inches(5.6), Inches(5.5))

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        "定量结果方面：四通道平均PSNR达29.5dB，SSIM达0.91。"
        "S0通道表现最好（38.2dB），因为总光强信息最容易从压缩测量中恢复。"
        "S1通道相对较低（22.3dB），这是误差传播效应导致的，与原文结论一致。"
        "关键的是，我们的低成本系统与原文34万元系统的性能差异不超过0.5dB，"
        "证明了三项简化的物理合理性。右图是各通道PSNR和SSIM的对比。"
    )

def make_visualization_slide():
    """Slide 7: Results - Visualization"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    txb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(8), Inches(0.7))
    set_text(txb.text_frame, "实验结果：Stokes重建可视化", size=26, bold=True, color=TEXT_WHITE)

    # Full-width figure
    fig3_path = os.path.join(FIG_DIR, "fig3_stokes_vis.png")
    if os.path.exists(fig3_path):
        slide.shapes.add_picture(fig3_path, Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.8))

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        "这张图展示了Stokes四通道的重建结果可视化。"
        "上方是真值（五角度法获得），下方是我们单帧压缩测量后的DL-M1网络重建结果。"
        "可以看到，重建图像与真值在视觉上高度一致，验证了方法的有效性。"
    )

def make_teaching_slide():
    """Slide 8: Teaching Value"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    txb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(8), Inches(0.7))
    set_text(txb.text_frame, "教学价值与应用前景", size=26, bold=True, color=TEXT_WHITE)

    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.5), Inches(5.8))
    tf = txBox.text_frame
    tf.word_wrap = True

    set_text(tf, '突破"黑匣子"设计', size=20, bold=True, color=ACCENT)
    add_para(tf, "• 全透明光路：学生可见全部物理过程", size=16)
    add_para(tf, "• 每个器件的物理作用清晰可辨", size=16)
    add_para(tf, "", size=8)
    add_para(tf, "融合三大前沿概念", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 偏振光学：Mueller矩阵、Stokes参数", size=16)
    add_para(tf, "• 压缩感知：欠采样重建的数学框架", size=16)
    add_para(tf, "• 深度学习：CNN作为逆问题求解器", size=16)
    add_para(tf, "", size=8)
    add_para(tf, "分层次教学体系", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 4个层次、8个实验项目（大二至大四）", size=16)
    add_para(tf, "• 从偏振基础测量 → 压缩感知验证 → 网络训练 → 系统优化", size=16)
    add_para(tf, "", size=8)
    add_para(tf, "经济可行性", size=20, bold=True, color=ACCENT)
    add_para(tf, "• 单台成本 1,629 元", size=16)
    add_para(tf, "• 20台规模教学部署仅需约 3.8 万元", size=16, bold=True, color=HIGHLIGHT)
    add_para(tf, "  （传统方案需 680 万元）", size=14, color=RGBColor(0x66,0x66,0x66))

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        "教学价值方面：首先是突破黑匣子设计，学生可直接观察全部物理过程。"
        "其次，本仪器融合了偏振光学、压缩感知、深度学习三大前沿概念。"
        "我们设计了4个层次共8个实验项目，覆盖大二到大四。"
        "在经济可行性上，20台规模仅需3.8万元，相比传统方案680万元，"
        "使得大规模教学部署成为现实。"
    )

def make_conclusion_slide():
    """Slide 9: Conclusion"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_shape(slide, TITLE_BG)

    txb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(8), Inches(0.8))
    set_text(txb.text_frame, "总结与展望", size=28, bold=True, color=TEXT_WHITE)

    txBox = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(11), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = True

    set_text(tf, "核心成果", size=22, bold=True, color=RGBColor(0xAA, 0xD4, 0xE6))
    add_para(tf, "✓ 以不足原系统 0.6% 的成本实现同等核心性能", size=18, color=TEXT_WHITE)
    add_para(tf, "✓ 单帧采集 → 全Stokes重建（PSNR 29.5dB, SSIM 0.91）", size=18, color=TEXT_WHITE)
    add_para(tf, "✓ 训练仅需3分钟，推理0.02秒/帧", size=18, color=TEXT_WHITE)
    add_para(tf, "", size=10)
    add_para(tf, "方法论贡献", size=22, bold=True, color=RGBColor(0xAA, 0xD4, 0xE6))
    add_para(tf, '✓ 验证了"物理维度分析指导器件简化"的方法论', size=18, color=TEXT_WHITE)
    add_para(tf, "✓ 将前沿科研方法转化为可操作的教学实验", size=18, color=TEXT_WHITE)
    add_para(tf, "", size=10)
    add_para(tf, "展望", size=22, bold=True, color=RGBColor(0xAA, 0xD4, 0xE6))
    add_para(tf, "• 拓展至多光谱偏振成像", size=18, color=TEXT_WHITE)
    add_para(tf, "• 开发实时在线教学演示系统", size=18, color=TEXT_WHITE)
    add_para(tf, "• 探索更复杂场景的泛化能力", size=18, color=TEXT_WHITE)

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = (
        "总结：我们以不足原系统0.6%的成本实现了同等核心性能。"
        "方法论上，验证了通过物理维度分析可以指导系统器件的极端简化。"
        "展望方面，未来可拓展至多光谱偏振成像，并开发实时教学演示系统。"
        "谢谢各位评委老师！"
    )

def make_thanks_slide():
    """Slide 10: Thank You"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_shape(slide, TITLE_BG)

    txBox = slide.shapes.add_textbox(Inches(2), Inches(2.5), Inches(9.3), Inches(2.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    set_text(tf, "谢谢各位评委老师！", size=44, bold=True, color=TEXT_WHITE, align=PP_ALIGN.CENTER)
    add_para(tf, "\n敬请批评指正", size=24, color=RGBColor(0xAA, 0xD4, 0xE6))
    tf.paragraphs[-1].alignment = PP_ALIGN.CENTER

    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = "感谢评委老师的聆听，欢迎提问。"

# Generate all slides
print("Generating presentation...")
make_title_slide()
make_background_slide()
make_innovation_slide()
make_principle_slide()
make_experiment_slide()
make_results_slide()
make_visualization_slide()
make_teaching_slide()
make_conclusion_slide()
make_thanks_slide()

# Save
prs.save(OUTPUT)
print(f"Presentation saved to: {OUTPUT}")
print(f"Total slides: {len(prs.slides)}")
