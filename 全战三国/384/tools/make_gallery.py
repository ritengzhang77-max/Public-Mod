# -*- coding: utf-8 -*-
"""
384 · 淝水之后 —— 展示页图片生成器：人物卡（portrait_*.jpg）、登场条（debuts.jpg）、横幅（banner.jpg）、分节线（divider.png）。

用法：
    python make_gallery.py --src <立绘原图目录> [--out ../images] [--only cards|banner|debuts|divider]

原图要求：<id>_norm.png，全身立绘，透明底（或纯绿幕底，会自动抠掉）。
依赖：Pillow、numpy、scipy。字体用 Windows 自带的楷体 / 微软雅黑，换系统时改 FONT_* 几个常量。
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

FONT_KAI = r"C:/Windows/Fonts/simkai.ttf"
FONT_HEI = r"C:/Windows/Fonts/msyhbd.ttc"
FONT_TXT = r"C:/Windows/Fonts/msyh.ttc"
FONT_NUM = r"C:/Windows/Fonts/georgiab.ttf"

# 势力主色（和 make_map.py 的图例一致）
COLOURS = {
    "秦": (242, 193, 78), "燕": (215, 38, 61), "西燕": (232, 93, 154), "后秦": (142, 91, 217), "晋": (47, 191, 143),
    "北府": (63, 167, 224), "桓氏": (140, 198, 63), "代": (91, 127, 217), "铁弗": (138, 147, 166), "吕氏": (230, 184, 126),
}

# 画廊 16 人：id、姓名、势力（取主色）
CARDS = [
    ("murong_chui", "慕容垂", "燕"), ("fu_jian", "苻坚", "秦"), ("xie_an", "谢安", "晋"), ("yao_chang", "姚苌", "后秦"),
    ("murong_chong", "慕容冲", "西燕"), ("xie_xuan", "谢玄", "北府"), ("liu_yu", "刘裕", "北府"), ("tuoba_gui", "拓跋珪", "代"),
    ("lu_guang", "吕光", "吕氏"), ("helian_bobo", "赫连勃勃", "铁弗"), ("fu_deng", "苻登", "秦"), ("liu_laozhi", "刘牢之", "北府"),
    ("xie_daoyun", "谢道韫", "晋"), ("duan_yuanfei", "段元妃", "燕"), ("jiumoluoshi", "鸠摩罗什", "吕氏"), ("huan_xuan", "桓玄", "桓氏"),
]

# “后浪”登场条：到岁数才登场的人
DEBUTS = [
    ("huan_xuan", "桓玄", "385", "桓氏"), ("tan_daoji", "檀道济", "385", "北府"), ("tuoba_gui", "拓跋珪", "386", "代"),
    ("wang_zhene", "王镇恶", "389", "秦"), ("helian_bobo", "赫连勃勃", "397", "铁弗"), ("cui_hao", "崔浩", "397", "燕"),
]


# ------------------------------------------------------------------------------------------------ 抠图与找头
def key_green(image):
    """去掉纯绿幕。把每个像素看成“前景 × α + 绿幕 × (1 − α)”：绿得超出红 / 蓝多少，就有多少是绿幕，
    据此解出 α 和前景色——发丝、飘带这类半透明的地方也能抠干净。只在真正的背景附近这样做，衣服上的绿色不受影响。"""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w = rgb.shape[:2]
    screen = np.median(np.concatenate([rgb[:6, :6].reshape(-1, 3), rgb[:6, -6:].reshape(-1, 3), rgb[-6:, :6].reshape(-1, 3), rgb[-6:, -6:].reshape(-1, 3)]), axis=0)
    spill = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
    share = np.clip(spill / (screen[1] - max(screen[0], screen[2])), 0, 1)      # 这个像素里绿幕占多少
    solid = share > 0.75
    labels, n = ndimage.label(solid)
    sizes = ndimage.sum(solid, labels, index=np.arange(n + 1))
    keep = sizes >= 12
    keep[np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))] = True
    keep[0] = False
    near = ndimage.binary_dilation(keep[labels], iterations=28)
    share = np.where(near, share, 0)
    alpha = 1 - share
    front = (rgb - share[..., None] * screen) / np.maximum(alpha, 0.12)[..., None]
    # 出图时发丝、飘带和绿幕是“揉”在一起的，不是严格的线性叠加：沾了绿的像素再透一些，
    # 颜色上把绿通道压到“偏暖的棕 / 红”该有的水平（0.6 红 + 0.4 蓝），不然会剩一层橄榄色
    tainted = share > 0.03
    alpha = np.where(tainted, np.clip(1 - share * 1.7, 0, 1), alpha)
    # 红飘带 + 绿幕 = 黄绿色，绿并不“超出”红，上面那一步认不出来：在明显沾绿的像素周围几个像素内一并处理
    zone = ndimage.binary_dilation(share > 0.12, iterations=12) & near
    warm = 0.6 * front[..., 0] + 0.4 * front[..., 2]
    olive = zone & (front[..., 1] > warm + 6)
    alpha = np.where(olive & ~tainted, alpha * 0.7, alpha)
    alpha = np.where(alpha < 0.10, 0, alpha)
    front[..., 1] = np.where(zone | tainted, np.minimum(front[..., 1], warm), front[..., 1])
    out = np.dstack([front.clip(0, 255), alpha * 255]).astype(np.uint8)
    return Image.fromarray(out)


def load_cutout(path):
    im = Image.open(path)
    if im.mode != "RGBA":
        px = im.convert("RGB")
        pts = [(2, 2), (px.width - 3, 2), (2, px.height - 3), (px.width - 3, px.height - 3)]
        if sum(1 for p in pts if px.getpixel(p)[1] > 200 and px.getpixel(p)[0] < 90 and px.getpixel(p)[2] < 90) >= 3:
            return key_green(im)
    return im.convert("RGBA")


def figure_metrics(im):
    """人物外框、身体中轴（x）、头顶（y）。头顶只在中轴附近找，所以枪尖、旗子不会被当成脑袋。"""
    a = np.asarray(im)[..., 3] > 60
    ys, xs = np.nonzero(a)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h = y1 - y0
    band = a[y0 + int(h * 0.35): y0 + int(h * 0.85)]
    cols = band.sum(axis=0).astype(float)
    axis = float((cols * np.arange(a.shape[1])).sum() / cols.sum())
    half = int(a.shape[1] * 0.085)
    rows = a[:, max(0, int(axis) - half): int(axis) + half].sum(axis=1)
    return (x0, y0, x1, y1), axis, int(np.argmax(rows >= 8))


# ------------------------------------------------------------------------------------------------ 水墨底
def noise(shape, scale, seed):
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, shape[0] // scale + 3), max(2, shape[1] // scale + 3))).astype(np.float32)
    big = ndimage.zoom(small, scale, order=3)[: shape[0], : shape[1]]
    return (big - big.min()) / (big.max() - big.min() + 1e-6)


def ink_background(w, h, tint, seed, glow_at=(0.5, 0.36), glow=0.55):
    """深色水墨底：纵向渐变 + 势力色的光晕 + 几层墨云 + 纸纹 + 暗角。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    top, bottom = np.array([30, 32, 40], np.float32), np.array([9, 9, 12], np.float32)
    t = (yy / h)[..., None]
    base[:] = top * (1 - t) + bottom * t
    gx, gy = glow_at[0] * w, glow_at[1] * h
    d = np.sqrt(((xx - gx) / (w * 0.62)) ** 2 + ((yy - gy) / (h * 0.5)) ** 2)
    halo = np.clip(1 - d, 0, 1) ** 2.2
    base += halo[..., None] * np.array(tint, np.float32) * glow
    mist = np.clip(noise((h, w), max(40, w // 5), seed) * 1.6 - 0.75, 0, 1) ** 1.5
    ink = np.clip(noise((h, w), max(30, w // 8), seed + 7) * 1.7 - 0.8, 0, 1) ** 1.3
    base += mist[..., None] * np.array([58, 56, 60], np.float32) * 0.55
    base *= (1 - 0.55 * ink)[..., None]
    grain = np.random.default_rng(seed + 3).normal(0, 3.2, (h, w)).astype(np.float32)
    base += grain[..., None]
    v = np.sqrt(((xx - w / 2) / (w * 0.75)) ** 2 + ((yy - h / 2) / (h * 0.75)) ** 2)
    base *= np.clip(1.08 - v ** 2.4 * 0.75, 0.25, 1.0)[..., None]
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def soft_shadow(cutout, blur, strength=0.75, offset=(0, 6)):
    a = cutout.split()[3].filter(ImageFilter.GaussianBlur(blur))
    shadow = Image.new("RGBA", cutout.size, (0, 0, 0, 0))
    shadow.putalpha(a.point(lambda v: int(v * strength)))
    return ImageChops.offset(shadow, *offset)


def rim_light(cutout, colour, width=2.2, strength=0.55):
    """沿人物外缘的一圈势力色辉光。"""
    a = cutout.split()[3]
    grown = a.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(width * 3))
    glow = Image.new("RGBA", cutout.size, tuple(colour) + (0,))
    glow.putalpha(grown.point(lambda v: int(v * strength)))
    return glow


def seal(name, size=34, pad=7):
    """竖排朱砂名章。"""
    kai = ImageFont.truetype(FONT_KAI, size)
    hei = ImageFont.truetype(FONT_HEI, size - 4)
    w, h = size + pad * 2, len(name) * (size + 3) + pad * 2 - 3
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=5, fill=(168, 30, 32, 235))
    d.rounded_rectangle([3, 3, w - 4, h - 4], radius=3, outline=(255, 230, 210, 120), width=1)
    for i, ch in enumerate(name):
        f = kai if kai.getmask(ch).getbbox() else hei
        bb = d.textbbox((0, 0), ch, font=f)
        d.text((w / 2 - (bb[0] + bb[2]) / 2, pad + i * (size + 3) + size / 2 - (bb[1] + bb[3]) / 2), ch, font=f, fill=(255, 244, 232, 255))
    arr = np.asarray(img).astype(np.float32)
    speck = noise((h, w), 3, len(name) * 11 + size)
    arr[..., 3] *= np.where(speck > 0.86, 0.55, 1.0)
    return Image.fromarray(arr.astype(np.uint8))


# ------------------------------------------------------------------------------------------------ 人物卡
def bust(cutout, w, h, zoom=0.50, headroom=0.05):
    """以头顶和身体中轴为准裁出半身：取人物高度的 zoom 这一段，缩放到 w × h。"""
    bbox, axis, head = figure_metrics(cutout)
    fh = bbox[3] - head
    ch = fh * zoom
    cw = ch * w / h
    top, left = head - fh * headroom, axis - cw / 2
    scale = h / ch
    big = cutout.resize((int(cutout.width * scale), int(cutout.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(big, (int(round(-left * scale)), int(round(-top * scale))), big)
    return canvas


def bottom_fade(w, h, start=0.72, strength=215):
    fade = np.zeros((h, w, 4), np.uint8)
    ramp = np.clip((np.arange(h) - h * start) / (h * (1 - start)), 0, 1) ** 1.6
    fade[..., 3] = (ramp * strength).astype(np.uint8)[:, None]
    return Image.fromarray(fade)


def make_card(src, pid, name, faction, w=360, h=480, zoom=0.50):
    tint = COLOURS[faction]
    seed = sum(ord(c) for c in pid)
    card = ink_background(w, h, tint, seed).convert("RGBA")
    fig = bust(load_cutout(os.path.join(src, pid + "_norm.png")), w, h, zoom=zoom)
    card.alpha_composite(rim_light(fig, tint))
    card.alpha_composite(soft_shadow(fig, 10))
    card.alpha_composite(fig)
    card.alpha_composite(bottom_fade(w, h))
    d = ImageDraw.Draw(card)
    d.rectangle([7, 7, w - 8, h - 8], outline=(214, 180, 110, 110), width=1)
    d.rectangle([7, h - 12, w - 8, h - 8], fill=tuple(tint) + (230,))
    card.alpha_composite(seal(name), (18, 18))
    return card.convert("RGB")


# ------------------------------------------------------------------------------------------------ 横幅
def ridge(w, base_y, amp, seed, rough=3):
    """一条山脊线：几层不同频率的噪声叠起来。"""
    rng = np.random.default_rng(seed)
    x = np.arange(w, dtype=np.float32)
    y = np.zeros(w, np.float32)
    for k in range(rough + 2):
        n = 3 * 2 ** k
        pts = rng.random(n + 3).astype(np.float32)
        y += np.interp(x, np.linspace(0, w, n + 3), pts) * amp / (1.7 ** k)
    return base_y - y


def banner_background(w, h, seed=384):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (yy / h)[..., None]
    sky_top, sky_mid, sky_low = np.array([16, 18, 26], np.float32), np.array([44, 36, 40], np.float32), np.array([10, 10, 13], np.float32)
    base = np.where(t < 0.55, sky_top + (sky_mid - sky_top) * (t / 0.55), sky_mid + (sky_low - sky_mid) * ((t - 0.55) / 0.45))
    # 残阳
    sx, sy, sr = w * 0.685, h * 0.40, h * 0.34
    d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    disc = np.clip((sr - d) / 3.0, 0, 1)
    halo = np.clip(1 - d / (sr * 3.2), 0, 1) ** 2.4
    base = base + halo[..., None] * np.array([150, 52, 38], np.float32) * 0.85
    base = base * (1 - disc[..., None] * 0.92) + disc[..., None] * np.array([172, 46, 40], np.float32) * 0.92
    # 远山三重
    for i, (by, amp, tone, fog) in enumerate([(h * 0.62, h * 0.30, (58, 50, 58), 0.55), (h * 0.76, h * 0.26, (36, 33, 40), 0.4), (h * 0.93, h * 0.22, (17, 17, 21), 0.2)]):
        line = ridge(w, by, amp, seed + i * 13)
        below = np.clip((yy - line[None, :]) / 2.0, 0, 1)
        depth = np.clip((yy - line[None, :]) / (h * 0.30), 0, 1)
        layer = np.array(tone, np.float32) * (1 - fog * depth[..., None]) + np.array([92, 84, 88], np.float32) * (fog * depth[..., None])
        base = base * (1 - below[..., None]) + layer * below[..., None]
    mist = np.clip(noise((h, w), 150, seed + 5) * 1.5 - 0.62, 0, 1) ** 1.4
    band = np.clip(1 - np.abs(yy / h - 0.80) / 0.22, 0, 1)
    base += (mist * band)[..., None] * np.array([70, 64, 66], np.float32) * 0.75
    ink = np.clip(noise((h, w), 90, seed + 9) * 1.7 - 0.85, 0, 1)
    base *= (1 - 0.5 * ink)[..., None]
    base += np.random.default_rng(seed).normal(0, 3.0, (h, w)).astype(np.float32)[..., None]
    v = np.sqrt(((xx - w / 2) / (w * 0.72)) ** 2 + ((yy - h / 2) / (h * 0.9)) ** 2)
    base *= np.clip(1.1 - v ** 2.6 * 0.8, 0.2, 1.0)[..., None]
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def gradient_text(size, text, font, top, bottom, stroke=0):
    """竖向渐变填充的文字（返回 RGBA 图层）。"""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.text((0, 0), text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    bb = mask.getbbox() or (0, 0, 1, 1)
    ramp = np.linspace(0, 1, max(1, bb[3] - bb[1]), dtype=np.float32)
    col = np.zeros((size[1], size[0], 3), np.float32)
    col[:] = np.array(bottom, np.float32)
    seg = np.array(top, np.float32)[None, :] * (1 - ramp[:, None]) + np.array(bottom, np.float32)[None, :] * ramp[:, None]
    col[bb[1]: bb[3]] = seg[:, None, :]
    return Image.fromarray(np.dstack([col, np.asarray(mask, np.float32)]).astype(np.uint8))


def put_text(canvas, xy, text, font, top, bottom=None, shadow=6, stroke=0, spacing=0):
    """带投影的文字；spacing > 0 时逐字排（字距）。返回文字右端的 x。"""
    x, y = xy
    chunks = list(text) if spacing else [text]
    for chunk in chunks:
        wbox = ImageDraw.Draw(canvas).textbbox((0, 0), chunk, font=font, stroke_width=stroke)
        size = (wbox[2] + 8, wbox[3] + 8)
        layer = gradient_text(size, chunk, font, top, bottom or top, stroke)
        if shadow:
            sh = Image.new("RGBA", (size[0] + shadow * 6, size[1] + shadow * 6), (0, 0, 0, 0))
            sh.paste((0, 0, 0, 255), (shadow * 3, shadow * 3), layer.split()[3])
            sh = sh.filter(ImageFilter.GaussianBlur(shadow))
            canvas.alpha_composite(sh, (int(x - shadow * 3 + 2), int(y - shadow * 3 + 4)))
        canvas.alpha_composite(layer, (int(x), int(y)))
        x += wbox[2] + spacing
    return x


def square_seal(text, side=104):
    """2 × 2 的朱砂方印（右起竖读）。"""
    img = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, side - 1, side - 1], radius=8, fill=(172, 32, 34, 240))
    d.rounded_rectangle([5, 5, side - 6, side - 6], radius=5, outline=(255, 232, 214, 150), width=2)
    font = ImageFont.truetype(FONT_KAI, int(side * 0.40))
    cell = (side - 16) / 2
    for i, ch in enumerate(text[:4]):
        col, row = 1 - i // 2, i % 2
        bb = d.textbbox((0, 0), ch, font=font)
        cx, cy = 8 + col * cell + cell / 2, 8 + row * cell + cell / 2
        d.text((cx - (bb[0] + bb[2]) / 2, cy - (bb[1] + bb[3]) / 2), ch, font=font, fill=(255, 244, 232, 255))
    arr = np.asarray(img).astype(np.float32)
    arr[..., 3] *= np.where(noise((side, side), 3, 77) > 0.84, 0.5, 1.0)
    return Image.fromarray(arr.astype(np.uint8))


def standing(cutout, fig_h, tint_to=None, tint=0.0, trim_left=None):
    """按人物身高 fig_h 缩放的全身像；tint > 0 时往背景色里退一点（远处的人）。返回 (图, 中轴 x, 头顶 y)。
    trim_left：中轴左侧多少像素（原图尺度）以外淡出——用来拿掉会压到标题上的长兵器。"""
    bbox, axis, head = figure_metrics(cutout)
    if trim_left:
        arr = np.asarray(cutout).astype(np.float32)
        ramp = np.clip((np.arange(cutout.width) - (axis - trim_left - 60)) / 60.0, 0, 1)
        arr[..., 3] *= ramp[None, :]
        cutout = Image.fromarray(arr.astype(np.uint8))
    scale = fig_h / float(bbox[3] - head)
    img = cutout.resize((int(cutout.width * scale), int(cutout.height * scale)), Image.LANCZOS)
    if tint > 0:
        arr = np.asarray(img).astype(np.float32)
        arr[..., :3] = arr[..., :3] * (1 - tint) + np.array(tint_to, np.float32) * tint
        img = Image.fromarray(arr.astype(np.uint8))
    return img, axis * scale, head * scale


def make_banner(src, w=1600, h=600):
    canvas = banner_background(w, h).convert("RGBA")
    # 人物：x = 中轴位置，top = 头顶位置，height = 身高（像素），z 大的后画
    lineup = [
        ("murong_chong", 668, 96, 930, 0, 175), ("yao_chang", 842, 78, 960, 1, None), ("fu_jian", 1006, 60, 1010, 3, None),
        ("murong_chui", 1170, 70, 990, 2, None), ("xie_an", 1320, 84, 950, 1, None), ("liu_yu", 1454, 98, 920, 0, None),
    ]
    far = (20, 20, 28)
    for pid, x, top, fig_h, z, trim in sorted(lineup, key=lambda f: f[4]):
        img, axis, head = standing(load_cutout(os.path.join(src, pid + "_norm.png")), fig_h, far, {0: 0.34, 1: 0.2, 2: 0.08, 3: 0.0}[z], trim)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        layer.paste(img, (int(x - axis), int(top - head)), img)
        canvas.alpha_composite(soft_shadow(layer, 14, 0.6, (5, 8)))
        canvas.alpha_composite(layer)
    # 下沿的雾和压暗
    canvas.alpha_composite(bottom_fade(w, h, 0.62, 235))
    fog = np.zeros((h, w, 4), np.float32)
    fog[..., :3] = (120, 108, 108)
    fog[..., 3] = np.clip(noise((h, w), 170, 21) * 1.5 - 0.55, 0, 1) * np.clip((np.arange(h) / h - 0.66) / 0.2, 0, 1)[:, None] * 95
    canvas.alpha_composite(Image.fromarray(fog.astype(np.uint8)))
    # 左侧让出标题的位置
    left = np.zeros((h, w, 4), np.uint8)
    left[..., 3] = (np.clip(1 - np.arange(w) / (w * 0.50), 0, 1) ** 1.2 * 200).astype(np.uint8)[None, :]
    canvas.alpha_composite(Image.fromarray(left))

    gold_a, gold_b = (250, 228, 168), (196, 144, 62)
    put_text(canvas, (64, 58), "TOTAL WAR: THREE KINGDOMS  ·  MOD", ImageFont.truetype(FONT_NUM, 21), (206, 188, 150), shadow=3, spacing=3)
    x = put_text(canvas, (56, 84), "384", ImageFont.truetype(FONT_NUM, 176), gold_a, gold_b, shadow=8)
    canvas.alpha_composite(square_seal("群雄并起", 108), (int(x) + 26, 150))
    put_text(canvas, (58, 276), "淝水之后", ImageFont.truetype(FONT_KAI, 124), (250, 244, 232), (214, 204, 188), shadow=8, stroke=1, spacing=6)
    d = ImageDraw.Draw(canvas)
    d.line([(64, 432), (560, 432)], fill=(214, 180, 110, 200), width=2)
    d.line([(64, 437), (360, 437)], fill=(214, 180, 110, 110), width=1)
    put_text(canvas, (64, 452), "淝水之战次年 · 前秦土崩 · 十六国群雄并起", ImageFont.truetype(FONT_TXT, 27), (232, 224, 210), shadow=4)
    put_text(canvas, (64, 500), "21 个势力  ·  200 位历史人物  ·  人人有立绘", ImageFont.truetype(FONT_TXT, 23), gold_a, gold_b, shadow=4)
    d.rectangle([0, 0, w - 1, h - 1], outline=(214, 180, 110, 90), width=2)
    return canvas.convert("RGB")


# ------------------------------------------------------------------------------------------------ 登场条
def make_debuts(src, cw=250, ch=300):
    n = len(DEBUTS)
    strip = Image.new("RGBA", (cw * n, ch + 54), (11, 12, 16, 255))
    year_font, name_font = ImageFont.truetype(FONT_NUM, 30), ImageFont.truetype(FONT_KAI, 30)
    for i, (pid, name, year, faction) in enumerate(DEBUTS):
        tint = COLOURS[faction]
        cell = ink_background(cw, ch, tint, 50 + i, glow=0.5).convert("RGBA")
        fig = bust(load_cutout(os.path.join(src, pid + "_norm.png")), cw, ch, zoom=0.33, headroom=0.04)
        cell.alpha_composite(rim_light(fig, tint))
        cell.alpha_composite(fig)
        cell.alpha_composite(bottom_fade(cw, ch, 0.70, 230))
        strip.alpha_composite(cell, (i * cw, 0))
        d = ImageDraw.Draw(strip)
        d.rectangle([i * cw, ch, (i + 1) * cw - 1, ch + 3], fill=tuple(tint) + (255,))
        d.ellipse([i * cw + cw / 2 - 7, ch - 5, i * cw + cw / 2 + 7, ch + 9], fill=(250, 228, 168, 255), outline=(11, 12, 16, 255), width=2)
        yb = d.textbbox((0, 0), year, font=year_font)
        nb = d.textbbox((0, 0), name, font=name_font)
        total = yb[2] + 12 + nb[2]
        x0 = i * cw + (cw - total) / 2
        d.text((x0, ch + 12), year, font=year_font, fill=(250, 228, 168, 255))
        d.text((x0 + yb[2] + 12, ch + 12), name, font=name_font, fill=(240, 234, 222, 255))
        if i:
            d.line([(i * cw, 0), (i * cw, ch)], fill=(0, 0, 0, 255), width=2)
    return strip.convert("RGB")


def make_divider(w=1600, h=36):
    """分节用的细金线：两端淡出，中间一枚朱砂菱形。透明底，深浅两种主题下都看得见。"""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    arr = np.zeros((h, w, 4), np.float32)
    fade = np.clip(1 - np.abs(np.linspace(-1, 1, w)) ** 1.6, 0, 1)
    arr[h // 2 - 1: h // 2 + 1, :, :3] = (201, 162, 86)
    arr[h // 2 - 1: h // 2 + 1, :, 3] = fade[None, :] * 255
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    cx, cy, r = w // 2, h // 2, 13
    d.polygon([(cx - r - 10, cy), (cx, cy - r), (cx + r + 10, cy), (cx, cy + r)], fill=(168, 30, 32, 255), outline=(222, 190, 120, 255))
    for dx in (-46, 46):
        d.polygon([(cx + dx - 6, cy), (cx + dx, cy - 5), (cx + dx + 6, cy), (cx + dx, cy + 5)], fill=(201, 162, 86, 255))
    return img


def save_jpg(img, path, quality=88, limit=300_000):
    while True:
        img.save(path, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=0)
        if os.path.getsize(path) <= limit or quality <= 60:
            return os.path.getsize(path)
        quality -= 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="立绘原图目录（<id>_norm.png）")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images"))
    ap.add_argument("--only", choices=["cards", "banner", "debuts", "divider"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.only in (None, "cards"):
        for pid, name, faction in CARDS:
            size = save_jpg(make_card(args.src, pid, name, faction), os.path.join(args.out, "portrait_%s.jpg" % pid))
            print("card", pid, size)
    if args.only in (None, "debuts"):
        print("debuts", save_jpg(make_debuts(args.src), os.path.join(args.out, "debuts.jpg")))
    if args.only in (None, "divider"):
        make_divider().save(os.path.join(args.out, "divider.png"), optimize=True)
        print("divider", os.path.getsize(os.path.join(args.out, "divider.png")))
    if args.only in (None, "banner"):
        print("banner", save_jpg(make_banner(args.src), os.path.join(args.out, "banner.jpg"), quality=90))


if __name__ == "__main__":
    main()
