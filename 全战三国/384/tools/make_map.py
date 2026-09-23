# -*- coding: utf-8 -*-
"""
384 · 淝水之后 —— 开局势力示意图。

    python make_map.py            # 读同目录的 territory_384.json，写 ../images/territory_384.jpg

这是一张**示意图**：游戏没有公开的地图坐标，所以每个郡按“郡治在历史上大致在哪”手工给了一个经纬度，
海岸线和江河也是手描的折线。一点 = 一郡（游戏地图共 84 郡），颜色 = 384 年六月开局时郡治归谁；
同一郡里归属不同的城（邺城内外、长安城内外……）另画小点。要改归属或位置，改 territory_384.json 后重跑。

依赖：matplotlib、numpy、Pillow。中文字体默认用 Windows 的微软雅黑 / 楷体，换系统时改 FONT*。
"""
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = r"C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = r"C:/Windows/Fonts/msyhbd.ttc"
FONT_KAI = r"C:/Windows/Fonts/simkai.ttf"

LON0, LON1, LAT0, LAT1 = 97.6, 125.4, 17.9, 42.9
K = float(np.cos(np.radians(33.0)))          # 经度方向的压缩（等距圆柱投影，标准纬线 33°N）
WIDTH_IN, DPI, MAX_BYTES = 10.4, 160, 295_000

# id: (游戏里的名字, 颜色)
FACTIONS = {
    "former_qin":           ("秦", "#F2C14E"),
    "former_qin_guandong":  ("秦·关东", "#E07B39"),
    "former_qin_bingzhou":  ("秦·并州", "#A89F45"),
    "former_qin_youzhou":   ("秦·幽州", "#EDE08A"),
    "former_qin_pingzhou":  ("秦·平州", "#9C7A3C"),
    "former_qin_qingzhou":  ("秦·青州", "#F6C28B"),
    "former_qin_luoyang":   ("秦·洛阳", "#D9D0A0"),
    "former_qin_liangzhou": ("秦·凉州", "#D9A066"),
    "later_yan":            ("燕", "#D7263D"),
    "dingling_zhai":        ("丁零", "#B0596D"),
    "western_yan":          ("西燕", "#E85D9A"),
    "later_qin":            ("后秦", "#8E5BD9"),
    "dai_tuoba":            ("代", "#5B7FD9"),
    "tiefu":                ("铁弗", "#8A93A6"),
    "qifu_xianbei":         ("乞伏", "#B0724A"),
    "eastern_jin":          ("晋", "#2FBF8F"),
    "jin_beifu":            ("晋·北府", "#3FA7E0"),
    "jin_jingzhou_huan":    ("晋·荆州桓氏", "#8CC63F"),
    "jin_guangzhou":        ("晋·岭南", "#7ED6C1"),
    "jin_ningzhou":         ("晋·宁州", "#1F7A4A"),
}
LEGEND = [
    [("秦 · 七镇皆为附庸", ["former_qin", "former_qin_guandong", "former_qin_bingzhou", "former_qin_youzhou", "former_qin_pingzhou",
                        "former_qin_qingzhou", "former_qin_luoyang", "former_qin_liangzhou"])],
    [("燕 · 附庸丁零", ["later_yan", "dingling_zhai"]), ("各自为战", ["western_yan", "later_qin", "dai_tuoba", "tiefu", "qifu_xianbei"])],
    [("晋 · 四镇皆为附庸", ["eastern_jin", "jin_beifu", "jin_jingzhou_huan", "jin_guangzhou", "jin_ningzhou"])],
]
# 都城（君主开局所在的城）：势力 id -> 城名
CAPITALS = {
    "former_qin": "长安", "former_qin_guandong": "邺", "former_qin_bingzhou": "晋阳", "former_qin_youzhou": "蓟", "former_qin_pingzhou": "阳乐",
    "former_qin_qingzhou": "剧县", "former_qin_luoyang": "洛阳", "former_qin_liangzhou": "姑臧", "later_yan": "怀县", "dingling_zhai": "东平",
    "western_yan": "弘农", "later_qin": "三水", "dai_tuoba": "阴馆", "tiefu": "河阴", "qifu_xianbei": "金城", "eastern_jin": "建业",
    "jin_beifu": "淮阴", "jin_jingzhou_huan": "江陵", "jin_guangzhou": "番禺", "jin_ningzhou": "味县",
}
SHOWN_AS = {"建业": "建康"}                      # 图上写 384 年的叫法
PREFER = {"弘农": (0, -1), "怀县": (1, 0), "邺": (1, 0), "长安": (1, 0)}   # 都城名优先摆在标记的哪一侧
# 开局战争：进攻方的城 -> 防守方的城，弧度
WARS = [("弘农", "长安", 0.25), ("三水", "长安", 0.25), ("淮阴", "剧县", 0.22), ("淮阴", "洛阳", 0.16), ("江陵", "长安", 0.18)]
# 势力名摆在哪（经度, 纬度, 字号）
FACTION_LABELS = [
    ("former_qin", 105.4, 30.2, 22), ("former_qin_guandong", 118.75, 35.3, 11.5), ("former_qin_bingzhou", 111.55, 38.5, 12.5),
    ("former_qin_youzhou", 117.3, 40.6, 12.5), ("former_qin_pingzhou", 122.2, 42.1, 12.5), ("former_qin_qingzhou", 120.4, 36.95, 12),
    ("former_qin_luoyang", 112.5, 34.08, 10), ("former_qin_liangzhou", 102.1, 37.05, 12.5), ("later_yan", 115.9, 36.65, 19),
    ("dingling_zhai", 117.35, 36.3, 10.5), ("western_yan", 110.35, 36.2, 12.5), ("later_qin", 108.0, 37.3, 14), ("dai_tuoba", 113.3, 39.9, 16),
    ("tiefu", 108.5, 40.2, 14), ("qifu_xianbei", 104.4, 35.35, 13), ("eastern_jin", 116.7, 27.6, 25), ("jin_beifu", 121.35, 33.55, 13),
    ("jin_jingzhou_huan", 111.3, 29.55, 14.5), ("jin_guangzhou", 110.0, 22.65, 15), ("jin_ningzhou", 103.0, 24.35, 14),
]

# ---------------------------------------------------------------------------------------------- 手描的海岸与江河
COAST = [
    (125.6, 39.5), (124.7, 39.65), (124.3, 39.9), (123.6, 39.75), (122.9, 39.55), (122.2, 39.15), (121.6, 38.85), (121.15, 38.75),
    (121.35, 39.1), (121.6, 39.5), (122.05, 40.2), (122.2, 40.6), (121.9, 40.9), (121.3, 40.95), (120.95, 40.7), (120.4, 40.3),
    (119.85, 39.95), (119.3, 39.45), (118.95, 39.15), (118.3, 39.05), (117.85, 39.2), (117.7, 38.9), (117.6, 38.5), (118.0, 38.2),
    (118.6, 38.1), (119.05, 37.8), (119.0, 37.4), (119.2, 37.15), (119.6, 37.1), (119.9, 37.3), (120.3, 37.65), (120.75, 37.83),
    (121.4, 37.6), (122.1, 37.5), (122.65, 37.4), (122.5, 36.95), (122.0, 36.9), (121.5, 36.75), (120.9, 36.55), (120.6, 36.3),
    (120.3, 36.05), (120.1, 35.85), (119.6, 35.5), (119.35, 35.05), (119.25, 34.75), (119.6, 34.5), (120.2, 34.3), (120.5, 33.8),
    (120.8, 33.2), (121.0, 32.7), (121.4, 32.3), (121.85, 31.95), (121.6, 31.68), (121.9, 31.2), (121.95, 30.9), (121.4, 30.75),
    (120.95, 30.5), (120.6, 30.35), (120.35, 30.3), (120.7, 30.2), (121.2, 30.3), (121.6, 30.05), (121.95, 29.9), (121.95, 29.5),
    (121.6, 29.15), (121.65, 28.7), (121.4, 28.3), (121.0, 27.95), (120.65, 27.5), (120.3, 27.2), (120.05, 26.8), (119.7, 26.5),
    (119.65, 26.0), (119.5, 25.5), (119.15, 25.2), (118.8, 24.85), (118.2, 24.45), (117.9, 24.05), (117.4, 23.7), (116.8, 23.3),
    (116.2, 22.95), (115.5, 22.75), (114.8, 22.6), (114.25, 22.3), (113.85, 22.5), (113.6, 22.8), (113.5, 22.3), (113.1, 22.0),
    (112.4, 21.8), (111.7, 21.6), (111.0, 21.45), (110.55, 21.2), (110.45, 20.6), (110.2, 20.3), (109.9, 20.4), (109.75, 21.0),
    (109.6, 21.5), (109.15, 21.45), (108.6, 21.7), (108.1, 21.55), (107.4, 21.3), (106.9, 20.85), (106.65, 20.3), (106.1, 19.95),
    (105.85, 19.3), (105.75, 18.7), (106.15, 18.2), (106.6, 17.6),
]
ISLANDS = [
    [(110.7, 20.1), (111.0, 19.65), (110.55, 18.8), (110.0, 18.4), (109.5, 18.2), (108.7, 18.5), (108.65, 19.3), (109.2, 19.85), (109.9, 20.0)],
    [(121.55, 25.3), (122.0, 25.0), (121.8, 24.4), (121.6, 23.8), (121.3, 22.9), (120.85, 21.95), (120.6, 22.4), (120.2, 22.9), (120.1, 23.5),
     (120.4, 24.2), (120.9, 24.8), (121.2, 25.1)],
]
RIVERS = {
    "河水": (2.2, [(101.2, 35.2), (100.75, 35.75), (100.95, 36.12), (101.45, 36.03),
                  (102.5, 35.85), (103.3, 35.95), (103.8, 36.07), (104.7, 36.57), (105.2, 37.5), (106.0, 38.0), (106.35, 38.5),
                  (106.8, 39.25), (106.8, 39.7), (107.0, 40.33), (107.4, 40.75), (108.65, 40.72), (109.85, 40.55), (111.2, 40.28),
                  (111.4, 39.6), (111.1, 39.0), (110.5, 38.0), (110.7, 37.45), (110.45, 36.15), (110.6, 35.65), (110.25, 34.85),
                  (110.28, 34.6), (111.2, 34.78), (112.45, 34.85), (113.5, 34.95), (114.5, 35.5), (115.05, 35.85), (115.95, 36.45),
                  (116.6, 36.95), (117.5, 37.45), (118.3, 37.5), (118.95, 37.75)]),
    "江水": (2.4, [(101.7, 26.6), (102.6, 26.3), (102.9, 26.9), (103.6, 27.8), (104.15, 28.65), (104.63, 28.77), (105.44, 28.88),
                  (106.25, 29.28), (106.58, 29.57), (107.4, 29.72), (108.05, 30.3), (108.4, 30.82), (109.5, 31.03), (109.88, 31.08),
                  (110.7, 30.85), (111.3, 30.7), (111.75, 30.4), (112.2, 30.3), (112.4, 29.72), (112.9, 29.8), (113.15, 29.45),
                  (113.9, 29.98), (114.3, 30.58), (114.9, 30.4), (115.1, 30.22), (116.0, 29.72), (116.25, 29.75), (117.05, 30.5),
                  (117.8, 30.95), (118.37, 31.35), (118.75, 32.1), (119.45, 32.2), (120.28, 31.92), (120.85, 32.0), (121.6, 31.68)]),
    "淮水": (1.4, [(113.4, 32.4), (114.75, 32.35), (115.4, 32.45), (116.78, 32.65), (117.4, 32.93), (118.5, 33.0), (119.0, 33.55), (119.9, 34.15), (120.2, 34.3)]),
    "汉水": (1.4, [(106.8, 33.15), (107.03, 33.07), (107.8, 33.0), (108.3, 32.85), (109.03, 32.69), (110.8, 32.83), (111.5, 32.55),
                  (112.13, 32.03), (112.25, 31.7), (112.58, 31.17), (112.9, 30.5), (113.45, 30.37), (114.28, 30.57)]),
    "渭水": (1.3, [(104.6, 35.0), (105.7, 34.6), (107.15, 34.37), (108.7, 34.33), (109.5, 34.52), (110.25, 34.6)]),
    "汾水": (0.9, [(112.3, 38.5), (112.55, 37.85), (111.9, 37.0), (111.5, 36.08), (111.2, 35.6), (110.6, 35.62)]),
    "岷江": (0.9, [(103.6, 31.0), (103.9, 30.6), (103.77, 29.57), (104.63, 28.77)]),
    "湘水": (0.9, [(111.6, 26.4), (112.6, 26.9), (112.95, 28.2), (113.0, 29.1)]),
    "赣水": (0.9, [(114.93, 25.85), (115.0, 27.1), (115.9, 28.68), (116.1, 29.2)]),
    "郁水": (1.2, [(108.35, 22.8), (109.6, 23.1), (110.08, 23.4), (111.3, 23.48), (112.45, 23.05), (113.3, 23.0), (113.6, 22.7)]),
    "红河": (0.9, [(103.9, 22.5), (105.0, 21.5), (105.85, 21.03), (106.6, 20.35)]),
    "辽水": (0.9, [(123.6, 42.9), (123.0, 42.0), (122.6, 41.5), (122.1, 41.0), (121.9, 40.9)]),
}
RIVER_LABELS = [("河 水", 109.2, 41.0, 0), ("江 水", 108.95, 30.6, 22), ("江 水", 116.95, 30.1, 38),
                ("淮 水", 114.35, 32.62, 0), ("汉 水", 109.9, 32.5, 0), ("渭 水", 106.4, 34.75, -12)]
LAKES = [(112.9, 29.1, 0.9, 0.5), (116.25, 29.15, 0.5, 0.85), (120.2, 31.2, 0.65, 0.55), (100.2, 36.9, 0.95, 0.45)]
SEAS = [("渤 海", 119.6, 38.75), ("黄 海", 122.9, 35.6), ("东 海", 123.7, 29.6)]


def project(points):
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 0] *= K
    return pts


def chaikin(points, rounds=2):
    """折线倒角，让手描的折线圆润一点。"""
    pts = np.asarray(points, dtype=float)
    for _ in range(rounds):
        q = pts[:-1] * 0.75 + pts[1:] * 0.25
        r = pts[:-1] * 0.25 + pts[1:] * 0.75
        mid = np.empty((len(q) * 2, 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def halo(width=2.5, colour="#0b0d12", alpha=0.95):
    return [pe.withStroke(linewidth=width, foreground=colour, alpha=alpha)]


# ---------------------------------------------------------------------------------------------- 数据
def load():
    with io.open(os.path.join(HERE, "territory_384.json"), encoding="utf-8") as f:
        data = json.load(f)["commanderies"]
    sites, cities = [], {}
    for c in data:
        c["owner"] = c["cities"][0]["owner"]
        for city in c["cities"]:
            lon, lat = city.get("lon", c["lon"]), city.get("lat", c["lat"])
            city["own_spot"] = "lon" in city and (lon, lat) != (c["lon"], c["lat"])
            cities.setdefault(city["name"], (lon, lat))
            if city["owner"] and (city["seat"] or city["own_spot"]):
                sites.append((lon, lat, city["owner"], 1.0 if city["seat"] else 0.62))
    return data, sites, cities


# ---------------------------------------------------------------------------------------------- 势力范围的晕染
def influence(sites, land_path, nx=1000):
    """每个郡治 / 分治的城向四周“晕”出一块本势力的颜色，离得越远越淡；两家相遇处以势力强的一方为界。"""
    ny = int(nx * (LAT1 - LAT0) / ((LON1 - LON0) * K))
    lon, lat = np.linspace(LON0, LON1, nx), np.linspace(LAT1, LAT0, ny)
    gx, gy = np.meshgrid(lon * K, lat)
    ids = list(FACTIONS)
    fields = np.zeros((len(ids), ny, nx), np.float32)
    for slon, slat, owner, weight in sites:
        sigma = 1.18 if weight >= 1 else 0.55
        d2 = (gx - slon * K) ** 2 + (gy - slat) ** 2
        k = ids.index(owner)
        fields[k] = np.maximum(fields[k], weight * np.exp(-d2 / (2 * sigma ** 2)))
    owner, strength = fields.argmax(axis=0), fields.max(axis=0)
    inside = land_path.contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(ny, nx)
    rgba = np.zeros((ny, nx, 4), np.float32)
    for k, fid in enumerate(ids):
        rgba[owner == k, :3] = matplotlib.colors.to_rgb(FACTIONS[fid][1])
    rgba[..., 3] = np.clip((strength - 0.10) / 0.55, 0, 1) ** 0.9 * 0.52 * inside
    return rgba, owner, strength * inside, (lon, lat)


# ---------------------------------------------------------------------------------------------- 标注避让
class Labels:
    """贪心避让：每个标注在标记周围试几个方位，挑第一个不和已有标注 / 标记重叠的；实在放不下的小字就不放。"""

    OPTIONS = [(1, 0, "left", "center"), (-1, 0, "right", "center"), (0, 1, "center", "bottom"), (0, -1, "center", "top"),
               (1, 1, "left", "bottom"), (1, -1, "left", "top"), (-1, 1, "right", "bottom"), (-1, -1, "right", "top")]

    def __init__(self, ax, fig):
        self.ax, self.boxes, self.renderer = ax, [], fig.canvas.get_renderer()
        (self.x0, self.y0), (self.x1, self.y1) = ax.transData.transform([(LON0 * K, LAT0), (LON1 * K, LAT1)])

    def reserve(self, lon0, lat0, lon1, lat1):
        (x0, y0), (x1, y1) = self.ax.transData.transform([(lon0 * K, lat0), (lon1 * K, lat1)])
        self.boxes.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    def block(self, lon, lat, radius_px):
        x, y = self.ax.transData.transform((lon * K, lat))
        self.boxes.append((x - radius_px, y - radius_px, x + radius_px, y + radius_px))

    def _free(self, box):
        x0, y0, x1, y1 = box
        if x0 < self.x0 + 2 or x1 > self.x1 - 2 or y0 < self.y0 + 2 or y1 > self.y1 - 2:
            return False
        return not any(x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1] for b in self.boxes)

    def place(self, lon, lat, text, gap=6.0, force=True, prefer=None, **kw):
        options = ([o for o in self.OPTIONS if (o[0], o[1]) == prefer] if prefer else []) + self.OPTIONS
        for reach in (1.0, 1.8, 2.7):
            for dx, dy, ha, va in options:
                t = self.ax.annotate(text, (lon * K, lat), xytext=(dx * gap * reach, dy * gap * reach), textcoords="offset points", ha=ha, va=va, **kw)
                bb = t.get_window_extent(self.renderer)
                box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
                if self._free(box):
                    self.boxes.append(box)
                    return t
                t.remove()
        if force:
            dx, dy, ha, va = options[0]
            t = self.ax.annotate(text, (lon * K, lat), xytext=(dx * gap, dy * gap), textcoords="offset points", ha=ha, va=va, **kw)
            bb = t.get_window_extent(self.renderer)
            self.boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
            return t
        return None

    def fixed(self, lon, lat, text, **kw):
        t = self.ax.text(lon * K, lat, text, **kw)
        bb = t.get_window_extent(self.renderer)
        self.boxes.append((bb.x0 + 2, bb.y0 + 2, bb.x1 - 2, bb.y1 - 2))
        return t


# ---------------------------------------------------------------------------------------------- 题签与图例
def draw_cartouche(ax, labels, kai):
    """左侧（青藏高原的空白处）的竖排题签。"""
    labels.reserve(97.7, 27.3, 101.4, 36.3)
    ax.text(99.95 * K, 35.9, "\n".join("淝水之后"), fontproperties=kai, fontsize=46, color="#f3e6c4", ha="center", va="top", linespacing=1.02,
            path_effects=halo(4), zorder=9)
    ax.text(98.62 * K, 35.75, "\n".join("公元三八四年六月"), fontproperties=kai, fontsize=17, color="#d6b46e", ha="center", va="top", linespacing=1.05,
            path_effects=halo(3), zorder=9)
    ax.text(101.15 * K, 35.75, "\n".join("开局势力示意图"), fontproperties=kai, fontsize=17, color="#d6b46e", ha="center", va="top", linespacing=1.05,
            path_effects=halo(3), zorder=9)
    # 朱印
    sx, sy, side = 98.62 * K, 28.1, 0.92
    ax.add_patch(FancyBboxPatch((sx - side / 2, sy - side / 2), side, side, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#a81e20",
                                edgecolor="#f6d9c8", linewidth=0.8, zorder=9))
    for i, ch in enumerate("群雄并起"):
        col, row = 1 - i // 2, i % 2
        ax.text(sx + (col - 0.5) * side * 0.47, sy + (0.5 - row) * side * 0.47, ch, fontproperties=kai, fontsize=11.5, color="#fff4e8", ha="center",
                va="center", zorder=9.5)


def draw_legend(ax, labels, bold):
    """右下角（南海）的图例，三栏势力 + 一栏符号。"""
    lon0, lon1, lat0, lat1 = 111.85, 125.25, 18.05, 21.5
    labels.reserve(lon0, lat0, lon1, lat1)
    ax.add_patch(Polygon(project([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]), closed=True, facecolor="#0b0d12", edgecolor="#d6b46e",
                         linewidth=0.8, alpha=0.82, zorder=8.5))
    row = 0.345
    widths = [3.0, 2.6, 3.45, 4.1]
    x = lon0 + 0.22
    for column, width in zip(LEGEND, widths):
        y = lat1 - 0.30
        for title, ids in column:
            ax.text(x * K, y, title, fontproperties=bold, fontsize=7.2, color="#d6b46e", va="center", zorder=9)
            y -= row * 1.05
            for fid in ids:
                name, colour = FACTIONS[fid]
                ax.scatter([(x + 0.2) * K], [y], s=46, c=colour, edgecolors="#f4efe3", linewidths=0.7, zorder=9)
                ax.text((x + 0.55) * K, y, name, fontsize=7.8, color="#ece6d8", va="center", zorder=9)
                y -= row
            y -= row * 0.35
        x += width
    # 第三栏下半：日后入场
    x3 = lon0 + 0.22 + widths[0] + widths[1]
    y = lat1 - 0.30 - row * 1.05 - row * len(LEGEND[2][0][1]) - row * 0.35
    ax.text(x3 * K, y, "日后入场", fontproperties=bold, fontsize=7.2, color="#d6b46e", va="center", zorder=9)
    for text in ("凉·吕氏：385 年，武威", "孙恩：399 年，会稽"):
        y -= row
        ax.scatter([(x3 + 0.2) * K], [y], s=70, marker="*", c="#ffe08a", edgecolors="#0b0d12", linewidths=0.6, zorder=9)
        ax.text((x3 + 0.55) * K, y, text, fontsize=7.4, color="#ece6d8", va="center", zorder=9)
    # 第四栏：符号
    x4 = x3 + widths[2]
    y = lat1 - 0.30
    ax.text(x4 * K, y, "符号", fontproperties=bold, fontsize=7.2, color="#d6b46e", va="center", zorder=9)
    y -= row * 1.05
    for kind, text in (("dot", "一点一郡，颜色 = 郡治归属"), ("small", "小点：同郡里另属他家的城"), ("ring", "外圈：开局都城"), ("pass", "三角：关隘"),
                       ("none", "空心：未纳入，保持原版"), ("war", "红箭头：开局即交战"), ("battle", "淝水战场（383 年）")):
        cx = (x4 + 0.2) * K
        if kind == "dot":
            ax.scatter([cx], [y], s=50, c="#9aa3ad", edgecolors="#f4efe3", linewidths=0.7, zorder=9)
        elif kind == "small":
            ax.scatter([cx], [y], s=20, c="#9aa3ad", edgecolors="#f4efe3", linewidths=0.6, zorder=9)
        elif kind == "ring":
            ax.scatter([cx], [y], s=30, c="#9aa3ad", edgecolors="#f4efe3", linewidths=0.6, zorder=9)
            ax.scatter([cx], [y], s=110, facecolors="none", edgecolors="#f4efe3", linewidths=1.0, zorder=9)
        elif kind == "pass":
            ax.scatter([cx], [y], s=34, marker="^", c="#9aa3ad", edgecolors="#f4efe3", linewidths=0.7, zorder=9)
        elif kind == "none":
            ax.scatter([cx], [y], s=50, facecolors="none", edgecolors="#6f7783", linewidths=1.0, zorder=9)
        elif kind == "war":
            ax.add_patch(FancyArrowPatch(((x4 - 0.05) * K, y), ((x4 + 0.45) * K, y), arrowstyle="-|>", mutation_scale=9, lw=1.5, color="#ff5a4d", zorder=9))
        else:
            ax.scatter([cx], [y], s=44, marker="x", c="#ffd166", linewidths=1.6, zorder=9)
        ax.text((x4 + 0.55) * K, y, text, fontsize=7.2, color="#c9c2b2", va="center", zorder=9)
        y -= row
    ax.text((lon1 - 0.15) * K, lat0 + 0.08, "位置为按史地手工估算的示意，\n非游戏坐标", fontsize=6.4, color="#8f8a80", ha="right", va="bottom",
            linespacing=1.25, zorder=9)     # 两行：一行的话会碰到第三栏的「孙恩」


# ---------------------------------------------------------------------------------------------- 画图
def main():
    for path in (FONT, FONT_BOLD, FONT_KAI):
        font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    bold = font_manager.FontProperties(fname=FONT_BOLD)
    kai = font_manager.FontProperties(fname=FONT_KAI)

    data, sites, cities = load()
    sea, land, ink = "#09131d", "#1a1f27", "#0b0d12"
    fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * (LAT1 - LAT0) / ((LON1 - LON0) * K)), dpi=DPI, facecolor=sea)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(LON0 * K, LON1 * K)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect("equal")
    ax.axis("off")

    # 陆地、经纬网
    coast = chaikin(COAST, 2)
    ring = np.vstack([coast, [(106.6, 10.0), (90.0, 10.0), (90.0, 50.0), (130.0, 50.0), (130.0, 39.5)]])
    land_path = Path(project(ring), closed=True)
    ax.add_patch(PathPatch(land_path, facecolor=land, edgecolor="none", zorder=1))
    for island in ISLANDS:
        ax.add_patch(Polygon(project(chaikin(island + island[:1], 2)), closed=True, facecolor=land, edgecolor="#4a6885", linewidth=0.9, zorder=1))
    for lon in range(100, 126, 5):
        ax.plot([lon * K, lon * K], [LAT0, LAT1], color="#ffffff", alpha=0.04, lw=0.6, zorder=1.2)
    for lat in range(20, 43, 5):
        ax.plot([LON0 * K, LON1 * K], [lat, lat], color="#ffffff", alpha=0.04, lw=0.6, zorder=1.2)

    # 势力范围
    rgba, owner, strength, (glon, glat) = influence(sites, land_path)
    ax.imshow(rgba, extent=(LON0 * K, LON1 * K, LAT0, LAT1), origin="upper", interpolation="bilinear", zorder=2)
    for k, fid in enumerate(FACTIONS):
        mask = ((owner == k) & (strength > 0.22)).astype(float)
        if mask.any():
            ax.contour(glon * K, glat, mask, levels=[0.5], colors=[FACTIONS[fid][1]], linewidths=0.9, alpha=0.6, zorder=2.5)

    # 海岸线、江河、湖
    ax.plot(*project(coast).T, color="#56789a", lw=1.2, alpha=0.95, zorder=3)
    for name, (lw, pts) in RIVERS.items():
        p = project(chaikin(pts, 2))
        ax.plot(p[:, 0], p[:, 1], color="#3f78a8", lw=lw, alpha=0.72, solid_capstyle="round", zorder=3)
    for lon, lat, w, h in LAKES:
        ax.add_patch(Ellipse((lon * K, lat), w * K, h, facecolor="#2b5d88", edgecolor="none", alpha=0.6, zorder=3))

    fig.canvas.draw()
    labels = Labels(ax, fig)
    draw_cartouche(ax, labels, kai)
    draw_legend(ax, labels, bold)

    # 战争箭头（燕围邺就在城下，不画箭头，另有标注）
    for a, b, rad in WARS:
        (lon_a, lat_a), (lon_b, lat_b) = cities[a], cities[b]
        ax.add_patch(FancyArrowPatch((lon_a * K, lat_a), (lon_b * K, lat_b), connectionstyle="arc3,rad=%s" % rad, arrowstyle="-|>", mutation_scale=13,
                                     shrinkA=8, shrinkB=10, lw=1.6, color="#ff5a4d", alpha=0.88, zorder=5, path_effects=halo(3.0, ink, 0.75)))

    # 郡标记
    capital_cities = set(CAPITALS.values())
    for c in data:
        colour = FACTIONS[c["owner"]][1] if c["owner"] else "none"
        edge = "#f4efe3" if c["owner"] else "#6f7783"
        if c["pass"]:
            ax.scatter([c["lon"] * K], [c["lat"]], s=34, marker="^", c=colour, edgecolors=edge, linewidths=0.7, zorder=6)
            labels.block(c["lon"], c["lat"], 4.5)
        else:
            ax.scatter([c["lon"] * K], [c["lat"]], s=74, marker="o", c=colour, edgecolors=edge, linewidths=0.9, zorder=6)
            labels.block(c["lon"], c["lat"], 6)
        for city in c["cities"]:
            if city["own_spot"]:
                lon, lat = city["lon"], city["lat"]
                ax.plot([c["lon"] * K, lon * K], [c["lat"], lat], color="#d8d2c4", lw=0.5, alpha=0.3, zorder=5.5)
                ax.scatter([lon * K], [lat], s=28, marker="o", c=FACTIONS[city["owner"]][1], edgecolors="#f4efe3", linewidths=0.7, zorder=6)
                labels.block(lon, lat, 4)
    for fid, city in CAPITALS.items():
        lon, lat = cities[city]
        ax.scatter([lon * K], [lat], s=200, marker="o", facecolors="none", edgecolors=FACTIONS[fid][1], linewidths=1.6, zorder=6.5)
        ax.scatter([lon * K], [lat], s=290, marker="o", facecolors="none", edgecolors="#f4efe3", linewidths=0.5, alpha=0.7, zorder=6.5)
        labels.block(lon, lat, 9)
    # 淝水战场、吕光、孙恩
    ax.scatter([116.95 * K], [32.3], s=90, marker="x", c="#ffd166", linewidths=2.0, zorder=7)
    labels.block(116.95, 32.3, 6)
    ax.add_patch(FancyArrowPatch((98.2 * K, 40.1), (102.6 * K, 38.6), connectionstyle="arc3,rad=-0.2", arrowstyle="-|>", mutation_scale=14, lw=1.7,
                                 color="#f4efe3", alpha=0.9, linestyle=(0, (4, 2)), zorder=5, path_effects=halo(3, ink, 0.8)))
    ax.scatter([121.15 * K], [28.85], s=130, marker="*", c="#ffe08a", edgecolors=ink, linewidths=0.8, zorder=7)
    labels.block(121.15, 28.85, 7)

    # 势力名（固定位置，大字）
    for fid, lon, lat, size in FACTION_LABELS:
        name, colour = FACTIONS[fid]
        labels.fixed(lon, lat, name, fontproperties=bold, fontsize=size, color=colour, ha="center", va="center", zorder=8, path_effects=halo(3.4, ink, 0.95))
    # 都城名、事件
    for fid, city in CAPITALS.items():
        lon, lat = cities[city]
        if FACTIONS[fid][0].endswith(city):          # “秦·洛阳”的势力名就压在洛阳边上，不再重复写城名
            continue
        labels.place(lon, lat, SHOWN_AS.get(city, city), gap=9, prefer=PREFER.get(city), fontproperties=bold, fontsize=9.0, color="#fff6dd", zorder=8, path_effects=halo(2.6, ink))
    labels.place(116.95, 32.3, "淝水 · 383", gap=8, prefer=(1, -1), fontproperties=bold, fontsize=8.2, color="#ffd166", zorder=8, path_effects=halo(2.6, ink))
    labels.place(121.15, 28.85, "孙恩 399 年起事", gap=9, fontproperties=bold, fontsize=8.4, color="#ffe08a", zorder=8, path_effects=halo(2.6, ink))
    labels.place(98.25, 40.35, "吕光 385 年东归", gap=3, prefer=(1, 1), fontproperties=bold, fontsize=8.4, color="#f4efe3", zorder=8, path_effects=halo(2.6, ink))
    labels.place(114.35, 36.3, "燕军围邺", gap=9, prefer=(-1, 1), force=False, fontproperties=bold, fontsize=7.4, color="#ff8a7d", zorder=8, path_effects=halo(2.4, ink))
    # 郡名、分治的小城、关隘
    for c in data:
        if not c["pass"] and not (c["cities"][0]["name"] in capital_cities and c["name"] == c["cities"][0]["name"]):
            labels.place(c["lon"], c["lat"], c["name"], gap=5.5, force=False, fontsize=7.0, color="#c3c9d1" if c["owner"] else "#7d8692", zorder=7,
                         path_effects=halo(2.0, ink))
    for c in data:
        for city in c["cities"]:
            if city["own_spot"] and city["name"] not in capital_cities and city["name"] not in ("休屠", "临海"):
                labels.place(city["lon"], city["lat"], city["name"], gap=4.5, force=False, fontsize=6.4, color="#d9d3c5", zorder=7, path_effects=halo(2.0, ink))
    for c in data:
        if c["pass"]:
            labels.place(c["lon"], c["lat"], c["name"], gap=4.5, force=False, fontsize=6.2, color="#9aa3ad", zorder=7, path_effects=halo(1.8, ink))
    # 江河、海
    for text, lon, lat, rot in RIVER_LABELS:
        ax.text(lon * K, lat, text, fontsize=7.0, color="#6fa3cf", alpha=0.85, rotation=rot, ha="center", va="center", style="italic", zorder=4,
                path_effects=halo(1.8, ink, 0.7))
    for text, lon, lat in SEAS:
        ax.text(lon * K, lat, text, fontsize=11, color="#3f6283", alpha=0.9, ha="center", va="center", zorder=4)

    out = os.path.join(HERE, "..", "images", "territory_384.jpg")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=sea)
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    quality = 92
    while True:
        img.save(out, "JPEG", quality=quality, optimize=True, progressive=True)
        if os.path.getsize(out) < MAX_BYTES or quality <= 60:
            break
        quality -= 2
    print("territory_384.jpg", img.size, os.path.getsize(out), "bytes, quality", quality)


if __name__ == "__main__":
    main()
