"""Rebuild the guide's diagrams from its archived snapshot; requires Pillow.

Run from any directory: python path/to/generate_figures.py
The conceptual diagrams are teaching examples, not planner measurements.
"""
from pathlib import Path
import json
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, MUTED = '#18334c', '#506779'
BLUE, TEAL, ORANGE, PINK = '#2366bb', '#008778', '#ba6a08', '#b93588'
BG, LINE = '#f4f7fb', '#d7e1eb'


def font(size=26, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def text(x, y, value, size=26, color=INK, bold=False):
    draw.text((x, y), value, font=font(size, bold), fill=color, spacing=12)


def canvas(title, subtitle, height=780):
    global im, draw
    im = Image.new('RGB', (1440, height), BG)
    draw = ImageDraw.Draw(im)
    text(48, 28, title, 38, bold=True)
    text(48, 89, subtitle, 24, MUTED)


def card(box, title, body, color=BLUE, size=26):
    x, y, x2, y2 = box
    draw.rounded_rectangle(box, 18, fill='white', outline=LINE, width=2)
    draw.rounded_rectangle((x, y, x+8, y2), 4, fill=color)
    text(x+24, y+20, title, 29, color, True)
    text(x+24, y+77, body, size)


def arrow(a, b, color=BLUE, width=5):
    draw.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1]-a[1], b[0]-a[0])
    pts = [b, (b[0]-17*math.cos(ang-.5), b[1]-17*math.sin(ang-.5)),
           (b[0]-17*math.cos(ang+.5), b[1]-17*math.sin(ang+.5))]
    draw.polygon(pts, fill=color)


def save(name):
    im.save(ROOT/name, optimize=True)


def pipeline():
    canvas('锚点从哪里来', '先找到能站的位置，再筛出能看清的方向，最后减少站位和照片。', 920)
    card((48, 150, 465, 325), '输入一  地图', '栅格：哪里能走\nAABB：物体在哪里、多大')
    card((510, 150, 927, 325), '输入二  机器人', '长、宽、安全距离\n决定底盘需要多大空间')
    card((972, 150, 1392, 325), '输入三  相机', '内参、安装位姿、距离\n决定怎样拍才算合格')
    cards = [((48, 405, 465, 575), 'A1  安全站位', '每个安全格中心一个候选\n包含全部安全连通区域', TEAL),
             ((510, 405, 927, 575), 'A2  合格观察关系', '每个物体反查附近站位\n逐方向检查入镜与遮挡', BLUE),
             ((972, 405, 1392, 575), 'A3  选站与选照片', '新增覆盖优先，质量破平票\n再删除冗余站位和照片', PINK)]
    for box, title, body, color in cards:
        card(box, title, body, color)
    arrow((465, 490), (505, 490)); arrow((927, 490), (967, 490))
    arrow((250, 325), (250, 400), TEAL)
    arrow((719, 325), (320, 400), TEAL)
    arrow((1180, 325), (780, 400))
    arrow((1180, 580), (1180, 665), PINK)
    card((48, 670, 1392, 854), '最终复核与输出', '重新计算最终拍摄动作 → 锚点位置 + 拍摄朝向 + 合格覆盖物体 + 未覆盖原因\n当前位置到各站的路径、访问顺序和真实图像采集，需要后续执行模块完成。', TEAL)
    save('01_pipeline.png')


def safety():
    canvas('安全站位要给整个机器人留空间', '教学示意：空闲格中心不一定安全；靠墙的格子可能容不下底盘转身。', 860)
    x0, y0, s = 65, 173, 49
    for r in range(10):
        for c in range(12):
            blocked = c in (0, 11) or r in (0, 9) or (c == 6 and r < 6)
            safe = not blocked and c in (2, 3, 4, 8, 9) and r in range(2, 8)
            color = '#40556b' if blocked else '#c8eee5' if safe else '#fce9c8'
            draw.rectangle((x0+c*s,y0+r*s,x0+(c+1)*s,y0+(r+1)*s), fill=color, outline='white', width=2)
            if safe:
                xx, yy = x0+(c+.5)*s, y0+(r+.5)*s
                draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill=TEAL)
    for x, label, col in [(65,'障碍', '#40556b'), (230,'空闲但太近','#fce9c8'), (480,'安全候选','#c8eee5')]:
        draw.rectangle((x,700,x+24,724),fill=col)
        text(x+34,691,label,24)
    cx, cy = 990, 355
    draw.ellipse((cx-150,cy-150,cx+150,cy+150), fill='#edf6fc', outline=BLUE, width=4)
    draw.rectangle((cx-83,cy-98,cx+83,cy+98), fill='#c6d9ed', outline=BLUE,width=3)
    draw.ellipse((cx-7,cy-7,cx+7,cy+7),fill=PINK)
    arrow((cx,cy),(cx+146,cy), PINK)
    text(cx+30,cy-50,'安全半径',24,PINK)
    text(800,535,'r = √(长² + 宽²) / 2 + 安全距离',26,BLUE,True)
    text(800,590,'当前配置：r ≈ 0.643 m',29,bold=True)
    text(800,650,'外接圆包住底盘的所有转向姿态。\n格中心到障碍边界的距离须 ≥ r。',25)
    text(65,786,'左图仅解释概念，不对应当前地图，也不按右侧尺寸计算。',23,MUTED)
    save('02_safe_stations.png')


def poses():
    canvas('一个锚点，可以拍多个方向', '锚点是底盘站立的位置；照片对应“这个位置 + 一次指定朝向”。', 790)
    cx, cy, r = 355, 420, 208
    for deg in range(0,360,30):
        a=math.radians(deg)
        end=(cx+r*math.cos(a),cy-r*math.sin(a))
        arrow((cx,cy),end,PINK if deg in (0,90,210) else '#aab8c6',4)
        text(end[0]-24,end[1]-16,f'{deg}°',22,PINK if deg in (0,90,210) else MUTED)
    draw.ellipse((cx-29,cy-29,cx+29,cy+29),fill=TEAL,outline='white',width=4)
    text(cx-17,cy-23,'A',30,'white',True)
    card((720,175,1375,370),'默认每隔 30° 检查一次','0°、30°、…、330°，共 12 个候选方向。\n最后只保留有用的方向，不必拍满 12 张。',BLUE,25)
    card((720,405,1375,650),'示例  A 站留下三张照片','朝 0° 看见杯子；朝 90° 看见书；\n朝 210° 看见盒子。\n统计为 1 个锚点、3 张照片。',PINK,25)
    text(65,711,'图中方向只是教学例子。世界坐标：0° 沿 +X，90° 沿 +Y；角度描述底盘朝向。',24,MUTED)
    save('03_station_and_yaw.png')


def quality():
    canvas('看得到，还要通过三道画面门槛', '教学示意：三个量各有分母，必须分别达标；不能只看一个总分。', 970)
    card((48,150,480,660),'① 入镜比例 ≥ 80%','完整投影有多少落在画面里？',BLUE,23)
    card((504,150,936,660),'② 未遮挡比例 ≥ 60%','入镜部分有多少没有被挡住？',TEAL,23)
    card((960,150,1392,660),'③ 可见面积 ≥ 400 px²','没被挡住的部分够不够大？',ORANGE,23)
    for xx in (78,534,990):
        draw.rectangle((xx,303,xx+370,505),fill='#eaf0f6',outline='#7892ab',width=3)
    draw.rectangle((400,333,496,473),fill='#9bc8ed',outline=BLUE,width=3)
    draw.line((448,303,448,505),fill=PINK,width=7)
    text(88,540,'画面外的部分也在分母中。\n只露出一半：入镜率约 50%。',23)
    draw.rectangle((620,333,820,473),fill='#9edbcd',outline=TEAL,width=3)
    draw.rectangle((620,333,700,473),fill='#e5a36b',outline=ORANGE,width=3)
    text(544,540,'橙色挡住 40%，剩下 60%。\n目标后面的物体不算遮挡。',23)
    draw.rectangle((1140,390,1160,410),fill=TEAL)
    text(1000,540,'面积 400 px²，相当于 20×20。\n这是面积门槛，并非边长门槛。',23)
    card((48,710,1392,901),'先过门槛，再用质量分比较','例：完整投影 2,000 px²，入镜 1,800 px²，未遮挡 1,260 px²。\n入镜率 90%，未遮挡率 70%，可见面积 1,260 px² → 三项通过；quality = 0.9 × 0.7 = 0.63。',TEAL,26)
    save('04_quality_gates.png')


def voting():
    canvas('动态投票：每轮只数还没完成的物体', '教学例子：同一物体在同一站有多个合格方向，也只投一票。', 895)
    xs=[65,360,630,900,1170]
    rows=[['站位','杯子','书','盒子','初始票数'],['A','✓','✓','—','2'],['B','—','✓','✓','2'],['C','—','—','✓','1']]
    # Use symbols covered by the bundled CJK font.
    rows=[[v.replace('✓','有') for v in row] for row in rows]
    for ri,row in enumerate(rows):
        yy=170+ri*95
        draw.rounded_rectangle((48,yy,1392,yy+85),12,fill='#dfeaf6' if ri==0 else 'white')
        for x,v in zip(xs,row): text(x,yy+20,v,28,bold=ri==0)
    card((48,600,680,825),'第一轮：A 与 B 都有 2 票','若 A 的最佳质量之和更高，先选 A。\n杯子和书完成，撤销它们在所有站的票。',BLUE,25)
    card((720,600,1392,825),'第二轮：只剩盒子需要覆盖','B 从 2 票变成 1 票，C 仍是 1 票。\n再比盒子的质量；若 B 更高，就选 B。',TEAL,25)
    save('05_dynamic_votes.png')


def pruning():
    canvas('删掉冗余站位，再挑真正要拍的照片', '教学例子：删除的条件是覆盖不丢失，不是距离近，也不是看起来重复。', 860)
    card((48,150,680,480),'站位层：检查每个站的不可替代贡献','A 能看见：杯子、书\nB 能看见：书、盒子\nC 能看见：杯子、盒子\n若三站都已入选，删除任一站仍能覆盖三物体。',BLUE,26)
    card((720,150,1392,480),'逐个删除，并及时更新覆盖计数','先删除 A → 只保留 B、C。\n此时 B 的书、C 的杯子都不可替代。\n不能继续沿用删除前的计数把 B、C 也删掉。\n保留集合中每一站都至少有一份独有贡献。',TEAL,25)
    card((48,535,1392,790),'照片层：在全部保留站位的方向里，再做一次覆盖选择','每次挑能新增覆盖最多物体的一张照片，平票比较质量；然后逐张删除冗余照片。\n例如：B 的 0° 看书、90° 看盒子，C 的 180° 看杯子和盒子。\n先选 C 的 180°，再选 B 的 0° → 2 个锚点，2 张照片即可完成这个教学例子。',PINK,26)
    save('06_prune_and_photos.png')


def real_map(snapshot):
    canvas('当前默认地图的真实选点结果', '2026-10-19 运行快照：箭头是拍摄朝向，不是行走路线。A1～A4 按输出顺序编号。', 1060)
    scene,result=snapshot['scene'],snapshot['result']
    rows=scene['grid']['rows']; h,w=len(rows),len(rows[0]); scale=4
    x0,y0=60,165
    for r,line in enumerate(rows):
        for c,v in enumerate(line):
            color='#40556b' if v=='#' else '#a6aebb' if v=='?' else '#e8edf3'
            if snapshot['safe_rows'][r][c]=='1': color='#d2eee5'
            draw.rectangle((x0+c*scale,y0+r*scale,x0+(c+1)*scale,y0+(r+1)*scale),fill=color)
    origin=scene['grid']['origin']; res=scene['grid']['resolution']
    def point(p):return x0+(p[0]-origin[0])/res*scale,y0+(h-(p[1]-origin[1])/res)*scale
    for obj in scene['objects']:
        a,b=point(obj['aabb']['min']),point(obj['aabb']['max'])
        draw.rectangle((a[0],b[1],b[0],a[1]),fill='#f2c990',outline=ORANGE,width=2)
        xx,yy=point(obj['position'])
        label=obj['id']; box=draw.textbbox((xx+5,yy-19),label,font=font(17))
        draw.rectangle(box,fill='white');text(xx+5,yy-19,label,17)
    colors=[BLUE,PINK,TEAL,'#8549ba']
    for i,anchor in enumerate(result['anchors']):
        xx,yy=point(anchor['position']); col=colors[i%len(colors)]
        for action in anchor['observations']:
            a=action['yaw_rad'];arrow((xx,yy),(xx+48*math.cos(a),yy-48*math.sin(a)),col,4)
        draw.ellipse((xx-17,yy-17,xx+17,yy+17),fill=col,outline='white',width=3)
        text(xx-10,yy-16,str(i+1),22,'white',True)
        yy2=170+i*170
        angles='、'.join(str(round(math.degrees(a['yaw_rad']))) for a in anchor['observations'])
        card((900,yy2,1390,yy2+145),f"A{i+1}  ·  站位 {anchor['station_index']}",
             f"位置 ({anchor['position'][0]:.3f}, {anchor['position'][1]:.3f}) m\n朝向 {angles}°",col,22)
    text(900,870,f"{result['anchor_count']} 个锚点 / {result['photo_count']} 张照片",30,bold=True)
    text(900,920,f"{len(result['covered_object_ids'])} / {len(scene['objects'])} 个物体模型合格",27,TEAL,True)
    text(60,990,'深灰：栅格障碍   浅灰：非安全空闲格   浅绿：安全站位区域   橙色：物体 AABB',23,MUTED)
    save('07_current_map.png')


if __name__ == '__main__':
    pipeline(); safety(); poses(); quality(); voting(); pruning()
    real_map(json.loads((ROOT/'run_snapshot.json').read_text()))
    print('Generated seven PNG figures.')
