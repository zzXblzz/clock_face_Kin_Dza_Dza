# Сборка проекта EasyFace для циферблата «Пепелац» (Mi Band 9 Pro, 336x480, Lua/LVGL).
#   py build_face.py  -> Pepelac_face/face/Pepelac/  (Pepelac.fprj, app/lua/*, images/, AOD/)
# Что делает:
#  * фоны skies/bg_<погода>.png -> app/lua/bg_<погода>.png
#  * кадры «КУ» (55..100, стабилизация как в ku_preview.py) обрезаются по общей рамке -> ku_NN.png
#  * подбирает тонировку человека под погоду (LVGL img_recolor) по тому, как skies.py
#    перекрашивает землю, и вписывает её в main.lua
#  * рисует цифры шрифтом Oswald (600 — основной экран, 200 — AOD)
import os, sys, shutil, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

D = '/mnt/project-files/Kin-Dza_Dza/'
P = D + 'Pepelac_face/'
SRC = P + 'src/'
# Браслет кэширует картинки по имени файла: после переустановки циферблата со старыми
# именами он показывает старые размеры/содержимое. Поэтому у каждой сборки свой префикс
# имён и своя папка проекта — меняйте TAG при каждой новой версии.
TAG = 'v4_'
OUT = sys.argv[1] if len(sys.argv) > 1 else P + 'face/Pepelac_%s/' % TAG.strip('_')
LUA = OUT + 'app/lua/'
AOD = OUT + 'AOD/'
W, H = 336, 480

FRAMES = list(range(55, 101))   # все 46 кадров, 24 к/с ~ 1.9 с
FPS = 24
SCALE, CX, FEET = 0.93, 168, 460   # 0.62 * 1.5
WEATHERS = ['unknown', 'sunny', 'cloudy', 'snow', 'rain', 'storm', 'night']
# параметры ground() из skies.py: gain, tint, sat, gamma
GROUND = {'unknown': (1, (1, 1, 1), 1, 1), 'sunny': (1.08, (1.04, 1.0, .94), 1.08, 1),
          'cloudy': (.97, (.99, 1, 1.02), .9, 1), 'snow': (.95, (.97, 1, 1.06), .7, 1),
          'rain': (.72, (.93, .97, 1.05), .65, 1), 'storm': (.62, (.9, .92, 1.1), .55, 1),
          'night': (.42, (.75, .85, 1.25), .45, 1.15)}

# ---------- разметка (по preview.png) ----------
TIME_Y = 92          # верх цифр времени
TIME_H = 92          # высота цифр (кегль подбирается под неё)
SMALL_H = 21         # высота цифр даты и заряда
DATE_X, DATE_Y = 28, 13
BATT_R, BATT_Y = 308, 13      # правый край заряда

for d in (LUA, OUT + 'images', AOD + 'images'):
    os.makedirs(d, exist_ok=True)

# ---------- 1. кадры «КУ» ----------
def ku_frames():
    src = [Image.open(D + 'КУ_кадры/frame_%03d.png' % n).convert('RGBA') for n in FRAMES]
    feet, torso = [], []
    for f in src:
        a = np.array(f)[:, :, 3] > 128
        rows = np.nonzero(a.sum(1) > 25)[0]
        top, bot = rows.min(), rows.max(); h = bot - top
        _, bx = np.nonzero(a[top + int(.15 * h):top + int(.5 * h)])
        feet.append(bot); torso.append(bx.mean())
    N = np.array(FRAMES); feet = np.array(feet, float); torso = np.array(torso)
    k = N >= 70
    feet = np.polyval(np.polyfit(N[k], feet[k], 1), N)
    torso = np.convolve(np.pad(torso, 2, mode='edge'), np.ones(5) / 5, 'valid')
    w, h = round(640 * SCALE), round(360 * SCALE)
    full = []
    for f, fy, tx in zip(src, feet, torso):
        c = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        c.alpha_composite(f.resize((w, h), Image.LANCZOS), (round(CX - tx * SCALE), round(FEET - fy * SCALE)))
        a = np.array(c); a[a[..., 3] < 8] = 0          # срезаем почти прозрачную «пыль»
        full.append(Image.fromarray(a))
    box = None
    for c in full:
        b = c.getbbox()
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    return [c.crop(box) for c in full], box

# ---------- 2. тонировка под погоду ----------
def ground_tf(rgb, gain, tint, sat, gamma):
    lum = rgb.mean(1, keepdims=True)
    x = np.clip(lum + (rgb - lum) * sat, 0, 1) ** gamma * gain * np.asarray(tint)
    return np.clip(x, 0, 1)

def fit_recolor(px, w):
    """out = (1-a)*in + a*c  ~  ground() из skies.py. Возвращает (a, c[0..255]).
    k = 1-a подбирается по контрасту (яркости), c — так, чтобы совпал средний цвет."""
    y = ground_tf(px, *GROUND[w])
    k = min(1.0, y.mean(1).std() / px.mean(1).std())
    k = min(k, (y.mean(0) / px.mean(0)).min() * 0.98)   # чтобы c не уходил в минус: средний цвет совпадёт точно
    a = 1 - k
    if a < 0.02: return 0, (255, 255, 255)
    b = y.mean(0) - k * px.mean(0)
    return a, tuple(int(v) for v in np.clip(b / a * 255, 0, 255))

def recolor(img, a, c):
    x = np.array(img).astype(np.float32)
    x[..., :3] = x[..., :3] * (1 - a) + np.asarray(c, np.float32) * a
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))

# ---------- 3. цифры ----------
def font(weight, cap_h):
    path = SRC + 'fonts/oswald-latin-%d-normal.woff' % weight
    size = cap_h
    for _ in range(4):                       # подгоняем кегль под высоту цифры '0'
        f = ImageFont.truetype(path, size)
        b = f.getbbox('0'); size = round(size * cap_h / (b[3] - b[1]))
    return ImageFont.truetype(path, size)

def glyphs(chars, f, color, shadow, hpad=None, cell=None):
    """Каждый символ — PNG одной высоты; цифры в ячейке одинаковой ширины."""
    top = f.getbbox('0')[1]; cap = f.getbbox('0')[3] - top
    pad = 2 * shadow if shadow else 1
    hp = pad if hpad is None else hpad   # поле слева/справа: соседние картинки не должны перекрываться
    dw = max(f.getbbox(ch)[2] - f.getbbox(ch)[0] for ch in '0123456789')
    out = {}
    for ch in chars:
        bb = f.getbbox(ch)
        gw = (cell or dw) if ch.isdigit() else bb[2] - bb[0] + 2
        im = Image.new('RGBA', (gw + 2 * hp, cap + 2 * pad), (0, 0, 0, 0))
        x = hp + (gw - (bb[2] - bb[0])) // 2 - bb[0]
        m = Image.new('L', im.size, 0)
        ImageDraw.Draw(m).text((x, pad - top), ch, font=f, fill=255)
        if shadow:
            s = m.filter(ImageFilter.GaussianBlur(shadow)).point(lambda v: min(255, int(v * 1.6)))
            sh = Image.new('RGBA', im.size, (0, 0, 0, 0)); sh.putalpha(s.point(lambda v: v * 150 // 255))
            im.alpha_composite(sh, (0, 1))
        fg = Image.new('RGBA', im.size, color); fg.putalpha(m)
        im.alpha_composite(fg)
        out[ch] = im
    return out, pad, hp

def save_png(im, path, colors=None):
    if colors: im = im.quantize(colors, method=Image.FASTOCTREE, dither=Image.FLOYDSTEINBERG)
    im.save(path, optimize=True)

# ================= сборка =================
frames, (KX, KY, _, _) = ku_frames()
NF = len(frames)
for i, f in enumerate(frames):
    save_png(f, LUA + 'ku_%02d.png' % i, 256)       # 256 цветов: ~7 КБ на кадр
px = np.array(frames[-1]).reshape(-1, 4); px = px[px[:, 3] > 200][:, :3] / 255.
TINT = {w: fit_recolor(px, w) for w in WEATHERS}
for w in WEATHERS:
    shutil.copy(P + 'skies/bg_%s.png' % w, LUA + 'bg_%s.png' % w)
# луна: фазы из skies.py (центр 70,70 в 1x, картинка 120x120)
MN = 30
for i in range(MN):
    shutil.copy(P + 'skies/moon/moon_%02d.png' % i, LUA + 'moon_%02d.png' % i)
MX, MY = 70 - 60, 70 - 60

# --- цифры основного экрана ---
big, BP, BH = glyphs('0123456789:', font(600, TIME_H), (255, 255, 255, 255), 3, hpad=3)
small, SPD, SH = glyphs('0123456789.%-', font(600, SMALL_H), (255, 255, 255, 255), 2, hpad=1)
for ch, im in big.items():   im.save(LUA + 't_%s.png' % ('c' if ch == ':' else ch))
for ch, im in small.items(): im.save(LUA + 's_%s.png' % {'.': 'dot', '%': 'pct', '-': 'dash'}.get(ch, ch))
SP = 6                                           # зазор между цифрами времени (= 2*BH: картинки встык)
DW = big['0'].width - 2 * BH; CW = big[':'].width - 2 * BH
TOTAL = 4 * DW + 2 * SP + CW + 2 * SP + 0
x = (W - TOTAL) // 2
TX = []
for k in range(5):
    TX.append(x - BH); x += (CW if k == 2 else DW) + SP
SDW = small['0'].width - 2 * SH; SDOT = small['.'].width - 2 * SH; SPCT = small['%'].width - 2 * SH
SS = 2                                           # зазор между мелкими символами (= 2*SH)
DX = []; x = DATE_X
for k in range(5):
    DX.append(x - SH); x += (SDOT if k == 2 else SDW) + SS

# --- AOD: тонкий шрифт, без тени ---
thin, AP, _ = glyphs('0123456789:', font(200, TIME_H), (225, 225, 225, 255), 0)
ADW = thin['0'].width; ACW = thin[':'].width
ATOTAL = 4 * ADW + ACW + 2 * SP
AX = (W - ATOTAL) // 2
for ch in '0123456789':
    thin[ch].save(AOD + 'images/h_%s.png' % ch); thin[ch].save(AOD + 'images/m_%s.png' % ch)
thin[':'].save(AOD + 'images/colon.png')
Image.new('RGBA', (W, H), (0, 0, 0, 255)).save(AOD + 'images/black.png')

# ---------- main.lua ----------
tint_lua = ',\n'.join('    %s = { "#%02x%02x%02x", %d }' % (w, *TINT[w][1], round(TINT[w][0] * 255)) for w in WEATHERS)
lua = open(SRC + 'main_template.lua', encoding='utf-8').read()
for k, v in {'TAG': TAG, 'NF': NF, 'MN': MN, 'MX': MX, 'MY': MY, 'FPS': FPS, 'KX': KX, 'KY': KY, 'TINT': tint_lua,
             'TX': ', '.join(map(str, TX)), 'TY': TIME_Y - BP,
             'DX': ', '.join(map(str, DX)), 'DY': DATE_Y - SPD,
             'BR': BATT_R, 'BY': BATT_Y - SPD, 'SPD': SH, 'SDW': SDW, 'SPCT': SPCT, 'SS': SS}.items():
    lua = lua.replace('@%s@' % k, str(v))
open(LUA + 'main.lua', 'w', encoding='utf-8', newline='\n').write(lua)

# ---------- .fprj (ASCII без BOM!) ----------
FPRJ = '''<?xml version="1.0" encoding="utf-16" ?>
<FaceProject DeviceType="367">
    <Screen Title="%s" Bitmap="preview.png">
%s
    </Screen>
</FaceProject>
'''
def write_fprj(path, title, widgets):
    open(path, 'w', encoding='ascii', newline='\r\n').write(FPRJ % (title, '\n'.join('        ' + w for w in widgets)))

write_fprj(OUT + 'Pepelac.fprj', 'Pepelac',
           ['<Widget Shape="34" Name="app_lua%2Fmain.lua" X="0" Y="0" Width="336" Height="480" Alpha="0" Visible_Src="0" />'])
th = thin['0'].height
hl = '|'.join('h_%s.png' % c for c in '01234567890')
ml = '|'.join('m_%s.png' % c for c in '01234567890')
write_fprj(AOD + 'AOD.fprj', 'AOD', [
    '<Widget Shape="30" Name="image_00" X="0" Y="0" Width="336" Height="480" Alpha="255" Visible_Src="0" Bitmap="black.png" />',
    '<Widget Shape="32" Name="widget_01" X="%d" Y="%d" Width="%d" Height="%d" Alpha="255" Digits="2" Alignment="1" Spacing="0" Blanking="0" Visible_Src="0" Value_Src="811" BitmapList="%s" />' % (AX, TIME_Y, 2 * ADW, th, hl),
    '<Widget Shape="30" Name="image_02" X="%d" Y="%d" Width="%d" Height="%d" Alpha="255" Visible_Src="0" Bitmap="colon.png" />' % (AX + 2 * ADW + SP, TIME_Y, ACW, th),
    '<Widget Shape="32" Name="widget_03" X="%d" Y="%d" Width="%d" Height="%d" Alpha="255" Digits="2" Alignment="1" Spacing="0" Blanking="0" Visible_Src="0" Value_Src="1011" BitmapList="%s" />' % (AX + 2 * ADW + ACW + 2 * SP, TIME_Y, 2 * ADW, th, ml),
])

# ---------- превью ----------
def compose(w, hh='10', mm='08', date='26.10', batt='80', frame=-1, moon=16):
    im = Image.open(LUA + 'bg_%s.png' % w).convert('RGBA')
    if w == 'night': im.alpha_composite(Image.open(LUA + 'moon_%02d.png' % moon), (MX, MY))
    a, c = TINT[w]
    im.alpha_composite(recolor(frames[frame], a, c), (KX, KY))
    for k, ch in enumerate(hh + ':' + mm): im.alpha_composite(big[ch], (TX[k], TIME_Y - BP))
    for k, ch in enumerate(date): im.alpha_composite(small[ch], (DX[k], DATE_Y - SPD))
    s = batt + '%'; x = BATT_R - (len(batt) * (SDW + SS) + SPCT)
    for ch in s:
        im.alpha_composite(small[ch], (x - SH, BATT_Y - SPD)); x += (SPCT if ch == '%' else SDW) + SS
    return im

def compose_aod(hh='10', mm='08'):
    im = Image.new('RGBA', (W, H), (0, 0, 0, 255))
    x = AX
    for ch in hh: im.alpha_composite(thin[ch], (x, TIME_Y)); x += ADW
    x += SP; im.alpha_composite(thin[':'], (x, TIME_Y)); x += ACW + SP
    for ch in mm: im.alpha_composite(thin[ch], (x, TIME_Y)); x += ADW
    return im

compose('sunny').convert('RGB').save(OUT + 'images/preview.png')
compose_aod().convert('RGB').save(AOD + 'images/preview.png')
shots = [compose(w) for w in WEATHERS] + [compose_aod()]
grid = Image.new('RGB', (len(shots) * (W + 10) - 10, H + 30), (30, 30, 30))
dr = ImageDraw.Draw(grid)
for i, s in enumerate(shots):
    grid.paste(s.convert('RGB'), (i * (W + 10), 30))
    dr.text((i * (W + 10) + 8, 6), (WEATHERS + ['AOD'])[i], fill=(220, 220, 220), font=ImageFont.truetype(SRC + 'fonts/oswald-latin-600-normal.woff', 18))
grid.save(P + 'face/preview_face.png')
# анимация на ночном фоне — проверить, что человек не «светится»
anim = [compose('night', frame=i).convert('RGB') for i in range(NF)]
anim[0].save(P + 'face/preview_night_anim.gif', save_all=True, append_images=anim[1:] + [anim[-1]] * 24, duration=1000 // FPS, loop=0)
for f in os.listdir(LUA):
    if f.endswith('.png') and not f.startswith(TAG): os.replace(LUA + f, LUA + TAG + f)
print('frames', NF, 'box', (KX, KY), 'tint', TINT)
