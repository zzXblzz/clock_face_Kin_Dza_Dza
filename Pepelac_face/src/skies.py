# Фоны неба по погоде для циферблата «Пепелац» (336x480).
# Небо рисуется процедурно в 3x (1008x1440), затем поверх кладётся слой земли
# ground_1008x1440.png (небо в нём прозрачное, горизонт ~y=332 в 1x),
# земля подкрашивается под освещение, результат уменьшается до 336x480.
# Выход: Pepelac_face/skies/bg_<погода>.png (готовый фон) и sky_<погода>.png (только небо).
import os, numpy as np, cv2
from PIL import Image
D = '/mnt/project-files/Kin-Dza_Dza/Pepelac_face/'
OUT = D + 'skies/'
S = 3; W, H = 336 * S, 480 * S; HZ = 332 * S
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
rng = np.random.default_rng(7)

def fbm(oct=6, base=4, seed=0, aspect=1.0):
    r = np.random.default_rng(seed); acc = np.zeros((H, W), np.float32); amp = 1; tot = 0
    for o in range(oct):
        n = base * 2 ** o
        g = r.random((max(2, int(n * H / W / aspect)), n)).astype(np.float32)
        acc += amp * cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= 0.5
    return acc / tot

def blur(a, s): return cv2.GaussianBlur(a, (0, 0), s)
def sstep(a, b, x): t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
def lerp(c0, c1, t): return np.asarray(c0, np.float32) * (1 - t[..., None]) + np.asarray(c1, np.float32) * t[..., None]

def gradient(top, mid, bot):
    t = np.clip(YY / HZ, 0, 1)
    return np.where((t < .6)[..., None], lerp(top, mid, t / .6), lerp(mid, bot, (t - .6) / .4))

def over(base, col, a):
    a = np.clip(a, 0, 1)[..., None]
    return base * (1 - a) + np.asarray(col, np.float32) * a

# ---------- кучевые облака: шары-«клубы», освещённые сверху, рисуются сверху вниз ----------
NOISE = None
def cumulus(clouds, seed, lit, shade, dark_base, rim=0.0, light=(-.25, -.85, .45)):
    """clouds: [(cx, base_y, width, height)] в 1x. Возвращает (rgb, alpha)."""
    global NOISE
    if NOISE is None: NOISE = fbm(5, 24, 99)
    r = np.random.default_rng(seed)
    L = np.array(light, np.float32); L /= np.linalg.norm(L)
    lit, shade, dark_base = (np.asarray(c, np.float32) for c in (lit, shade, dark_base))
    col = np.zeros((H, W, 3), np.float32); alpha = np.zeros((H, W), np.float32)
    for cx, by, w, h in clouds:
        cx, by, w, h = cx * S, by * S, w * S, h * S
        puffs = []
        for _ in range(int(6 + w / (14 * S))):                   # крупные купола
            u = r.uniform(-1, 1)
            rad = h * r.uniform(.35, .55) * (1 - .45 * abs(u))
            top = h * (1 - u * u) ** .6
            py = by - rad * .6 - r.uniform(0, 1) * max(0, top - rad * 1.3)
            puffs.append((cx + u * (w / 2 - rad * .5), py, rad))
        for px, py, rad in list(puffs):                          # мелкие клубы по верхнему краю
            for _ in range(r.integers(2, 5)):
                ang = r.uniform(-np.pi * .95, -np.pi * .05)
                rr = rad * r.uniform(.3, .5)
                puffs.append((px + np.cos(ang) * rad * .8, py + np.sin(ang) * rad * .8, rr))
        puffs.sort(key=lambda p: p[1] - p[2])                    # верхние первыми, нижние поверх
        for px, py, rad in puffs:
            x0, x1 = int(max(0, px - rad * 1.2)), int(min(W, px + rad * 1.2))
            y0, y1 = int(max(0, py - rad * 1.2)), int(min(H, by + 4 * S))
            if x1 <= x0 or y1 <= y0: continue
            X, Y = XX[y0:y1, x0:x1], YY[y0:y1, x0:x1]
            nz_ = NOISE[y0:y1, x0:x1]
            reff = rad * (1 + (nz_ - .5) * .35)
            nx, ny = (X - px) / reff, (Y - py) / reff
            d2 = nx * nx + ny * ny
            a = sstep(1.0, .8, np.sqrt(d2)) * sstep(by + 3 * S, by - 6 * S, Y)   # плоское основание
            nzz = np.sqrt(np.clip(1 - d2, 0, 1))
            lam = np.clip(nx * L[0] + ny * L[1] + nzz * L[2], 0, 1)
            hgt = np.clip((by - Y) / max(h, 1), 0, 1)               # 0 у основания, 1 на верху
            c = shade + (lit - shade) * (.25 + .75 * lam)[..., None]
            c = c + (dark_base - c) * (np.clip(1 - hgt * 2.2, 0, 1) * .75)[..., None]
            if rim: c += rim * 255 * np.clip(d2 * 1.4 - .5, 0, 1)[..., None] * lam[..., None]
            ca = col[y0:y1, x0:x1]; aa = alpha[y0:y1, x0:x1]
            col[y0:y1, x0:x1] = ca * (1 - a[..., None]) + c * a[..., None]
            alpha[y0:y1, x0:x1] = aa + (1 - aa) * a
    col *= (1 + (fbm(5, 16, seed + 2) - .5) * .1)[..., None]
    return np.clip(col, 0, 255), alpha

def layer_cloud(sky, clouds, seed, **kw):
    c, a = cumulus(clouds, seed, **kw)
    return sky * (1 - a[..., None]) + c * a[..., None]

# ---------- осадки ----------
def snow(img, n, seed, front=False):
    r = np.random.default_rng(seed); out = img.copy()
    for size, cnt, bl, alpha in ((1.1, n, .6, .75), (2.0, n // 3, 1.2, .85), (3.6, n // 10, 2.6, .8)):
        m = np.zeros((H, W), np.float32)
        ys = r.uniform(0, H if front else HZ + 30 * S, cnt); xs = r.uniform(0, W, cnt)
        for x, y in zip(xs, ys):
            cv2.circle(m, (int(x), int(y)), max(1, int(size * S * r.uniform(.7, 1.3))), 1, -1, cv2.LINE_AA)
        m = blur(m, bl * S) * (1.6 if bl > 1 else 1)
        if not front: m *= sstep(HZ + 30 * S, HZ - 20 * S, YY)
        out = over(out, (250, 252, 255), np.clip(m, 0, 1) * alpha)
    return out

def rain(img, n, seed, ang=12, col=(200, 210, 225), op=.45, front=False, lim=None):
    r = np.random.default_rng(seed); m = np.zeros((H, W), np.float32)
    dx, dy = np.sin(np.radians(ang)), np.cos(np.radians(ang))
    for _ in range(n):
        x, y = r.uniform(-60, W + 60), r.uniform(-60, (H if front else HZ + 20 * S))
        L = r.uniform(14, 34) * S
        cv2.line(m, (int(x), int(y)), (int(x + dx * L), int(y + dy * L)), r.uniform(.4, 1), max(1, int(S * r.uniform(.5, 1.0))), cv2.LINE_AA)
    m = blur(m, .5 * S)
    if lim is not None: m *= lim
    return over(img, col, m * op)

# ---------- молния: смещение средней точки ----------
def bolt(pts, depth, r, disp):
    if depth == 0: return [pts]
    (x0, y0), (x1, y1) = pts
    mx, my = (x0 + x1) / 2 + r.normal(0, disp), (y0 + y1) / 2 + r.normal(0, disp * .3)
    return bolt(((x0, y0), (mx, my)), depth - 1, r, disp / 2) + bolt(((mx, my), (x1, y1)), depth - 1, r, disp / 2)

def lightning(img, start, end, seed):
    r = np.random.default_rng(seed); core = np.zeros((H, W), np.float32)
    def draw(a, b, w, depth, disp, br):
        segs = bolt((a, b), depth, r, disp)
        for (p, q) in segs:
            cv2.line(core, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), br, w, cv2.LINE_AA)
        return segs
    segs = draw(start, end, 3 * S // 2, 7, 70 * S, 1.0)
    for i in r.choice(len(segs), 4, replace=False):            # ответвления
        p = segs[i][0]
        q = (p[0] + r.uniform(-90, 90) * S, p[1] + r.uniform(50, 110) * S)
        draw(p, q, S // 2 + 1, 5, 25 * S, .7)
    glow = blur(core, 3 * S) * 2.5 + blur(core, 14 * S) * 4 + blur(core, 45 * S) * 3
    out = over(img, (200, 190, 255), np.clip(glow, 0, .8))
    return over(out, (255, 255, 255), np.clip(core * 1.3 + blur(core, 1.2 * S), 0, 1)), glow

# ---------- земля: подкраска под освещение ----------
G = np.asarray(Image.open(D + 'ground_1008x1440.png').convert('RGBA')).astype(np.float32)
def ground(gain=1.0, tint=(1, 1, 1), sat=1.0, gamma=1.0):
    rgb = G[..., :3] / 255
    lum = rgb.mean(2, keepdims=True)
    rgb = lum + (rgb - lum) * sat
    rgb = np.clip(rgb, 0, 1) ** gamma * gain * np.asarray(tint, np.float32)
    return np.clip(rgb * 255, 0, 255), G[..., 3] / 255

def finish(name, sky, gnd=None, front=None):
    gr, ga = gnd if gnd else ground()
    img = sky * (1 - ga[..., None]) + gr * ga[..., None]
    # мягкая дымка у горизонта, чтобы небо и земля сошлись
    haze = np.exp(-((YY - HZ) / (10 * S)) ** 2) * .25
    haze *= 1 - sstep(62 * S, 80 * S, XX) * sstep(282 * S, 264 * S, XX)   # не по корпусу Пепелаца
    hc = sky[HZ - 6 * S].mean(0)
    img = img * (1 - haze[..., None]) + hc * haze[..., None]
    if front: img = front(img)
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((336, 480), Image.LANCZOS).save(OUT + 'bg_%s.png' % name)
    Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8)).resize((336, 480), Image.LANCZOS).save(OUT + 'sky_%s.png' % name)
    print(name)

# облака раскладываем так, чтобы они были видны вокруг Пепелаца (он по центру, верх ~y=85)
FAIR = [(60, 120, 150, 70), (275, 95, 140, 60), (175, 50, 170, 55), (30, 250, 120, 45), (300, 230, 110, 45), (150, 300, 240, 40)]
HEAVY = [(40, 130, 230, 120), (250, 115, 260, 120), (150, 60, 300, 90), (40, 245, 200, 80), (290, 250, 200, 85), (170, 305, 380, 70), (160, 180, 330, 80)]

# 1) неизвестно — чистое голубое небо
unknown = gradient((70, 135, 215), (125, 180, 235), (190, 220, 240))
finish('unknown', unknown)

# 2) ясно — яркое солнце
sun_c = (262 * S, 62 * S)
d = np.hypot(XX - sun_c[0], YY - sun_c[1])
clear = gradient((40, 120, 225), (100, 175, 245), (200, 230, 250))
clear = over(clear, (255, 250, 225), np.exp(-d / (95 * S)) * .75)        # ореол
clear = over(clear, (255, 255, 240), np.exp(-(d / (34 * S)) ** 2) * .95)
rays = (np.cos(np.arctan2(YY - sun_c[1], XX - sun_c[0]) * 14) * .5 + .5) ** 6
clear = over(clear, (255, 250, 220), rays * np.exp(-d / (70 * S)) * .35)
clear = over(clear, (255, 255, 252), sstep(24 * S, 20 * S, d))           # диск
finish('sunny', clear, ground(1.08, (1.04, 1.0, .94), 1.08))

# 3) облачно / пасмурно / туман / пыль — белые кучевые облака
cl = gradient((80, 140, 210), (135, 185, 230), (195, 220, 238))
cl = layer_cloud(cl, FAIR, 11, lit=(255, 255, 255), shade=(185, 195, 212), dark_base=(160, 170, 188))
finish('cloudy', cl, ground(.97, (.99, 1, 1.02), .9))

# 4) снег — серое небо, идёт снег
sn = gradient((150, 160, 178), (185, 192, 205), (215, 220, 228))
sn = layer_cloud(sn, HEAVY, 21, lit=(225, 230, 238), shade=(160, 168, 182), dark_base=(140, 148, 162))
sn = snow(sn, 900, 3)
finish('snow', sn, ground(.95, (.97, 1, 1.06), .7), front=lambda im: snow(im, 140, 4, front=True))

# 5) дождь / ливень — тёмные кучевые облака, дождь
rn = gradient((70, 76, 88), (100, 106, 118), (135, 140, 150))
rn = layer_cloud(rn, HEAVY, 31, lit=(120, 125, 135), shade=(45, 48, 56), dark_base=(30, 32, 38))
rn = rain(rn, 1300, 5)
finish('rain', rn, ground(.72, (.93, .97, 1.05), .65),
       front=lambda im: rain(im, 220, 6, op=.3, front=True))

# 6) гроза — грозовые облака и молния
st = gradient((35, 32, 52), (62, 58, 82), (95, 90, 110))
st = layer_cloud(st, HEAVY + [(100, 200, 200, 90)], 41, lit=(105, 100, 130), shade=(28, 26, 40), dark_base=(18, 16, 26), rim=.08)
st, glow = lightning(st, (52 * S, 150 * S), (40 * S, 334 * S), 9)
st = rain(st, 700, 7, ang=16, op=.35)
flash = np.clip(blur(glow, 120 * S) * 3, 0, .35)
gr, ga = ground(.62, (.9, .92, 1.1), .55)
gr = gr + flash[..., None] * np.array([150, 150, 190], np.float32)
finish('storm', st, (np.clip(gr, 0, 255), ga))

# 7) ночь — 21:00–06:00 при любой погоде
nt = gradient((5, 8, 25), (14, 22, 52), (35, 45, 80))
r = np.random.default_rng(1); m = np.zeros((H, W), np.float32)
for _ in range(700):
    x, y = r.uniform(0, W), r.uniform(0, HZ)
    m[int(y), int(x)] = r.uniform(.3, 1) ** 2 * (1 - y / HZ * .6)
stars = blur(m, .6 * S) * 9 + blur(m, 2 * S) * 6
nt = over(nt, (235, 240, 255), np.clip(stars, 0, 1))
nt = over(nt, (60, 70, 140), (fbm(6, 3, 51) > .58) * .0 + np.clip((fbm(6, 3, 51) - .5) * 1.2, 0, .25) * (YY < HZ))  # лёгкая туманность
# Луна рисуется отдельно (фазы по дате, см. ниже) — на ночном фоне её нет.
MOON_C, MOON_R, MOON_N = (70, 70), 20, 30      # центр и радиус в 1x, число фаз
def moon_phases():
    """skies/moon/moon_NN.png: NN = 0..29, возраст луны (NN+0.5)/30 синодического месяца.
    0 — новолуние, 15 — полнолуние; растущая освещена справа (северное полушарие)."""
    os.makedirs(OUT + 'moon/', exist_ok=True)
    B = 60 * S                                        # полуразмер картинки (с ореолом)
    yy, xx = np.mgrid[-B:B, -B:B].astype(np.float32) + .5
    R = MOON_R * S; d = np.hypot(xx, yy)
    disk = sstep(R + .7 * S, R - .7 * S, d)
    ny = np.clip(yy / R, -1, 1); nx = xx / R
    half = np.sqrt(np.clip(1 - ny ** 2, 0, 1))
    tex = fbm(5, 12, 61)[:2 * B, :2 * B]
    craters = 1 - np.clip((tex - .5) * 1.4, 0, .25)
    moon = np.array([240, 236, 215], np.float32) * craters[..., None]
    for i in range(MOON_N):
        th = 2 * np.pi * (i + .5) / MOON_N
        t = np.cos(th) * half                         # x терминатора
        edge = .06
        lit = sstep(t - edge, t + edge, nx) if th <= np.pi else sstep(-t + edge, -t - edge, nx)
        lit = lit * disk
        frac = (1 - np.cos(th)) / 2                   # освещённая доля
        rgb = np.zeros((2 * B, 2 * B, 3), np.float32); a = np.zeros((2 * B, 2 * B), np.float32)
        halo = np.exp(-d / (40 * S)) * .35 * frac * (1 - disk) * sstep(B * .97, B * .35, d) ** 2   # ореол гаснет до края картинки — без «квадрата»
        rgb[:] = (200, 210, 240); a = halo
        dark = disk * (1 - lit)                       # тёмная часть: чуть видна, закрывает звёзды
        rgb = rgb * (1 - dark[..., None]) + np.array([10, 14, 38], np.float32) * dark[..., None]
        a = a * (1 - dark) + dark * .9
        rgb = rgb * (1 - lit[..., None]) + moon * lit[..., None]
        a = a * (1 - lit) + lit
        im = np.dstack([np.clip(rgb, 0, 255), np.clip(a * 255, 0, 255)]).astype(np.uint8)
        Image.fromarray(im, 'RGBA').resize((2 * B // S, 2 * B // S), Image.LANCZOS).save(OUT + 'moon/moon_%02d.png' % i)
gr, ga = ground(.42, (.75, .85, 1.25), .45, 1.15)
# лунный свет слева сверху, блик на корпусе
finish('night', nt, (gr, ga))
moon_phases()
