# Превью анимации «КУ» (кадры 55..100) поверх ground_336x480.png
# Масштаб взят по Back_clean.jpg: Уэф там ~189px ростом на экране 480px,
# в присевшей позе (кадр 85) он 302px в кадре 640x360 -> SCALE ~0.62
# Стабилизация: в исходнике камера едет, человек «плывёт» вверх и влево.
# Для каждого кадра находим ступни и центр торса и ставим их в одну точку экрана.
import numpy as np
from PIL import Image
D='/mnt/project-files/Kin-Dza_Dza/'
SCALE=0.62
CX, FEET = 168, 460          # центр экрана по X и уровень ступней
N=range(55,101)
g=Image.open(D+'Pepelac_face/ground_336x480.png')
base=Image.new('RGBA',g.size,(120,170,230,255)); base.alpha_composite(g)
src=[Image.open(D+'КУ_кадры/frame_%03d.png'%n) for n in N]

feet,torso=[],[]
for f in src:
    a=np.array(f)[:,:,3]>128
    cnt=a.sum(1); rows=np.nonzero(cnt>25)[0]   # >25 px в строке: отсекаем тонкую тень
    top,bot=rows.min(),rows.max(); h=bot-top
    _,bx=np.nonzero(a[top+int(.15*h):top+int(.5*h)])
    feet.append(bot); torso.append(bx.mean())
feet=np.array(feet,float); torso=np.array(torso)
# до 70-го кадра ступни обрезаны краем кадра -> берём линейный ход камеры по кадрам 70..100
k=[i for i,n in enumerate(N) if n>=70]
p=np.polyfit(np.array(N)[k],feet[k],1); feet=np.polyval(p,np.array(N))
torso=np.convolve(np.pad(torso,2,mode='edge'),np.ones(5)/5,'valid')   # сглаживание дрожания

w,h=round(640*SCALE),round(360*SCALE)
frames=[]
for f,fy,tx in zip(src,feet,torso):
    c=base.copy()
    c.alpha_composite(f.resize((w,h),Image.LANCZOS),(round(CX-tx*SCALE),round(FEET-fy*SCALE)))
    frames.append(c.convert('RGB'))
out=D+'Pepelac_face/ku_preview/'
frames[0].save(out+'ku_on_ground.gif',save_all=True,append_images=frames[1:],duration=42,loop=0)
strip=Image.new('RGB',(336*4+30,480),(30,30,30))
for i,j in enumerate([0,15,30,45]): strip.paste(frames[j],(i*346,0))
strip.save(out+'ku_frames_55_70_85_100.png'); print('ok', p)
