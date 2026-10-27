import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi
D='/mnt/project-files/Kin-Dza_Dza/'; Wd='/tmp/claude-0/w/'
S=3; CW,CH=336*S,480*S
def load(k,f):
    a=np.asarray(Image.open(D+f).convert('RGB')).astype(np.float32)
    al=np.load(Wd+k+'_alpha.npy').astype(np.float32); hz=np.load(Wd+k+'_hz.npy')
    rb=np.asarray(Image.open(Wd+k+'_mask.png').convert('L'))>128
    return a,al,hz,rb
def bbox(m):
    ys,xs=np.nonzero(m); return xs.min(),ys.min(),xs.max(),ys.max()
def rs(img,s,interp=cv2.INTER_AREA):
    return cv2.resize(img,(round(img.shape[1]*s),round(img.shape[0]*s)),interpolation=interp)

# ---- Pepelac: 2/3 of width, centred
pa,pal,phz,prb=load('pep','Пепелац.jpg')
x0,y0,x1,y1=bbox(ndi.binary_opening(prb,iterations=3))
s=(CW*2/3)/(x1-x0)
A=rs(pa,s); AL=rs(pal,s); HZ=phz*s
ox=round(CW/2-(x0+x1)/2*s); oy=round(CH/2-(y0+y1)/2*s)
canvas=np.zeros((CH,CW,3),np.float32); alpha=np.zeros((CH,CW),np.float32)
h,w=A.shape[:2]
# paste photo region
ys0,ys1=max(0,oy),min(CH,oy+h); xs0,xs1=max(0,ox),min(CW,ox+w)
canvas[ys0:ys1,xs0:xs1]=A[ys0-oy:ys1-oy,xs0-ox:xs1-ox]
alpha[ys0:ys1,xs0:xs1]=AL[ys0-oy:ys1-oy,xs0-ox:xs1-ox]
horizon=np.interp(np.arange(CW),np.arange(len(HZ))*s+ox,HZ+oy)
# extend sand below the photo by mirroring its bottom strip, with a soft seam
if ys1<CH:
    # extend sand below the photo: tile the shadow-free bottom rows, flipping every other tile
    need=CH-ys1; M=60; B=0
    src=canvas[ys1-M:ys1].copy()
    tiles=[]; flip=True
    while sum(t.shape[0] for t in tiles)<need:
        tiles.append(src[::-1] if flip else src); flip=not flip
    fill=np.vstack(tiles)
    fill=np.roll(fill,CW//3,axis=1)                      # shift so the tile seam isn't a straight copy
    canvas[ys1:CH]=fill[:need]
    alpha[ys1:CH]=1
ship=np.zeros((CH,CW),np.float32)
sm=rs(ndi.binary_fill_holes(prb).astype(np.float32),s,cv2.INTER_LINEAR)
ship[ys0:ys1,xs0:xs1]=sm[ys0-oy:ys1-oy,xs0-ox:xs1-ox]
ship_rgb=canvas.copy()
sand_ref=canvas[int(horizon.mean())+10:int(horizon.mean())+60].reshape(-1,3)
sand_mean,sand_std=sand_ref.mean(0),sand_ref.std(0)

def side(k,f,target_w,cx,pad_below=0.06,match=0.6):
    a,al,hz,rb=load(k,f)
    if k=='whl':
        obj=(al>0.5)
        rows=np.arange(a.shape[0])[:,None]; obj=obj&(rows<hz[None,:]-2)
    else:
        obj=rb|((al>0.5)&(np.arange(a.shape[0])[:,None]<hz[None,:]-2))
    obj=ndi.binary_opening(obj,iterations=2)
    lab,n=ndi.label(obj); sz=ndi.sum(obj,lab,range(1,n+1)); big=1+np.argmax(sz)
    keep=np.isin(lab,1+np.nonzero(sz>sz.max()*0.02)[0])
    bx0,by0,bx1,by1=bbox(keep)
    hzc=float(np.median(hz[bx0:bx1]))
    sc=target_w/(bx1-bx0)
    # crop: object box, down to a bit below horizon
    cy1=int(min(a.shape[0],hzc+(by1-by0)*pad_below+ (bx1-bx0)*0.08))
    mx=int((bx1-bx0)*0.08)
    cx0,cx1=max(0,bx0-mx),min(a.shape[1],bx1+mx)
    crop=a[by0:cy1,cx0:cx1]; cal=al[by0:cy1,cx0:cx1].copy()
    hzl=hz[cx0:cx1]-by0
    rows=np.arange(crop.shape[0])[:,None]
    # ground fades out under the horizon and towards crop sides
    depth=(rows-hzl[None,:])
    fade=np.clip(1-depth/max(4,(cy1-by0-hzc+by0)),0,1)
    xr=np.abs(np.linspace(-1,1,crop.shape[1]))[None,:]
    fadex=np.clip((1-xr)/0.35,0,1)
    g=np.where(depth>0,fade*fadex,1.0)
    cal=cal*g
    # colour match towards the pepelac sand
    gm=(depth>0)&(depth<40)
    ref=crop[np.broadcast_to(gm,crop.shape[:2])]
    m,sd=ref.mean(0),ref.std(0)+1e-3
    adj=(crop-m)*(1-match+match*sand_std/sd)+ (m*(1-match)+sand_mean*match)
    crop=np.clip(adj,0,255)
    C=rs(crop,sc); CA=rs(cal,sc)
    # vertical position: crop horizon -> canvas horizon at cx
    hzs=(hzc-by0)*sc
    px=round(cx-C.shape[1]/2); py=round(horizon[int(cx)]-hzs)
    return C,CA,px,py
layers=[side('tri','Тринога1.jpeg',150,86), side('whl','Колесо обозрения.jpg',237,CW-78)]
for C,CA,px,py in layers:
    h2,w2=CA.shape
    x0_,y0_=max(0,px),max(0,py); x1_,y1_=min(CW,px+w2),min(CH,py+h2)
    c=C[y0_-py:y1_-py,x0_-px:x1_-px]; a_=CA[y0_-py:y1_-py,x0_-px:x1_-px][...,None]
    canvas[y0_:y1_,x0_:x1_]=canvas[y0_:y1_,x0_:x1_]*(1-a_)+c*a_
    alpha[y0_:y1_,x0_:x1_]=np.maximum(alpha[y0_:y1_,x0_:x1_],a_[...,0])
# ship stays in front
canvas=canvas*(1-ship[...,None])+ship_rgb*ship[...,None]
rgba=np.dstack([np.clip(canvas,0,255),np.clip(alpha,0,1)*255]).astype(np.uint8)
big=Image.fromarray(rgba,'RGBA'); big.save(Wd+'ground_x3.png')
small=big.resize((336,480),Image.LANCZOS); small.save(Wd+'ground.png')
print('horizon at final px:',horizon.min()/S,horizon.mean()/S,horizon.max()/S,'ship box y',(oy+y0*s)/S,(oy+y1*s)/S)
