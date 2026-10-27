import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi
D='/mnt/project-files/Kin-Dza_Dza/'
W='/tmp/claude-0/w/'
# file, ground test (B-R below), dark-vs-sky delta, red test
CFG={'pep':('Пепелац.jpg',0,14),'tri':('Тринога1.jpeg',-10,14),'whl':('Колесо обозрения.jpg',10,10)}
def analyse(key):
    f,tg,dd=CFG[key]
    a=np.asarray(Image.open(D+f).convert('RGB')).astype(np.float32)
    h,w,_=a.shape
    sm=cv2.GaussianBlur(a,(0,0),1.2)
    R,G,B=sm[...,0],sm[...,1],sm[...,2]
    lum=(R+G+B)/3; br=B-R
    rows=np.arange(h)[:,None]
    rb=np.asarray(Image.open(W+key+'_mask.png').convert('L')).astype(np.float32)/255
    # sky reference = grey closing (removes thin dark things)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
    bgl=cv2.morphologyEx(lum,cv2.MORPH_CLOSE,k)
    dark=(bgl-lum)>dd
    warm=br<tg
    obj=dark|((R-B)>25) if key=='tri' else dark|warm
    lab,n=ndi.label(obj); sz=ndi.sum(obj,lab,range(1,n+1))
    obj=np.isin(lab,1+np.nonzero(sz>=60)[0])
    if key!='whl': obj|=rb>0.5
    # horizon: first row where ground (warm) persists 12 rows, on columns free of objects
    g=warm.astype(np.uint8)
    run=ndi.uniform_filter1d(g.astype(float),25,axis=0,origin=-12)>0.9
    first=np.argmax(run,axis=0).astype(float)
    if key=='whl':
        objcol=(dark[:int(h*0.75)]).sum(0)>3
    else:
        objcol=(rb[:int(h*0.7)]>0.3).sum(0)>0
    objcol=ndi.binary_dilation(objcol,iterations=w//80)
    xs=np.arange(w); ok=~objcol&(first>h*0.3)
    hz=np.interp(xs,xs[ok],ndi.median_filter(first,15)[ok])
    hz=ndi.uniform_filter1d(hz,w//100)
    if key=='whl':  # wheel: mound only counts as ground, horizon is the far flat line
        pass
    alpha=np.where(rows>=hz[None,:],1.0,obj.astype(float))
    if key=='pep':  # above horizon keep only the ship (rembg) minus sky-like halo
        ship=ndi.binary_erosion(ndi.binary_fill_holes(rb>0.5),iterations=2)
        alpha=np.where(rows>=hz[None,:],1.0,ship.astype(float))
    alpha=cv2.GaussianBlur(alpha.astype(np.float32),(0,0),0.7)
    return a.astype(np.uint8),alpha,hz
if __name__=='__main__':
    out=[]
    for k in CFG:
        a,al,hz=analyse(k)
        np.save(W+k+'_alpha.npy',al.astype(np.float16)); np.save(W+k+'_hz.npy',hz)
        im=Image.fromarray(a); m=Image.fromarray((al*255).astype(np.uint8))
        bg=Image.new('RGB',im.size,(255,0,255)); bg.paste(im,mask=m); bg.thumbnail((700,520)); out.append(bg)
        print(k,'horizon',hz.min(),np.median(hz),hz.max())
    W2=sum(i.width for i in out); c=Image.new('RGB',(W2,520)); x=0
    for i in out: c.paste(i,(x,0)); x+=i.width
    c.save(W+'cuts3.jpg')
