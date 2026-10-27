from PIL import Image, ImageDraw
D='/mnt/project-files/Kin-Dza_Dza/'; W='/tmp/claude-0/w/'
g=Image.open(W+'ground.png')
def sky(f,box):
    return Image.open(D+f).convert('RGB').crop(box).resize((336,480),Image.LANCZOS)
skies=[('пустое небо',Image.new('RGB',(336,480),(255,0,255))),
       ('небо Пепелаца',sky('Пепелац.jpg',(0,0,700,780))),
       ('голубое',sky('Колесо обозрения.jpg',(1230,0,1440,420))),
       ('облака',sky('Тринога1.jpeg',(0,0,600,500)))]
out=Image.new('RGB',(336*4+30,480),(30,30,30))
for i,(n,s) in enumerate(skies):
    c=s.convert('RGBA'); c.alpha_composite(g); out.paste(c.convert('RGB'),(i*346,0))
out.save(W+'preview4.png'); print('ok')
