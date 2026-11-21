import sys
def crc8(d):
    c=0
    for b in d:
        c^=b
        for _ in range(8):
            c=((c<<1)^0x07)&0xff if c&0x80 else (c<<1)&0xff
    return c
def crc16(d):
    c=0
    for b in d:
        c^=b<<8
        for _ in range(8):
            c=((c<<1)^0x8005)&0xffff if c&0x8000 else (c<<1)&0xffff
    return c
src,dst,mode=sys.argv[1:4]
d=open(src,'rb').read()
hdr=d[:42]; body=d[42:]
# find frames
starts=[]
i=0
while i<len(body)-1:
    if body[i]==0xff and body[i+1]==0xf8:
        starts.append(i)
    i+=1
frames=[]
pos=0
# greedy: a frame is slice [s,e) with valid crc16 at e-2
cands=starts+[len(body)]
s=0
out=[]
k=0
while s<len(body):
    for e in cands:
        if e<=s+8: continue
        if crc16(body[s:e-2])==int.from_bytes(body[e-2:e],'big'):
            frames.append(body[s:e]); s=e; break
    else:
        raise SystemExit("frame parse failed at %d"%s)
res=bytearray(hdr)
for n,f in enumerate(frames):
    f=bytearray(f)
    assert f[2]>>4==7 and f[2]&15==0
    # utf8 frame number length
    b0=f[4]; ln=1 if b0<0x80 else (2 if b0<0xe0 else 3)
    hdr_end=4+ln+2  # before crc8
    bs=int.from_bytes(f[4+ln:4+ln+2],'big')+1
    h=bytearray(f[:4+ln])
    if mode in('bs','both') and bs==4096:
        h[2]=(12<<4)|(h[2]&15)
    else:
        h+=f[4+ln:4+ln+2]
    if mode in('rate','both'):
        h[2]=(h[2]&0xf0)|5
    nh=bytes(h)
    nf=nh+bytes([crc8(nh)])+bytes(f[hdr_end+1:-2])
    nf+=crc16(nf).to_bytes(2,'big')
    res+=nf
open(dst,'wb').write(res)
print(mode,len(frames),'frames')
