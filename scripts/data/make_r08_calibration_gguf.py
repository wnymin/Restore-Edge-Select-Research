"""Build a local depth-truncated GGUF calibration fixture, not a released model.

Only little-endian GGUF v3, non-split qwen35 is supported. Keep all tensor
shapes and bytes unchanged; drop layers 16..31 and update block_count.
No derived weights are included in research artifacts.
"""
from argparse import ArgumentParser
from pathlib import Path
import re
from struct import pack, unpack, calcsize

FORMATS={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}


def read_exact(f,n):
    b=f.read(n)
    if len(b)!=n: raise ValueError('Truncated GGUF')
    return b


def get(f,typ):
    return unpack('<'+typ,read_exact(f,calcsize('<'+typ)))[0]


def string(f):
    return read_exact(f,get(f,'Q')).decode('utf-8')


def encoded(s):
    raw=s.encode('utf-8')
    return pack('<Q',len(raw))+raw


def skip_values(f,k,n=1):
    if k in FORMATS: f.seek(n*calcsize('<'+FORMATS[k]),1)
    elif k==8:
        for _ in range(n): f.seek(get(f,'Q'),1)
    elif k==9 and n==1:
        typ,count=get(f,'I'),get(f,'Q')
        if typ==9: raise ValueError('Nested array unsupported')
        skip_values(f,typ,count)
    else: raise ValueError(f'Unsupported value {k}')


def aligned(n,a): return (n+a-1)//a*a


def truncate(src,dst):
    with src.open('rb') as f:
        if read_exact(f,4)!=b'GGUF' or get(f,'I')!=3: raise ValueError('Require little endian GGUF v3')
        count,nkeys=get(f,'Q'),get(f,'Q')
        meta={}; align=32
        for _ in range(nkeys):
            key=string(f); start=f.tell(); typ=get(f,'I')
            skip_values(f,typ); end=f.tell(); f.seek(start)
            raw=read_exact(f,end-start)
            if key in meta or key.startswith('split.'): raise ValueError('Require a non-split input with unique keys')
            if key=='general.alignment':
                assert typ==4
                align=unpack('<I',raw[4:])[0]
            if typ==9 and key.startswith('qwen35.') and unpack('<Q',raw[8:16])[0]==32:
                raise ValueError(f'Per-layer field requires explicit projection: {key}')
            meta[key]=raw
        assert align>0 and align&(align-1)==0
        assert meta['general.architecture']==pack('<I',8)+encoded('qwen35')
        assert meta['qwen35.block_count']==pack('<II',4,32)
        meta['qwen35.block_count']=pack('<II',4,16)
        meta['general.name']=pack('<I',8)+encoded('R08 qwen35-4b first 16 layers (profiling only)')
        tensors=[]; layer_ids=set()
        for _ in range(count):
            key=string(f); n=get(f,'I'); shape=tuple(get(f,'Q') for _ in range(n))
            dtype,offset=get(f,'I'),get(f,'Q')
            layer=re.match(r'^blk\.(\d+)\.',key)
            keep=not layer or int(layer[1])<16
            if layer: layer_ids.add(int(layer[1]))
            if not layer and key not in {'token_embd.weight','output_norm.weight','output.weight'}:
                raise ValueError(f'Unexpected non-layer tensor {key}')
            assert offset%align==0
            tensors.append((offset,key,shape,dtype,keep))
        assert layer_ids==set(range(32)),layer_ids
        start=aligned(f.tell(),align)
        end=src.stat().st_size-start
        tensors.sort()
        assert tensors[0][0]==0
        kept=[]; new_offset=0
        for idx,(offset,key,shape,dtype,keep) in enumerate(tensors):
            size=(tensors[idx+1][0] if idx+1<len(tensors) else end)-offset
            assert size>0
            if keep:
                kept.append((offset,size,key,shape,dtype,new_offset)); new_offset+=aligned(size,align)
        with dst.open('xb') as g:
            g.write(b'GGUF'+pack('<IQQ',3,len(kept),len(meta)))
            for key,raw in meta.items(): g.write(encoded(key)+raw)
            for offset,size,key,shape,dtype,tensor_offset in kept:
                g.write(encoded(key)+pack('<I',len(shape))+pack('<'+'Q'*len(shape),*shape)+pack('<IQ',dtype,tensor_offset))
            g.write(bytes(aligned(g.tell(),align)-g.tell()))
            for offset,size,*_ in kept:
                f.seek(start+offset)
                left=size
                while left:
                    buf=read_exact(f,min(left,8<<20)); g.write(buf); left-=len(buf)
                g.write(bytes(aligned(size,align)-size))
    print(f'Wrote derived 16-layer fixture: {len(kept)}/{count} tensors; {dst.stat().st_size} bytes')


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args(); truncate(a.source,a.destination)


if __name__=='__main__': main()
