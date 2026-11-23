"""Local NVIDIA calibration: pinned RAM upload, device copy, and unbuffered file upload."""
import argparse
import ctypes as C
from ctypes import wintypes as W
from hashlib import file_digest
import json
from pathlib import Path
import platform
from random import Random
import subprocess
from time import perf_counter_ns

M=2**20
SIZES=(8*M,16*M,32*M,25_755_648,51_511_296,153_944_064)


def sha(path):
    with path.open('rb') as f: return file_digest(f,'sha256').hexdigest()


def benchmark(path, rounds):
    api=C.WinDLL('nvcuda.dll')
    kernel=C.WinDLL('kernel32',use_last_error=True)
    P=C.c_void_p
    U=C.c_uint64
    S=C.c_size_t
    I=C.c_int
    def cuda(name,*types):
        fn=getattr(api,name);fn.argtypes=types;fn.restype=I
        def call(*args):
            if (status:=fn(*args))!=0: raise RuntimeError(f'{name}: CUDA {status}')
        return call
    def win(name,result,*types):
        fn=getattr(kernel,name);fn.argtypes=types;fn.restype=result
        return fn
    init=cuda('cuInit',C.c_uint); device=cuda('cuDeviceGet',C.POINTER(I),I)
    create=cuda('cuCtxCreate_v2',C.POINTER(P),C.c_uint,I)
    destroy=cuda('cuCtxDestroy_v2',P); sync=cuda('cuCtxSynchronize')
    malloc=cuda('cuMemAlloc_v2',C.POINTER(U),S); release=cuda('cuMemFree_v2',U)
    host_alloc=cuda('cuMemAllocHost_v2',C.POINTER(P),S);host_free=cuda('cuMemFreeHost',P)
    ram=cuda('cuMemcpyHtoD_v2',U,P,S); copy=cuda('cuMemcpyDtoD_v2',U,U,S)
    open_file=win('CreateFileW',W.HANDLE,W.LPCWSTR,W.DWORD,W.DWORD,P,W.DWORD,W.DWORD,W.HANDLE)
    fseek=win('SetFilePointerEx',W.BOOL,W.HANDLE,C.c_longlong,P,W.DWORD)
    read_file=win('ReadFile',W.BOOL,W.HANDLE,P,W.DWORD,C.POINTER(W.DWORD),P)
    close=win('CloseHandle',W.BOOL,W.HANDLE)
    def disk(handle,ptr,size,offset):
        got=W.DWORD()
        if not fseek(handle,offset,None,0) or not read_file(handle,ptr,size,C.byref(got),None): raise C.WinError(C.get_last_error())
        if got.value!=size: raise RuntimeError(f'short unbuffered read {got.value}/{size}')
    init(0);dev=I();device(C.byref(dev),0);ctx=P();create(C.byref(ctx),0,dev)
    dst,src,host=U(),U(),P();handle=None
    try:
        for p in (dst,src):malloc(C.byref(p),max(SIZES))
        host_alloc(C.byref(host),max(SIZES))
        assert host.value % 4096==0 and all(n%4096==0 for n in SIZES)
        C.memset(host,51,max(SIZES))
        # NO_BUFFERING: bypass OS page cache. Drive internal state is not reset.
        handle=open_file(str(path.resolve()),0x80000000,1,None,3,0x20000000,None)
        if handle in (None,P(-1).value): raise C.WinError(C.get_last_error())
        ram(src,host,max(SIZES)); copy(dst,src,max(SIZES));sync()
        disk(handle,host,8*M,0); ram(dst,host,8*M);sync()
        rng=Random(8033)
        order=[(kind,size,i) for kind in ('ram','ssd','copy') for size in SIZES for i in range(rounds)]
        rng.shuffle(order)
        out=[]
        for kind,size,i in order:
            offset=rng.randrange((path.stat().st_size-size)//M)*M if kind=='ssd' else None
            sync(); start=perf_counter_ns()
            if kind=='ram': ram(dst,host,size)
            elif kind=='copy': copy(dst,src,size)
            else: disk(handle,host,size,offset); ram(dst,host,size)
            sync(); ns=perf_counter_ns()-start
            out.append(dict(kind=kind,bytes=size,trial=i,offset=offset,ns=ns))
        return out
    finally:
        if handle not in (None,P(-1).value):close(handle)
        if host:host_free(host)
        for p in (dst,src):
            if p.value:release(p)
        destroy(ctx)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('file',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--rounds',type=int,default=9)
    a=p.parse_args()
    result=dict(platform=platform.platform(),data_sha256=sha(a.file),tool_sha256=sha(Path(__file__)),
                device=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
                rounds=a.rounds,samples=benchmark(a.file,a.rounds))
    with a.out.open('x',encoding='utf-8') as out: json.dump(result,out,indent=2)


if __name__=='__main__': main()
