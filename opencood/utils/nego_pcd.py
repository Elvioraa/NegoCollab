"""Strict XYZI reader for RGB-encoded OPV2V and intensity-encoded V2XSet PCD."""
from pathlib import Path
import numpy as np


def read_pcd(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Missing PCD DATA header: ' + str(path))
            parts = line.decode('ascii').strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == 'DATA':
                payload = stream.read()
                break
    fields = header.get('FIELDS', header.get('FIELD'))
    sizes = list(map(int,header['SIZE']))
    types = header['TYPE']
    counts = list(map(int,header.get('COUNT',['1']*len(fields))))
    n = int(header.get('POINTS',[str(int(header['WIDTH'][0])*int(header.get('HEIGHT',['1'])[0]))])[0])
    dtype = np.dtype([(f, np.dtype({'F':'<f','U':'<u','I':'<i'}[t]+str(s)), (c,))
                      if c != 1 else (f, np.dtype({'F':'<f','U':'<u','I':'<i'}[t]+str(s)))
                      for f,s,t,c in zip(fields,sizes,types,counts)])
    mode = header['DATA'][0]
    if mode == 'binary':
        if len(payload) != n*dtype.itemsize:
            raise ValueError('Truncated PCD payload: '+str(path))
        points = np.frombuffer(payload,dtype=dtype,count=n)
    elif mode == 'ascii':
        flat = np.loadtxt(payload.splitlines(),ndmin=2)
        if flat.shape != (n,sum(counts)):
            raise ValueError('Invalid ASCII PCD shape: '+str(path))
        points = np.empty(n,dtype=dtype)
        offset=0
        for f,c in zip(fields,counts):
            points[f] = flat[:,offset] if c==1 else flat[:,offset:offset+c]
            offset+=c
    elif mode == 'binary_compressed':
        import lzf
        if len(payload)<8:
            raise ValueError('Invalid compressed PCD')
        compressed, raw_size = np.frombuffer(payload[:8],dtype='<u4')
        raw = lzf.decompress(payload[8:8+int(compressed)],int(raw_size))
        if raw is None or len(raw) != n*dtype.itemsize:
            raise ValueError('Invalid decompressed PCD size')
        points=np.empty(n,dtype=dtype)
        offset=0
        for f,s,c in zip(fields,sizes,counts):
            length=n*s*c
            base=dtype[f].base
            values=np.frombuffer(raw[offset:offset+length],dtype=base).reshape(n,c)
            points[f]=values[:,0] if c==1 else values
            offset+=length
    else:
        raise ValueError('Unsupported PCD DATA: '+mode)
    if not all(f in fields for f in ('x','y','z')) or not n:
        raise ValueError('Empty PCD or missing XYZ: '+str(path))
    if 'rgb' in fields or 'rgba' in fields:
        # Same fourth channel as original Open3D colors[:,0], in [0,1].
        color=np.asarray(points['rgb' if 'rgb' in fields else 'rgba']).copy()
        packed=color.view('<u4') if color.dtype.kind=='f' else color.astype('<u4')
        intensity=((packed>>16)&255).astype(np.float32)/255.0
    elif 'intensity' in fields:
        intensity=points['intensity']
    else:
        raise ValueError('PCD has neither RGB nor intensity; refusing synthetic intensity: '+str(path))
    output=np.column_stack([points['x'],points['y'],points['z'],intensity]).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite PCD values: '+str(path))
    return output
