from pathlib import Path
from funkshuns import *
import pandas as pd, numpy as np
from random import choice
from dreemer import *
pth = Path('/Users/os/Docs/Art/Film/Dreamy')
pth
batch = "Rock_Star_Energized"
fx = False
batchpth = pth/'batches'
batchpth.mkdir(exist_ok=True)
batchxl = batchpth/(batch+'.xlsx')
batchxl
if fx:
    sorapth =Path(r'E:\His\VJ Content\FX') 
    vidpth = Path(r'E:\His\VJ Content\FX\TD')/batch
else:
    sorapth =Path('/Users/os/Docs/D/VJ/Sora')
    vidpth = Path('/Users/os/Docs/D/VJ/SoraRaw')/batch
# %run -n C:\Users\seanrm100\Desktop\Docs\TouchDesigner\TouchDiffuze\RIFLER\RIFLER.py

# vidpth = Path(r'E:\His\VJ Content\SoraBig')
# vid = vidpth/'octo (2).mp4'

pth = pth.parent/'prompta'
stylpth = pth/'styles'
p2pth  = pth/'pmtlists2'
p1pth  = pth/'pmtlists1'

# pmtxt = p2pth/'oldmaster#om,heady,art,me.txt'

s = iterPmts(stylpth)
s
p2 = pd.concat([
    iterPmts(p2pth)
    ,iterPmts(p1pth)
])
p2

styl = s.loc['oldmaster'].pmt
# prompt = pmt + ' ' + styl 
# prompt
outfps = 10
z = 2
use_denoising_batch = True
lora_dict = None
acceleration = "xformers"
enable_similar_image_filter = True
vidpth
styls = s.pmt.to_list()
vidz = list(vidpth.glob('*.mp4')) + list(vidpth.glob('*.mov'))
vidz
vids = pd.DataFrame(vidz, columns = ['pth'])
vids['nm'] = vids['pth'].apply(lambda x: x.stem[:-1])
vids
nms = vids['nm'].unique()
nms
s.loc['scifi']['pmt']
p2.index.to_list()
pmtmap = dict(zip(nms, [choice(p2.index.to_list()) for _ in range(len(nms))]))
pmtmap
# pmtmap = {'Fax_Machine_Horror_Show': 'carnivorous2'}
dopestyls = s.loc[['dslr','oldmaster','rococo','synth','omlite','microchip','scifi','threyda','cubism']].pmt.to_list()
dopestyls
stylmap = dict(zip(nms, [choice( [choice(styls) , choice(dopestyls)])
                          for _ in range(len(nms))]))
stylmap
# stylmap = {'Fax_Machine_Horror_Show': styls[2]}
vids['pmtnm'] = vids.nm.map(pmtmap)
vids
vids['styl'] = vids.nm.map(stylmap)
vids
vids['pmt'] = vids.apply(lambda x: choice(p2.loc[x['pmtnm']].pmt),axis=1)#.reset_index(drop = True)
vids
loramap = dict(zip(nms, [pickLora(lora)
                for _ in range(len(nms))]))
loramap
vids['lora'] = vids.nm.map(loramap)
vids['x'] = 3
vids
# ---
# if not batchxl.exists():
# vids.to_excel(batchxl)
# ! "{batchxl}"
 # edit xl
vids = pd.read_excel(batchxl,index_col=0)
vids['pth'] = vids['pth'].map(Path)

def randoPrompt(pmtnm):
    if pmtnm in p2.index:
        pmt = choice(p2.loc[pmtnm].pmt)
    else:
        pmt = pmtnm
    return pmt
vids['pmt'] = vids.pmtnm.map(randoPrompt)
vids
vids['prompt'] = vids.pmt.fillna('') + ' ' + vids.styl.fillna('')
vids
outvidpth = Path.cwd()/'soratemp'
outvidpth.mkdir(exist_ok = True)

# vids = pd.read_excel(batchxl,index_col=0)
# vids['pth'] = vids['pth'].map(Path)
# vids['prompt'] = vids.pmt.fillna('') + ' ' + vids.styl.fillna('')


# vids = vids.iloc[4:]
# vids
outvidpth = Path.cwd()/'soratemp'
outvidpth.mkdir(exist_ok = True)
vids
# outvidpth2 = sorapth/f'{vidpth.name}2'
# outvidpth2.mkdir(exist_ok = True)
# outvidpth2
stream = None
209/23
# 23 5sec in 209 min
# 9 min per 5 sec at x1
vids.iloc[5:]
# <2 min per 5s vid w cpu offload and sd1.5

import shutil

# def blastoff(vids):
#     '''
#     vids: DataFrame with columns ['pth','nm','prompt','lora','x']
#     '''
# t_index_list=[45,45,45,45] if fx else [3,3,2]
t_index_list = [4,4,4,3]
for nm,df in vids.iloc[:1].groupby('nm'):
    print(nm)
    vidz = df.pth.to_list()

    for vid,prompt in zip(vidz,df.prompt):
        prompt = prompt[:20]  # limit prompt length
        prompt = [prompt]*3
        outvid = outvidpth/vid.name
        bch = df.iloc[0]
        x = bch.x
        # loras = parseLora(bch.lora)
        print('no loras!')
        loras = None

        if outvid.exists():
            print(f'{outvid} exists')
            continue
        # try:
        
        if fx:
            frames,aud_, fps = read_video(vid)
            frames = frames[::int(x)]
            if len(frames)>400:
                frames = frames[:400]
        else:
            frames,fps = remSora(vid)
            frames,fps = loop(frames,fps)
            if len(frames) > 160:
                x= max(x,2)
                print(f'{len(frames)} frames, setting x={x}')
            frames = frames[::int(x)]
            frames = saturate(frames,1.5)
            frames = randomhue(frames)
            frames = sharpen(frames,3.0)
        # frames = frames[::int(fps/outfps)]
        frames = frames/255
        # 4tiles
        deep,stream= dream(
                        frames,
                        # t_index_list=[6,7,13,13,13,13] if fx else [3,3,2],
                        t_index_list=t_index_list,
                        z=2,
                        guidance_scale=1.2,
                        stream=stream,
                        prompt=prompt,
                        loras= loras,
                        )
        
        # mid rectangle
        if not fx:
            deep,stream= dream(
                            deep,
                            bbox='mid',
                            t_index_list=t_index_list,
                            overlap=35,
                            guidance_scale=1.2,
                            stream=stream,
                            prompt=prompt,
                            loras= loras,
                            )
            
        # del stream
        # deep = deep[2:]
        deep = deep[len(t_index_list)-1:]


        if x > 1:
            deep = upsamp(deep,x,
                        #   shrp=1.35
                            shrp=3
                            )
            # interpolated_indices = np.arange(deep.shape[0]) % 3 != 0
            # interpolated_frames = deep[interpolated_indices]
            # sharper = sharpen(interpolated_frames,2)
            # deep[interpolated_indices] = sharper
            # del sharper,interpolated_frames,interpolated_indices

        print('writing...')
        writeVid(deep,outvid)

        # (vidpth/'done').mkdir(exist_ok=True)
        # shutil.move(vid,vidpth/'done'/vid.name)

        print(outvid)
        # except Exception as e:
        #     print('ERR',e) 
        #     continue
# print(df.pmt.iloc[0])