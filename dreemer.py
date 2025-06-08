# import tqdm.auto
# import types

# # Save original
# _real_tqdm = tqdm.auto.tqdm

# # Patch: disables bars only if called from diffusers
# def patched_tqdm(*args, **kwargs):
#     from inspect import stack
#     for frame in stack():
#         if "diffusers" in frame.filename:
#             return iter(args[0]) if args else iter([])
#     return _real_tqdm(*args, **kwargs)

# tqdm.auto.tqdm = patched_tqdm


try:
    from .v2vwrapper import StreamV2VWrapper
except ImportError:
    from v2vwrapper import StreamV2VWrapper

import torch
import numpy as np, pandas as pd
import concurrent.futures
from typing import Generator, Optional, Tuple, List
# from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers import AutoencoderKL, StableDiffusion3Pipeline, StableDiffusionPipeline
from diffusers.utils import load_image

from streamv2v.image_utils import postprocess_image
from pathlib import Path

import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm
from patchify import patchify, unpatchify
from PIL import Image
from torchvision.transforms.functional import adjust_sharpness, adjust_hue, adjust_contrast,adjust_gamma

from functools import reduce
from operator import iconcat

from dotenv import load_dotenv
load_dotenv()
import os
hftoken = os.environ.get('HUGGING_FACE_ACCESS_TOKEN')


def getDevice():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
# print(f"Using device: {device}")

def flattenList(listOfLists):
    '''or list of tups'''
    return reduce(iconcat,listOfLists,[])

class xnp:
    def chunk(a, chunksize=None,
            n=None):
        if chunksize is None and n is None:
            raise ValueError('Either chunksize or n must be specified')
        if chunksize:
            n = int(np.ceil(len(a)/chunksize))
        chunks = np.array_split(a, n)
        return chunks

    def unchunk(chunks):
        return np.concatenate(chunks, axis=0)

    # def test_chunk():
    #     a = np.concatenate(
    #         [np.array([[i,i]]) for i in range(10)], axis=0
    #     )
    #     for sz in (2,3,4):
    #         assert np.array_equal(unchunk(chunk(a,sz)), a)


modelpth = Path('/Users/os/Docs/D/StableDiffuze')
sd = '3.5'
if sd == '1.5':
    model_id = str(modelpth/'models'/'v1-5-pruned-emaonly.safetensors')
    pipeline_cls = StableDiffusionPipeline
    pipe_kwargs = {}

    lcm_lora_id = None

    use_tiny_vae = True
    vae_id = None
    vae_kwargs = {}
elif sd == '3.5':
    
    # model_id = str(modelpth/'models'/'sd3.5m_turbo.safetensors')
    model_id = "tensorart/stable-diffusion-3.5-medium-turbo"
    pipeline_cls = StableDiffusion3Pipeline
    # pipe_kwargs = {'token': hftoken}
    pipe_kwargs = dict(variant="fp16",subfolder='transformer')

    lcm_lora_id = str(modelpth/'models'/'lora_sd3.5m_4steps.safetensors')

    use_tiny_vae = False
    vae_id = "tensorart/stable-diffusion-3.5-medium-turbo"
    vae_kwargs = dict(subfolder="vae")
    
# seed = np.random.randint(0,10000)
use_denoising_batch = True
# model_id = "stabilityai/sd-turbo"
# model_id = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
# if isinstance(model_id,Path):
#     assert model_id.exists(), f'Model not found: {model_id}'
# model_id_or_path: str = os.environ.get("MODEL", "KBlueLeaf/kohaku-v2.1")
# LoRA dictionary write like    field(default_factory=lambda: {'E:/stable-diffusion-webui/models/Lora_1.safetensors' : 1.0 , 'E:/stable-diffusion-webui/models/Lora_2.safetensors' : 0.2})
# lora_dict: dict = None
# LCM-LORA model
# lcm_lora_id: str = os.environ.get("LORA", "latent-consistency/lcm-lora-sdv1-5")
# TinyVAE model
# vae_id: str = os.environ.get("VAE", "madebyollin/taesd")
lora_dict = None
acceleration = "none" if torch.cuda.is_available() else "none"#"mps"  # "none", "mps", "cuda"
use_denoising_batch = True
enable_similar_image_filter = True


# lorapth = Path(r'E:\His\StableDiffuze\LoRA')
lorapth = modelpth/'LoRA'
assert lorapth.exists(), f'LoRA path not found: {lorapth}'
lora = pd.read_excel(lorapth/'loras.xlsx',index_col=0)
lora['pth'] = lora.stem.map(lambda x: str(lorapth/(x+'.safetensors')))
lora = lora[lora.rating!=-1]
assert lora.pth.map(lambda x: Path(x).exists()).all(), f'LoRA files not found: {lora[~lora.pth.map(lambda x: Path(x).exists())]}'

loramap = {
    'threyda': {
        'trippyArt':1.2,
        'wow':0.8,
        'detail':3.0,
    },
    'swirls': 'dreamcolor,wow,fashion'

}

nightlightsidx=[38,44,46,47,48,49,49]
# worlds,cosmos,wow
# vietfuture
# wow,topo:1.5,dark
# psych:0.2,wow,topo:1.5
# trippyArt:0.5,wow,detail:2,bismuth


def parseLora(lor):
    '''parse lora string into dict 
    lor (str): lora string, e.g. 'wow:0.8,detail:1' or 'threyda:2' or 'wow'
    '''
    if pd.isna(lor):
        return {}
    if isinstance(lor,dict):
        return lor
    if lor.split(':')[0] in loramap:
        m = 1
        if ':' in lor:
            lor,m = lor.split(':')
        loras = loramap[lor]
        loras = {k:v*float(m) for k,v in loras.items()}
    else:
        lors = [l if (':' in l) else f'{l}:1' for l in lor.split(',')]
        splits = [l.split(':') for l in lors]
        loras = dict(splits)
    loras = {k:float(v) for k,v in loras.items()}
    return loras

import torch
from torchvision.io import read_image  # reads image as [C, H, W]

def read_frames(file_list):
    if not isinstance(file_list, list):
        file_list = list(file_list.glob('*.png'))
    frames = []
    for file in file_list:
        # Convert pathlib.Path to str
        img = read_image(str(file))
        frames.append(img)
    video = torch.stack(frames, dim=0).permute(0,2,3,1)
    return video
def iterPmts(pth):
    pmts = pd.DataFrame({'txt':list(pth.glob('*.txt'))})
    pmts['nm'] = pmts.txt.apply(lambda x: x.stem.split('#')[0])
    pmts['pmt'] = pmts.txt.apply(lambda x: x.read_text())
    if not pth.name.startswith('styl'):
        pmts['pmt'] = pmts.pmt.apply(lambda x: x.strip(' ').strip('\n').split('\n'))
    pmts = pmts[~pmts.nm.str.startswith('_')]
    return pmts.set_index('nm')

import numpy as np
from patchify import patchify  # make sure you have the patchify library installed

import numpy as np
from patchify import patchify  # Ensure you have patchify installed

def patch(imgOrVid, z, overlap=20):
    """
    Split an image or video into patches.

    Parameters:
      imgOrVid : array-like
          Input image (H,W,C) or video (F,H,W,C)
      z : int or (int, int)
          If int, both vertical and horizontal patch counts will be z.
          If tuple, interpreted as (zu, zv) for vertical and horizontal splits.
      overlap : int
          Number of pixels to overlap (and later blend) between adjacent patches.

    Returns:
      patches : np.ndarray
          Array of patches.
      metadata : dict
          Contains:
             - 'padded_shape': shape after padding,
             - 'original_shape': shape of input,
             - 'tiles_shape': raw shape from patchify,
             - 'step': step sizes used,
             - 'patch_size': (patch_h, patch_w),
             - 'grid': (zu, zv) as provided.
    """
    # Interpret z as (zu, zv): zu = vertical (rows), zv = horizontal (columns)
    if isinstance(z, int):
        zu, zv = z, z
    else:
        zu, zv = z

    a = np.array(imgOrVid)
    ogshp = a.shape  # original shape before padding
    # Determine if input is video (has frame dimension)
    f = ogshp[0] if len(ogshp) > 3 else None

    # Pad spatial dimensions (but not frames if video)
    if overlap:
        pad_shp = [(0, overlap), (0, overlap), (0, 0)]
        if f is not None:
            pad_shp = [(0, 0)] + pad_shp
        a = np.pad(a, pad_shp, mode='reflect')
    padded_shape = a.shape

    # Get spatial dimensions from the original input (before padding)
    h, w, c = ogshp[-3:]
    
    # Compute patch size and step such that final stitched dimensions equal the original.
    # (h - overlap) is divided evenly among the vertical patches, then add back the overlap.
    patch_h = ((h - overlap) // zu) + overlap
    patch_w = ((w - overlap) // zv) + overlap
    step_h = patch_h - overlap  # This equals (h - overlap) // zu
    step_w = patch_w - overlap  # This equals (w - overlap) // zv

    # Set winshape and step for patchify
    if f is not None:
        winshape = [ogshp[0], patch_h, patch_w, c]
        step = (ogshp[0], step_h, step_w, c)
    else:
        winshape = [patch_h, patch_w, c]
        step = (step_h, step_w, c)

    # Create patches with patchify
    tiles = patchify(a, winshape, step=step)
    tileshp = tiles.shape

    # Set grid explicitly from (zu, zv)
    grid = (zu, zv)

    # Reshape to a list of patches
    patches = tiles.reshape(-1, *winshape)
    
    # Save metadata for unpatching
    metadata = {
        'padded_shape': padded_shape,
        'original_shape': ogshp,
        'tiles_shape': tileshp,
        'step': step,
        'patch_size': (patch_h, patch_w),
        'grid': grid
    }
    
    return patches, metadata

def create_symmetric_2dmask(h, w, overlap):
    """
    Creates a 2D mask of shape (h, w) that is 1.0 in the interior
    and fades linearly to 0.0 over 'overlap' pixels at each edge.
    """
    mask = np.ones((h, w), dtype=np.float32)
    if overlap <= 0:
        return mask
    # Fade top edge
    for row in range(overlap):
        alpha = row / overlap
        mask[row, :] *= alpha
    # Fade bottom edge
    for row in range(h - overlap, h):
        alpha = (h - row) / overlap
        mask[row, :] *= alpha
    # Fade left edge
    for col in range(overlap):
        alpha = col / overlap
        mask[:, col] *= alpha
    # Fade right edge
    for col in range(w - overlap, w):
        alpha = (w - col) / overlap
        mask[:, col] *= alpha
    return mask

def unpatch(patches, metadata, overlap=20, hres=None, wres=None):
    """
    Reconstruct a full image or video from a grid of patches.

    Uses the grid stored in metadata["grid"]. If hres/wres are not provided,
    they are taken from metadata["patch_size"].

    Parameters:
      patches : np.ndarray
          For a video: shape (num_tiles, f, patch_h, patch_w, C);
          For an image: shape (num_tiles, patch_h, patch_w, C).
      metadata : dict
          Must include:
             - "original_shape": original shape of the image/video.
             - "patch_size": a tuple (patch_h, patch_w) computed in patch().
             - "grid": the intended grid (zu, zv) as specified when patching.
      overlap : int
          The pixel overlap used during patching.
      hres : int, optional
          Height of each patch for reconstruction; if None, metadata["patch_size"][0] is used.
      wres : int, optional
          Width of each patch for reconstruction; if None, metadata["patch_size"][1] is used.

    Returns:
      np.ndarray : The fully stitched image (H, W, C) or video (f, H, W, C).

    The final stitched dimensions are computed as:
       final_H = zu * step_h + overlap
       final_W = zv * step_w + overlap
    """
    # Use patch size from metadata if not provided
    if hres is None or wres is None:
        hres, wres = metadata["patch_size"]
    
    # Use the explicit grid from metadata (this is (zu, zv))
    if "grid" not in metadata:
        raise ValueError("Metadata must contain a 'grid' key with the intended grid dimensions.")
    zu, zv = metadata["grid"]
    
    # For clarity, step_h and step_w can be computed as:
    step_h = hres - overlap
    step_w = wres - overlap

    # Debug prints so you can verify
    # print("DEBUG: Grid dimensions =", (zu, zv))
    # print("DEBUG: Patch size =", (hres, wres))
    
    ogshp = metadata["original_shape"]
    
    # Compute final stitched dimensions so that:
    # final_H = zu * step_h + overlap  and  final_W = zv * step_w + overlap
    final_H = zu * step_h + overlap
    final_W = zv * step_w + overlap

    # Check if we're working with video (4D original shape) or image (3D)
    if len(ogshp) == 4:
        f, H, W, C = ogshp
        out = np.zeros((f, final_H, final_W, C), dtype=np.float32)
        weight = np.zeros((f, final_H, final_W, C), dtype=np.float32)
        mask = create_symmetric_2dmask(hres, wres, overlap)[..., None]
        idx = 0
        for i in range(zu):
            for j in range(zv):
                top = i * step_h
                left = j * step_w
                # Each patch is assumed to be (f, hres, wres, C)
                out[:, top:top+hres, left:left+wres, :] += patches[idx] * mask
                weight[:, top:top+hres, left:left+wres, :] += mask
                idx += 1
        out /= np.maximum(weight, 1e-8)
        return out
    else:
        H, W, C = ogshp
        out = np.zeros((final_H, final_W, C), dtype=np.float32)
        weight = np.zeros((final_H, final_W, C), dtype=np.float32)
        mask = create_symmetric_2dmask(hres, wres, overlap)[..., None]
        idx = 0
        for i in range(zu):
            for j in range(zv):
                top = i * step_h
                left = j * step_w
                # Each patch is assumed to be (hres, wres, C)
                out[top:top+hres, left:left+wres, :] += patches[idx] * mask
                weight[top:top+hres, left:left+wres, :] += mask
                idx += 1
        out /= np.maximum(weight, 1e-8)
        return out



def cut_bbox_video(
    video: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    overlap: int = 20
) -> tuple[np.ndarray, dict]:
    """
    Extract a bounding box region from a video with extra pixels for blending.
    Also compute the location of the requested region within the patch.
    
    Parameters
    ----------
    video : np.ndarray
        Video with shape (F, H, W, C).
    x, y : int
        Top-left corner of the requested region (in original video coords).
    w, h : int
        Width and height of the requested region.
    overlap : int
        Requested number of extra pixels for feathering.
    
    Returns
    -------
    patch : np.ndarray
        Extracted patch with shape (F, patchH, patchW, C).
    metadata : dict
        Contains info to paste the patch back, including the patchs bounding box and
        the inner region coordinates.
    """
    video = np.array(video)
    F, H, W, C = video.shape

    # Compute bounding box with extra 'overlap' on each side, clamped to video boundaries.
    x0 = max(0, x - overlap)
    y0 = max(0, y - overlap)
    x1 = min(W, x + w + overlap)
    y1 = min(H, y + h + overlap)

    # print(y0,y1,x0,x1)
    patch = video[:, y0:y1, x0:x1, :].copy()

    # Compute the coordinates of the requested region within the patch.
    region_x0 = x - x0
    region_y0 = y - y0
    region_x1 = region_x0 + w
    region_y1 = region_y0 + h

    metadata = {
        "orig_shape": video.shape,
        "bbox": (x0, y0, x1, y1),
        "requested_xywh": (x, y, w, h),
        "overlap": overlap,
        "region_coords": (region_x0, region_y0, region_x1, region_y1)
    }
    return patch, metadata


def create_feather_mask_1d(length: int, region_start: int, region_end: int, requested_overlap: int) -> np.ndarray:
    """
    Create a 1D blending mask for one dimension (horizontal or vertical).
    
    The mask is defined so that:
      - It is 0 (use original video) outside the blending area.
      - It is 1 (use processed patch) inside the requested region.
      - It ramps linearly from 0 to 1 over a distance of `requested_overlap` pixels
        (or fewer if clamped by the patch boundaries).
    
    Parameters
    ----------
    length : int
        Total length of the dimension (patch width or height).
    region_start : int
        Start coordinate of the requested region within the patch.
    region_end : int
        End coordinate of the requested region within the patch.
    requested_overlap : int
        The desired ramp length.
    
    Returns
    -------
    mask : np.ndarray
        1D mask with shape (length,).
    """
    mask = np.zeros(length, dtype=np.float32)
    
    # Left ramp: available pixels before the inner region.
    left_avail = region_start
    ramp_left = min(requested_overlap, left_avail)
    if ramp_left > 0:
        # Ramp from 0 to 1 over the last 'ramp_left' pixels before region_start.
        mask[region_start - ramp_left:region_start] = np.linspace(0, 1, ramp_left, endpoint=False)
    
    # Inner region: full contribution (alpha = 1).
    mask[region_start:region_end] = 1.0
    
    # Right ramp: available pixels after the inner region.
    right_avail = length - region_end
    ramp_right = min(requested_overlap, right_avail)
    if ramp_right > 0:
        mask[region_end:region_end + ramp_right] = np.linspace(1, 0, ramp_right, endpoint=False)
    
    return mask


def paste_bbox_video(
    original_video: np.ndarray,
    patch: np.ndarray,
    metadata: dict
) -> np.ndarray:
    """
    Paste a processed patch back into the original video using alpha blending.
    
    The blending mask is computed relative to the requested (inner) region.
    Each edge is feathered independently using a ramp over `requested_overlap` pixels,
    or as many as are available if the patch is clipped by the video boundary.
    
    Parameters
    ----------
    original_video : np.ndarray
        Original video with shape (F, H, W, C).
    patch : np.ndarray
        Processed patch with shape (F, patchH, patchW, C).
    metadata : dict
        Contains 'bbox' (patch bounds), 'overlap', and 'region_coords' (inner region coords).
    
    Returns
    -------
    out : np.ndarray
        Blended video (float32 copy of original_video).
    """
    original_video = np.array(original_video)
    patch = np.array(patch)
    F, H, W, C = metadata["orig_shape"]
    (x0, y0, x1, y1) = metadata["bbox"]
    requested_overlap = metadata["overlap"]
    region_coords = metadata["region_coords"]
    region_x0, region_y0, region_x1, region_y1 = region_coords

    _, patchH, patchW, _ = patch.shape

    # Build 1D masks for horizontal and vertical directions.
    mask_x = create_feather_mask_1d(patchW, region_x0, region_x1, requested_overlap)
    mask_y = create_feather_mask_1d(patchH, region_y0, region_y1, requested_overlap)
    
    # Combine via outer product to obtain a 2D mask.
    mask_2d = np.outer(mask_y, mask_x)
    mask_3d = mask_2d[None, ..., None]  # Expand to shape (1, patchH, patchW, 1)

    # Create a float copy of the original video.
    out = original_video.astype(np.float32, copy=True)
    region = out[:, y0:y1, x0:x1, :]

    # Alpha blending: where mask==1, use the patch; where mask==0, retain original video.
    if requested_overlap:
        region_blended = region * (1.0 - mask_3d) + patch * mask_3d
    else:
        region_blended = patch
    out[:, y0:y1, x0:x1, :] = region_blended

    return out




import torch
from torchvision.io import read_video, write_video
def create_looped_video(
    input_path: str,
    output_path: str,
    blend_duration: float = 1.0,
    pts_unit: str = 'sec',
):
    """
    Reads a video, crossfades the last 'blend_duration' seconds 
    so that the final frame blends into the first frame (or first second),
    and writes the looped video to output_path.
    
    :param input_path: Path to the input video file.
    :param output_path: Path where the looped video will be saved.
    :param blend_duration: Duration (in seconds) to blend the last frames with the first frames.
    :param pts_unit: 'sec' or 'pts', see torchvision.io.read_video docs.
    """
    # Read the entire video + audio
    frames, audio, info = read_video(input_path, pts_unit=pts_unit)
    
    # Frames is a 4D tensor in shape: (num_frames, height, width, channels)
    num_frames = frames.shape[0]
    fps = info["video_fps"]
    
    if num_frames == 0:
        print("No frames found in video.")
        return
    
    # Number of frames to blend over:
    #   We clamp it so it doesn't exceed the number of frames
    blend_frames = min(int(round(blend_duration * fps)), num_frames - 1)
    
    # Edge case: if blend_frames < 1, just write out the original video
    if blend_frames < 1:
        write_video(output_path, frames, fps, audio_array=audio, audio_fps=info["audio_fps"])
        print("Blend duration too short for crossfade; wrote original video.")
        return
    
    # 1) Identify the last blend_frames of the video
    # 2) Identify the first blend_frames of the video
    last_segment = frames[-blend_frames:]   # shape: (blend_frames, H, W, C)
    first_segment = frames[:blend_frames]   # shape: (blend_frames, H, W, C)
    
    # We will crossfade from the last segment to the first segment
    #   alpha goes from 0 -> 1 across the blend_frames
    #   out_frame[i] = (1 - alpha) * last_segment[i] + alpha * first_segment[i]
    
    for i in range(blend_frames):
        alpha = (i + 1) / blend_frames
        # Blend pixel-wise
        frames[num_frames - blend_frames + i] = (
            (1.0 - alpha) * last_segment[i].float() + alpha * first_segment[i].float()
        ).to(frames.dtype)
    
    # Now the last frame of the video is effectively the "beginning" frame
    # in a blended manner.
    
    # Write out the new frames and the same audio track
    write_video(
        output_path,
        frames,
        fps,
        audio_array=audio,       # If you want to include original audio
        audio_fps=info["audio_fps"] if "audio_fps" in info else 48000,
        video_codec='h264',      # Adjust codec as needed
        options={"crf": "18"}    # Adjust as needed
    )


def loop(
    frames,fps,
    # output_path: str,
    input_path=None,
    blend_duration: float = 1.0,
    pts_unit: str = 'sec',
):
    """
    Reads a video, crossfades the last 'blend_duration' seconds 
    so that the final frame blends into the first frame (or first second),
    and writes the looped video to output_path.
    
    :param input_path: Path to the input video file.
    :param output_path: Path where the looped video will be saved.
    :param blend_duration: Duration (in seconds) to blend the last frames with the first frames.
    :param pts_unit: 'sec' or 'pts', see torchvision.io.read_video docs.
    """
    # Read the entire video + audio
    if frames is None:
        assert input_path
        frames, audio, info = read_video(input_path, pts_unit=pts_unit)
        fps = info["video_fps"]
    
    # Frames is a 4D tensor in shape: (num_frames, height, width, channels)
    num_frames = frames.shape[0]
    
    if num_frames == 0:
        print("No frames found in video.")
        return
    
    # Number of frames to blend over:
    #   We clamp it so it doesn't exceed the number of frames
    blend_frames = min(int(round(blend_duration * fps)), num_frames - 1)
    
    # Edge case: if blend_frames < 1, just write out the original video
    # if blend_frames < 1:
    #     write_video(output_path, frames, fps, audio_array=audio, audio_fps=info["audio_fps"])
    #     print("Blend duration too short for crossfade; wrote original video.")
    #     return
    
    # 1) Identify the last blend_frames of the video
    # 2) Identify the first blend_frames of the video
    last_segment = frames[-blend_frames:]   # shape: (blend_frames, H, W, C)
    first_segment = frames[:blend_frames]   # shape: (blend_frames, H, W, C)
    
    # We will crossfade from the last segment to the first segment
    #   alpha goes from 0 -> 1 across the blend_frames
    #   out_frame[i] = (1 - alpha) * last_segment[i] + alpha * first_segment[i]
    
    for i in range(blend_frames):
        alpha = (i + 1) / blend_frames
        # Blend pixel-wise
        frames[num_frames - blend_frames + i] = (
            (1.0 - alpha) * last_segment[i].float() + alpha * first_segment[i].float()
        ).to(frames.dtype)
    
    # Now the last frame of the video is effectively the "beginning" frame
    # in a blended manner.
    
    # Write out the new frames and the same audio track
    return frames, fps
    write_video(
        output_path,
        frames,
        fps,
        audio_array=audio,       # If you want to include original audio
        audio_fps=info["audio_fps"] if "audio_fps" in info else 48000,
        video_codec='h264',      # Adjust codec as needed
        options={"crf": "18"}    # Adjust as needed
    )

import contextlib
import subprocess

@contextlib.contextmanager
def workingDir(path):
    """Changes working directory and returns to previous on exit."""
    #this violates the abstraction architecture a bit but whatever
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
def cmder(cmd,*ARRGHHHSSSS):
    '''pass to cmd line and print output from py\n
    cmd: str 
    returns time elapsed, TODO allow returns from cmd\n
    TODO kwargs passed to cmd
    https://stackoverflow.com/a/6414278/13030053'''
    thenn = datetime.now()
    p = subprocess.Popen([cmd,*ARRGHHHSSSS], stdout=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        print(line.split(b'>')[-1]),
    p.stdout.close()
    p.wait()

    elaps = datetime.now() - thenn
    return elaps

from pathlib import Path
# def upsamp(vidpth,x=3):
#     vid = vidpth
#     assert vid.exists(), vid
#     cmder('python','inference_video.py','--video',str(vid),f'--multi={int(x)}')

from datetime import datetime
bpth = Path.cwd()/'benchmarks'

# def sampleFrame(
#         video,
#         t_index_list: List[int],
#         prompt: str,
#         i=0,
#         # bpth: Path,
#         # num_inference_steps=2,
#     ):
#     bpth.mkdir(exist_ok=True)

#     height,width = int(720/2),int(1280/2)
#     # height,width = hres,wres

#     stream = StreamV2VWrapper(
#         model_id_or_path=model_id,
#         lora_dict=lora_dict,
#         # t_index_list=[35,40],
#         # t_index_list=t_index_list,
#         t_index_list=t_index_list,
#         frame_buffer_size=1,
#         width=width,
#         height=height,
#         warmup=1,
#         acceleration=acceleration,
#         do_add_noise=True,
#         mode="img2img",
#         output_type="pt",
#         enable_similar_image_filter=False,
#         similar_image_filter_threshold=0.98,
#         use_denoising_batch=use_denoising_batch,
#         seed=np.random.randint(0,10000),
#     )
#     num_inference_steps=max(t_index_list)+1
#     stream.prepare(
#         prompt=prompt,
#         num_inference_steps=num_inference_steps,
#     )

#     f = 3
#     tiles, metadata= patch(video, 2, overlap=20)
#     tile = torch.tensor(tiles[i]).to(video.dtype)
#     for _ in range(stream.batch_size):
#         stream(tile[f].permute(2, 0, 1))
    
#     now = pd.Timestamp(datetime.now())

#     output_image = stream(tile[f].permute(2, 0, 1)).permute(1, 2, 0)*255

#     elaps = pd.Timestamp(datetime.now()) - now
#     samp = Image.fromarray((output_image).numpy().astype(np.uint8))
#     # samp.show() 
#     samp
#     stp = '-'.join([str(x) for x in t_index_list])
#     samp.save(bpth/f'samp{elaps.microseconds}_{stp}.png')
#     return samp
# [sampleFrame(video,t) for t in [
#     [1,1],
#     [1,1,1],
#     [1],
# ]]
from torchvision.transforms import Resize
import gc
 
def dream(invideo,
            z=2,
            stream=None,
            prompt=None,
            bbox=None, # [x,y,w,h] or 'mid'
            nshift=None,
            loras=None, # loramap['threyda']
            loralite=True, # only use first trigger word
            wfull=1280, #out resolution
            hfull=720,
            overrideRes=None, # of stream
            overlap=20,
            init=True,
            initframes=None,
            guidance_scale=1.2,
            model_id=model_id,
            lora_dict=lora_dict,
            # t_index_list=[35,40],
            # t_index_list=t_index_list,
            t_index_list=[3,3,2],
            frame_buffer_size=1,
            warmup=1,
            acceleration=acceleration,
            do_add_noise=True,
            mode="img2img",
            output_type="pt",
            enable_similar_image_filter=False,
            similar_image_filter_threshold=0.98,
            use_denoising_batch=use_denoising_batch,
            seed='random',
            ret='video',
            dtype=torch.float16,
            patch=patch,unpatch=unpatch):
    Zd = False
    if z is None:
        Zd = False
    elif isinstance(z,int):
        if z>1:
            Zd = True
    else:
        # if z[0] > 1 or z[1] > 1:
        Zd = True


    video = invideo/255 if invideo.max()>2 else invideo
    if bbox is not None:
        if bbox=='mid':
            height,width = video.shape[-3:-1]
            x = width//4
            y = height//6
            # h,w = height//2,width//2
            w,h = stream.width-2-overlap*2 ,stream.height-2-overlap*2
            bbox = [x,y,w,h]

        tile,metadata = cut_bbox_video(video, *bbox, overlap=overlap)
        tiles = [torch.tensor(tile).to(dtype)]
        initz = [initframes]
    elif Zd:
        tiles,metadata = patch(video, z, overlap=overlap)
        if not initframes:
            initz = [None]*len(tiles)
    else:
        tiles = [video]
        initz = [initframes]
        metadata = None

    assert prompt is not None
    num_inference_steps=max(t_index_list)+1
    print(f'num_inference_steps: {num_inference_steps}')
    # print('prompt',prompt)

    
    if loras:
        
        triglist = lora.loc[list(loras.keys())].trigs.dropna().str.split(',')
        if loralite:
            triglist = triglist.str[0].map(lambda x: [x])

        triglist = triglist.to_list()
        trigs = list(dict.fromkeys(flattenList(triglist)))
        trig = ' '.join([f'{t}' for t in trigs])
        if isinstance(prompt, list):
            prompt = [f'{trig} {p}' for p in prompt]
        else:
            prompt = f'{trig} {prompt}'
        


    # else:
    #     try:
    #         stream.stream.load_lcm_lora(
    #             pretrained_model_name_or_path_or_dict="latent-consistency/lcm-lora-sdv1-5",
    #             adapter_name="lcm"
    #             )
    #     except Exception as e:
    #         print('lcm',e)
    #     stream.stream.pipe.set_adapters(['lcm'],adapter_weights=[1.0])
    if seed=='random':
        seed = np.random.randint(0,10000)

    if not stream:
        if not bbox:
            if isinstance(z,int):
                zu,zv = z,z
            else:
                zv,zu = z
            w,h = wfull//zu, hfull//zv
            hres = int(h + overlap//2)
            wres = int(w + overlap//2)
        else:
            if overrideRes:
                wres,hres = overrideRes
            else:
                hres,wres = tiles[0].shape[-3:-1]
        print(f'res: {wres}x{hres}')
        # h,w,c = video.shape[-3:]
        # height,width = h//z+overlap, w//z+overlap,
        # imgshp, ogshp , tileshp = metadata
        # height,width = tileshp[-3:-1]
        # add buffer for overlap
        # scale = np.array([h,w])/(np.array(ogshp[-3:-1])/z)
        # height,width = np.array(tiles.shape[-3:-1])*scale
        # height,width
        # print(lora_dict)
        streamcfg = dict(
            device=getDevice(),
            model_id_or_path=model_id,
            pipeline_cls=pipeline_cls,
            pipe_kwargs=pipe_kwargs,
            
            lcm_lora_id=lcm_lora_id,
            
            use_tiny_vae=use_tiny_vae,
            vae_id=vae_id,
            vae_kwargs=vae_kwargs,

            lora_dict=lora_dict,
            # t_index_list=[35,40],
            # t_index_list=t_index_list,
            t_index_list=t_index_list,
            frame_buffer_size=frame_buffer_size,
            warmup=warmup,
            acceleration=acceleration,
            do_add_noise=do_add_noise,
            mode=mode,
            output_type=output_type,
            enable_similar_image_filter=enable_similar_image_filter,
            similar_image_filter_threshold=similar_image_filter_threshold,
            use_denoising_batch=use_denoising_batch,
            seed=np.random.randint(0,10000)
        )
        # print('streamcfg',streamcfg) 
        print('+2')
        stream = StreamV2VWrapper(
            **streamcfg,
            width=wres,
            height=hres,#+6
        )
        stream.prepare(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            negative_prompt='nudity boobs breast nipple man male masculine gray-hair old naked bosom bad proportion grainy blurry anime'# modern american houses'
        )
    else:
        adaps = stream.stream.pipe.get_active_adapters()
        if 'lcm' in adaps:
            adaps.pop(adaps.index('lcm'))
        print(adaps)
        oldloras = [adap for adap in adaps if adap not in loras]
        if oldloras:
            # for old in oldloras:
            try:
                # print(old)
                stream.stream.pipe.unload_lora_weights()
                # stream.stream.pipe.set_lora_device([old],'cpu')
                # stream.stream.pipe.delete_adapters([old])
            except Exception as e:
                print(f'cant: {e}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.mps.is_available():
                torch.mps.empty_cache()

            print('dumpd')
            
            # Returns the number of
            # objects it has collected
            # and deallocated
            collected = gc.collect()
            print("Garbage collector: collected",
                  "%d objects." % collected)
        stream.inject(prompt)

    try:
        stream.stream.load_lcm_lora(
            lcm_lora_id,
            adapter_name="lcm"
            )
    except Exception as e:
        print('lcm',e)

    if loras:
        for lor in loras.keys():
            adaps = stream.stream.pipe.get_active_adapters()
            if lor not in adaps:
                print(f'adding {lor}')
                try:
                    stream.stream.load_lora(lora.loc[lor].pth, adapter_name=lor)
                except ValueError as e:
                    print(e)
        stream.stream.pipe.set_adapters(['lcm']+list(loras.keys()),
                        adapter_weights=[1.0]+list(loras.values()))
        print('loras loaded')
    else:
        
        stream.stream.pipe.set_adapters(['lcm'],adapter_weights=[1.0])

    outtiles = []
    for tile,initfs in zip(tiles,initz):
        tile = torch.tensor(tile).to(dtype)
        # Gening video of wrong size?? -2px on each dim
        # video_result = torch.zeros(tile.shape[0], stream.height, stream.width, 3)
        # print(tile.shape)
        # print(video_result.shape)

        if init:
            if initfs is not None:
                initfs = torch.tensor(initfs)
                if initfs.max()>1:
                    initfs = initfs/255
                initfs = initfs.to(dtype)
                for f in range(stream.batch_size):
                    stream(initfs[f].permute(2, 0, 1))

            else: # use first frame
                for _ in range(stream.batch_size):
                    stream(tile[0].permute(2, 0, 1))

        for i in tqdm(range(tile.shape[0])):
            output_image = stream(tile[i].permute(2, 0, 1))
            res = output_image.permute(1, 2, 0)
            if i==0:
                print(res.shape)
                video_result = torch.zeros(tile.shape[0], *res.shape)
                hh,ww = res.shape[:2]
            video_result[i] = res

        video_result = video_result * 255
        outtiles += [video_result.cpu().numpy().astype(np.uint8)]

    # unpatch:


    if bbox is None:
        if ret!='video':
            return outtiles,stream
        if Zd:
            outvideo = unpatch(np.array(outtiles),metadata,overlap=overlap,hres=hh,wres=ww)#
        else:
            outvideo = outtiles[0]
    else:
        outtile = outtiles[0]
        # return outtile,stream
        nshift = max(stream.batch_size - 1, 0)
        if nshift:
            print(f'shifting by {nshift}')
            lastframes = (tile[-nshift:]*255).cpu().numpy().astype(np.uint8) 
            franken = np.concatenate([outtile[nshift:],lastframes])
            assert franken.shape == outtile.shape, f'{franken.shape} != {outtile.shape}'
            outtile = franken
        
        if not overrideRes:
            t = torch.tensor(outtile)
            hres,wres = tiles[0].shape[-3:-1]
            resize = Resize(
                # (H, W)
                size=(hres,wres)
            )
            del tiles
            rs = resize(t.permute(0,3,1,2))
            outtile = rs.permute(0,2,3,1).numpy()

        if ret!='video':
            return outtile,stream

        outvideo = paste_bbox_video(video*255,np.array(outtile),metadata)
        # stream.height,
        # stream.width,
    # if init:
    # outvideo = outvideo[stream.batch_size-1:]
    return outvideo,stream

# def dreamBbox(
#         invideo,
#         bbox='mid',
#             stream=None,
#             prompt=None,
#             loras=None, # loramap['threyda']
#             overlap=50,
#             # wfull=1280, #out resolution
#             # hfull=720,
#             guidance_scale=1.2,
#             streamcfg=dict(
#                 model_id_or_path=model_id,
#                 lora_dict=lora_dict,
#                 # t_index_list=[35,40],
#                 # t_index_list=t_index_list,
#                 t_index_list=[3,3,2],
#                 frame_buffer_size=1,
#                 warmup=1,
#                 acceleration=acceleration,
#                 do_add_noise=True,
#                 mode="img2img",
#                 output_type="pt",
#                 enable_similar_image_filter=False,
#                 similar_image_filter_threshold=0.98,
#                 use_denoising_batch=use_denoising_batch,
#                 seed=np.random.randint(0,10000),
#             ),
#             dtype=torch.float32,
#             # patch=patch,unpatch=unpatch
#             ):
#     '''bbox [xywh]'''
#     video = invideo/255 if invideo.max()>2 else invideo

#     if bbox=='mid':
#         height,width = video.shape[-3:-1]
#         x = width//4
#         y = height//6
#         # h,w = height//2,width//2
#         w,h = stream.width-2-overlap*2 ,stream.height-2-overlap*2
#         bbox = [x,y,w,h]

#     tile,metadata = cut_bbox_video(video, *bbox, overlap=overlap)
#     if not stream:
#         hres = int(h + overlap//2)
#         wres = int(w + overlap//2)
#         # h,w,c = video.shape[-3:]
#         # height,width = h//z+overlap, w//z+overlap,
#         # imgshp, ogshp , tileshp = metadata
#         # height,width = tileshp[-3:-1]
#         # add buffer for overlap
#         # scale = np.array([h,w])/(np.array(ogshp[-3:-1])/z)
#         # height,width = np.array(tiles.shape[-3:-1])*scale
#         # height,width
#         stream = StreamV2VWrapper(
#             **streamcfg,
#             width=wres,
#             height=hres,
#         )

#     if loras:
        
#         triglist = lora.loc[list(loras.keys())].trigs.dropna().str.split(',').to_list()
#         trigs = list(dict.fromkeys(flattenList(triglist)))
#         trig = ' '.join([f'{t}' for t in trigs])
#         prompt = f'{trig} {prompt}'

#         for lor in loras.keys():
#             try:
#                 stream.stream.load_lora(lora.loc[lor].pth, adapter_name=lor)
#             except ValueError as e:
#                 print(e)
#         stream.stream.pipe.set_adapters(['lcm']+list(loras.keys()),
#                         adapter_weights=[1.0]+list(loras.values()))
        
#     assert prompt is not None
#     num_inference_steps=max(streamcfg['t_index_list'])+1
#     print(f'num_inference_steps: {num_inference_steps}')
#     print('prompt',prompt)
#     stream.prepare(
#         prompt=prompt,
#         guidance_scale=guidance_scale,
#         num_inference_steps=num_inference_steps,
#         seed=np.random.randint(0,10000),
#     )

#     tile = torch.tensor(tile).to(dtype)
#     # Gening video of wrong size?? -2px on each dim
#     # video_result = torch.zeros(tile.shape[0], stream.height, stream.width, 3)
#     # print(tile.shape)
#     # print(video_result.shape)

#     for _ in range(stream.batch_size):
#         stream(tile[0].permute(2, 0, 1))

#     for i in tqdm(range(tile.shape[0])):
#         output_image = stream(tile[i].permute(2, 0, 1))
#         res = output_image.permute(1, 2, 0)
#         if i==0:
#             print(res.shape)
#             video_result = torch.zeros(tile.shape[0], *res.shape)
#             hh,ww = res.shape[:2]
#         video_result[i] = res

#     video_result = video_result * 255
#     assert tile.shape == video_result.shape, f'{tile.shape} != {video_result.shape} NOTIMPLEMENTED RESIZING PIL'
#     # if not tile.shape == video_result.shape:
#     #     print(tile.shape)
#     #     print(video_result.shape)
#     #     print('shapes dont match')
#     #     print('resizing')
#     #     video_result = video_result[:tile.shape[0]]
#     #     print(tile.shape)
#     #     print(video_result.shape)
#     outtile = video_result.cpu().numpy().astype(np.uint8)

#     nshift = 2#max(stream.batch_size - 1, 0)
#     if nshift:
#         print(f'shifting by {nshift}')
#         lastframes = (tile[-nshift:]*255).cpu().numpy().astype(np.uint8) 
#         franken = np.concatenate([outtile[nshift:],lastframes])
#         assert franken.shape == outtile.shape, f'{franken.shape} != {outtile.shape}'
#         outtile = franken
    
#     outvideo = paste_bbox_video(video*255,np.array(outtile),metadata)#
#         # stream.height,
#         # stream.width,
#     return outvideo,stream
import skvideo.io

def writeVid(outvideo,outvidmp4,
    fps=30,
    opts={
    '-vcodec': 'libx264',  #use the h.264 codec
    '-crf': '0',           #set the constant rate factor to 0, which is lossless
    '-preset':'veryslow',   #the slower the better compression, in princple, try 
    # '-r':'30',              #set the frame rate to 30fps
                            #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }):
    opts = {
        '-r':str(fps),
        **opts,
    }
    writer = skvideo.io.FFmpegWriter(outvidmp4, outputdict=opts) 
    for frame in tqdm(outvideo): #TODO do it all at once if possible inmem
        frame = np.array(frame).astype(np.uint8)
        #write the frame as RGB not BGR
        # frame = frame[:,:,::-1]
        writer.writeFrame(frame)  

    writer.close() #close the writer
    print(f'Video saved to {outvidmp4}')

# {
#         '-vcodec': 'libx264',  # Use H.264 codec
#         '-crf': '18',          # High quality but not lossless (23 is default)
#         '-preset': 'slow',     # Balance between compression and quality
#         '-r': '30',            # 30 FPS
#         '-pix_fmt': 'yuv420p', # Ensure compatibility with most players
#         '-movflags': 'faststart',  # Move metadata to the beginning for quick indexing
#         '-g': '30'             # Set GOP to 30 for better seeking/thumbnails
#     }
 # dream

 # upsamp
import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
from pathlib import Path
# from funkshuns import *
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
from datetime import datetime
from multiprocessing import Process, Queue, freeze_support
import pandas as pd, numpy as np
import shutil
warnings.filterwarnings("ignore")
import json
import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
from pathlib import Path
# from funkshuns import *
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
from datetime import datetime
from multiprocessing import Process, Queue, freeze_support
# def getFrame(imgA, imgB, ratio=0.5,W=640,H=360,model=None,
#     shrp=3.75):
#     '''W,H OG width/height, imgA,B are resized tensors'''
#     assert model is not None
#     _n, _c, h, w = imgA.shape
#     mid = model.inference(imgA, imgB, ratio)
#     mid = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
#     mid = cv2.resize(mid, (W,H))
#     return mid

# def getcount():
#     try:
#         count = int(ctxt.read_text())
#     except:
#         count = 0
#     return count
# def increment():
#     count = getcount()
#     count += 1
#     count = count % (s+1)
#     ctxt.write_text(str(count))

def preprocess(img,rsz=( 1280,768)):
    device = getDevice()
    # img0 = cv2.imread(inpng, cv2.IMREAD_UNCHANGED)
    h,w = img.shape[:2]
    img0 = img[:]
    # asp = img.shape[1]/img.shape[0]
    # asp2 = 1.75
    # img0 = cv2.resize(img0,( w , int(w/asp2)))
    if rsz:
        img0 = cv2.resize(img0,rsz)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    # print('pre',img0.shape) #576 576??
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    # print(padding) # 0 32 0 0
    imgA = F.pad(img0, padding)
    return imgA
def getFrame(imgA, imgB, ratio=0.5,model=None,
    shrp=3.75,rsz=(1280,768)):
    # W=640,H=360 ?? resized to that before
    '''W,H OG width/height, imgA,B are resized tensors'''
    assert model is not None
    _n, _c, h, w = imgA.shape
    if rsz:
        w,h = rsz
    mid = model.inference(imgA, imgB, ratio)
    mid = mid.cpu()
    if shrp:
        mid = adjust_sharpness(mid, shrp)
    mid = (mid[0] * 255).byte().numpy().transpose(1, 2, 0)[:h, :w]
    # print('getframe',h,w)
    if rsz:
        mid = cv2.resize(mid, rsz)
    return mid
def loadModel():
    device = getDevice().type
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    modelDir = "train_log"
    try:
        try:
            from train_log.RIFE_HDv3 import Model
            model = Model(device=device)
            model.load_model(modelDir, -1)
            print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HDv2 import Model
            model = Model(device=device)
            model.load_model(modelDir, -1)
            print("Loaded v2.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model(device=device)
        model.load_model(modelDir, -1)
        print("Loaded v1.x HD model")
    if not hasattr(model, 'version'):
        model.version = 0
    model.eval()
    # model.device()
    return model

def getFrames(B,A=None,s=2,imgA=None,model=None,
    shrp=3.75,
    rsz=(1280,768),
    ):
    assert model is not None, 'Model not loaded getframes'
    H,W = B.shape[:2]
    imgB = preprocess(B,rsz=rsz)
    if imgA is None:
        assert A is not None
        # print('getframes a',A.shape)
        imgA = preprocess(A,rsz=rsz)
        # print('getframes a pre"d',imgA.shape)
    mids = []
    for c in range(s):
        ratio = (c+1)/(s+1)
        mid = getFrame(imgA, imgB, ratio, model=model,shrp=shrp,
            rsz=(W,H) if rsz else None)
        mids += [mid]
    return mids , imgB
def upsamp(frames,x=3,
    shrp=None,
    loop=True,
    rsz='fromframes'  # tuple x,y or 'fromframes' to use first frame size
    ):
    '''Upsample video by x,
    sz: tuple, resize to sz before upsampling and revert after'''
    frames = np.array(frames)
    Y,X,c = frames[0].shape
    if rsz == 'fromframes':
        rsz = (X,Y)
    model = loadModel()
    s = x-1
    A = frames[0]
    out = [A]
    B = frames[1]
    mids,imgA = getFrames(B,A=A,model=model,s=s,shrp=shrp,rsz=rsz)
    print(mids[0].shape)
    # if shrp:
    #     mids = sharpen(torch.tensor(interp/255),3.75).numpy()*255
    out += mids
    out += [B]
    framez = frames[2:] if not loop else np.concatenate([
        frames[2:],[A]],
        axis=0)
    # print(len(framez))
    for B in tqdm(framez):
        # print(len(out))
        # assert B.max() <= 256, B.max()
        # print('f')
        mids,imgA = getFrames(B,imgA=imgA,model=model,s=s,shrp=shrp,rsz=rsz)
        # assert B.max() <= 256, 'after'
        out += mids
        out += [B]
    if loop:
        out = out[:-1]
    out = [out[:Y,:X] for out in out]
    try:
        out = np.array(out)
    except Exception as e:
        print('cannot join array from list',e)
    return out

def DLsora(xl=Path(r'C:\Users\seanrm100\Desktop\Docs\TouchDesigner\TouchDiffuze\Dreamy\soraUrls.xlsx'),
           vidpth = Path(r'E:\His\VJ Content\SoraRaw')):
    assert xl.exists(), xl
    urls = pd.read_excel(xl)
    urls
    urls['nm'] = urls.nm.fillna(method='ffill')
    urls
    urls['i'] = urls.groupby('nm').cumcount()
    urls
    urls['file'] = urls.nm.str.replace(' ','_') + urls.i.astype(str) + '.mp4'
    urls
    return DLmulti(urls.url.to_list(),vidpth,urls.file.to_list())


import torch
from torchvision.io import read_video, write_video

def copy_bbox(
    video_frames,
    # output_path: str,
    source_bbox: tuple,
    target_bbox: tuple
):
    """
    Copies a bounding box from one region of each frame to another
    region of the *same* frame, for ALL frames at once,
    using advanced indexing (no explicit for loop).

    Args:
        input_path (str): Path to the input video.
        output_path (str): Where to save the modified video.
        source_bbox (tuple): (top, left, height, width) to copy FROM in each frame.
        target_bbox (tuple): (top, left, height, width) to overwrite in each frame.
    """

    # 1. Read entire video into memory
    # video_frames, audio_frames, info = read_video(input_path, pts_unit="sec")
    num_frames, height, width, channels = video_frames.shape
    # fps = info["video_fps"]

    # 2. Unpack the bounding boxes
    s_top, s_left, s_h, s_w = source_bbox
    t_top, t_left, t_h, t_w = target_bbox

    # 3. Basic checks
    if (s_h != t_h) or (s_w != t_w):
        print(s_h, t_h, s_w, t_w)
        raise ValueError(
            "source_bbox and target_bbox must have the same height and width."
        )
    
    if not (0 <= s_top < height and 0 <= s_left < width):
        raise ValueError("source_bbox is out of frame bounds.")
    if not (s_top + s_h <= height and s_left + s_w <= width):
        raise ValueError("source_bbox extends beyond frame dimensions.")
    
    if not (0 <= t_top < height and 0 <= t_left < width):
        raise ValueError("target_bbox is out of frame bounds.")
    if not (t_top + t_h <= height and t_left + t_w <= width):
        raise ValueError("target_bbox extends beyond frame dimensions.")

    # 4. Vectorized assignment (no for loop!)
    #    For each frame i, this copies video_frames[i, s_top:s_top+s_h, s_left:s_left+s_w, :]
    #    into video_frames[i, t_top:t_top+t_h, t_left:t_left+t_w, :].
    video_frames[:, t_top : t_top + t_h, t_left : t_left + t_w, :] = \
        video_frames[:, s_top : s_top + s_h, s_left : s_left + s_w, :]

    return video_frames
    # 5. Write the modified video
    # write_video(output_path, video_frames, fps)
    # print(f"Modified video saved at: {output_path}")
def xyxy_to_tlwh(x0: int, y0: int, x1: int, y1: int) -> tuple:
    """
    Convert a bounding box from (x0, y0, x1, y1) format
    to (top, left, height, width) format.

    Args:
        x0 (int): left   coordinate of top-left corner
        y0 (int): top    coordinate of top-left corner
        x1 (int): right  coordinate of bottom-right corner
        y1 (int): bottom coordinate of bottom-right corner

    Returns:
        (int, int, int, int): (top, left, height, width)
    """
    top    = y0
    left   = x0
    height = y1 - y0
    width  = x1 - x0
    return (top, left, height, width)

def remSora(vid_pth):
    '''does not /255'''
    frames, audio_frames, info = read_video(vid_pth, pts_unit="sec")
    fps = info["video_fps"]
    h,w = frames.shape[1:3]
    w,h
    x0,y0 = 750,455
    if h==720:
        x0,y0 = int(x0*1.5),int(y0*1.5)
    elif h==1080:
        x0,y0 = int(x0*2.25),int(y0*2.25)
    src_bbox = (x0, y0, w, h)
    dst_bbox = (x0, h - (h-y0)*2, w, y0)
    frames = copy_bbox(
        frames,
        xyxy_to_tlwh(*src_bbox),
        xyxy_to_tlwh(*dst_bbox),
    )
    return frames,fps

def saturate(frames, factor=1.5):
    """
    Vectorized approach to increase the saturation of every frame in a video.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the saturated video.
        factor (float): Factor by which to increase saturation (e.g., 1.5 = +50%).

    Returns:
        None: Saves the saturated video to `output_path`.
    """
    # Read video
    video_frames =frames
    num_frames, height, width, channels = video_frames.shape

    # Convert video frames (torch.Tensor) to a NumPy array for batch processing
    mods = []
    if isinstance(video_frames, torch.Tensor):
        video_np = video_frames.numpy().astype(np.uint8)
        mods += ['torch']
    else:
        video_np = video_frames

    # Convert all frames to HSV (vectorized)
    video_hsv = np.zeros_like(video_np, dtype=np.uint8)
    for i in range(num_frames):
        video_hsv[i] = cv2.cvtColor(video_np[i], cv2.COLOR_RGB2HSV)

    # Increase saturation
    video_hsv[:, :, :, 1] = np.clip(video_hsv[:, :, :, 1] * factor, 0, 255).astype(np.uint8)

    # Convert all frames back to RGB (vectorized)
    video_rgb = np.zeros_like(video_np, dtype=np.uint8)
    for i in range(num_frames):
        video_rgb[i] = cv2.cvtColor(video_hsv[i], cv2.COLOR_HSV2RGB)

    # Convert back to PyTorch tensor
    if 'torch' in mods:
        video_frames_saturated = torch.from_numpy(video_rgb)
    else:
        video_frames_saturated = video_rgb
    return video_frames_saturated

def torchen(frames,shrp=2,funk=adjust_sharpness):
    mods = []
    if isinstance(frames,np.ndarray):
        frames = torch.tensor(frames)
        mods +=['np']
    if frames.max()>1:
        frames = frames/255
        mods +=['255']
    if frames.shape[-1] in (3,4):
        frames = frames.permute(0,3,1,2)
        mods += ['c']
    sh = funk(frames, shrp)
    if 'c' in mods:
        sh = sh.permute(0,2,3,1)
    if '255' in mods:
        sh = sh*255
    if 'np' in mods:
        sh = sh.numpy()
    return sh
sharpen = lambda frames,shrp=2: torchen(frames,shrp=shrp,funk=adjust_sharpness)
contrast = lambda frames,con=1.5: torchen(frames,shrp=con,funk=adjust_contrast)
gamma = lambda frames,gam=1.5: torchen(frames,shrp=gam,funk=adjust_gamma)
# saturate = lambda frames,sat=1.5: torchen(frames,shrp=sat,funk=adjust_saturation)
hue = lambda frames,hu: torchen(frames,shrp=hu,funk=adjust_hue)
# def sharpen(frames,shrp=2):
#     mods = []
#     if isinstance(frames,np.ndarray):
#         frames = torch.tensor(frames)
#         mods +=['np']
#     if frames.max()>1:
#         frames = frames/255
#         mods +=['255']
#     if frames.shape[-1] in (3,4):
#         frames = frames.permute(0,3,1,2)
#         mods += ['c']
#     sh = adjust_sharpness(frames, shrp)
#     if 'c' in mods:
#         sh = sh.permute(0,2,3,1)
#     if '255' in mods:
#         sh = sh*255
#     if 'np' in mods:
#         sh = sh.numpy()
#     return sh

# hue = lambda frames,hu: adjust_hue(frames.permute(0,3,1,2), hu).permute(0,2,3,1)
randomhue = lambda frames: hue(frames,(np.random.random(1)[0]-0.5)*0.2)

def DL(url,toPth,filename='infer',callback=None):
    '''DL's file from url and places in toPth Path\n
    returns request status code'''
    if not callback:
        print("downloading: ",url)
    if not toPth.exists():
        toPth.mkdir()
    # assumes that the last segment after the / represents the file name
    # if url is abc/xyz/file.txt, the file name will be file.txt
    if filename=='infer':
        file_name_start_pos = url.rfind("/") + 1
        file_name = url[file_name_start_pos:]
    else:
        file_name=filename

    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(toPth/file_name, 'wb') as f:
            for data in r:
                f.write(data)
    if callback:
        callback()
    return r.status_code

from multiprocessing.pool import ThreadPool

def DLmulti(urls,toPth,filenames='infer',threds=12):
    '''DL multi async\n
    will infer filenames if filenames=='infer', otherwise
    must specify a list of len(urls) of file names with suffix, to download to toPth
    '''
    toPth.mkdir(parents=True,exist_ok=True)
    #TODO starmap_async?
    fnames = ['infer']*len(urls) if filenames=='infer' else filenames
    results = ThreadPool(threds).starmap(DL, 
        [ 
            ( url,toPth,fname,
                prog(i,len(urls),f'Downloading {url}') ) 
            for i,(url,fname) in enumerate(zip(urls,fnames)) 
            ])
    return list(zip(urls,results))

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

def pickLora(df: pd.DataFrame) -> str:    # Work on a copy and fill missing wt with 1
    df = df.copy().reset_index()
    # df = df.dropna(subset=['rating'])
    df['rating'] = df['rating'].fillna(2)
    df['wt'] = df['wt'].fillna(1)
    
    # Determine number of LORAs to use: between 0 and min(4, number of available rows)
    # Determine the maximum number of LORAs possible
    max_possible = min(4, len(df))
    
    # Define default probabilities for selecting 0 to 4 LORAs.
    # This distribution makes 2 or 3 very likely, 1 moderately likely,
    # 4 a bit less likely, and 0 quite rare.
    default_probs = {0: 0.05, 1: 0.15, 2: 0.35, 3: 0.35, 4: 0.10}
    possible_nums = list(range(max_possible + 1))
    # Restrict to possible numbers and normalize the probabilities
    probs = np.array([default_probs[i] for i in possible_nums])
    probs /= probs.sum()
    
    # Select the number of LORAs using the custom probability distribution
    num_to_select = np.random.choice(possible_nums, p=probs)
    if num_to_select == 0:
        return ""
    
    # Compute sampling weights based on rating, using squared ratings to favor higher values.
    # For instance, a rating of 2 gives weight 4 while a rating of 4 gives weight 16.
    weights = df['rating'] ** 2
    
    # Randomly select the rows using the computed weights
    selected = df.sample(n=num_to_select, weights=weights, replace=False)
    
    # Parameters for the skewed multiplier:
    # We want a base multiplier with mean ~1 and ~80% of samples between 0.2 and 1.4.
    # Using a skew-normal distribution with negative skew (a = -5) tends to push values lower.
    a = -5         # shape parameter (negative value => left-skewed)
    scale = 0.3    # scale of variation
    loc = 1.234    # location parameter set so that the mean multiplier is near 1
    
    result_parts = []
    for _, row in selected.iterrows():
        # Draw a random multiplier from the skew-normal distribution.
        multiplier = skewnorm.rvs(a, loc=loc, scale=scale)
        # Ensure the multiplier is positive (fallback if an extreme sample occurs)
        multiplier = max(multiplier, 0.1)
        # Compute the new weight, rounding to two decimals
        new_wt = round(row['wt'] * multiplier, 2)
        result_parts.append(f"{row['nm']}:{new_wt:.1f}")
    
    return ",".join(result_parts)


'''utility funkshuns'''
from datetime import datetime
import errno
import pandas as pd
import numpy as np
from pathlib import PurePath,Path
import os
import shutil
from statistics import mean
import math
import json
import subprocess
printImportWarnings = False

regxASCII = "[^\W\d_]+"
import requests

def timeit(func):
    def wrapper(*args, **kwargs):
        nowe = datetime.now()
        print(f'Running {func.__name__}...\n \
            Started: {nowe.strftime("%H:%M")}\n') 
        ret = func(*args, **kwargs)
        timd = datetime.now()-nowe
        print(f'Finished in {timd}')
        return ret
    return wrapper
def prog(index, total,title='', bar_len=50 ):
    '''
    prints current progress to progress bar\n
    index is expected to be 0 based index. 
    0 <= index < total
    '''
    percent_done = (index)/total*100
    percent_done = np.ceil(percent_done*10)/10

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    
    emoji = '⏳' if (round(percent_done) < 100) else '✅'
    print(f'\t{emoji}{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')


import pathlib
from concurrent.futures import ThreadPoolExecutor, Executor
from typing import Union, List, Tuple
import numpy as np
from PIL import Image


def _saveWrapper(pair: Tuple[Image.Image, pathlib.Path]) -> pathlib.Path:
    """
    Helper function to save an image to a specified path.

    :param pair: Tuple (img, outpng) where img is a PIL.Image and outpng is the destination path.
    :return: The pathlib.Path of the saved image.
    """
    img, outpng = pair
    img.save(str(outpng))
    return outpng


def writePngs(
    imgs: Union[List[Image.Image], np.ndarray],
    outPngs: List[pathlib.Path],
    executorType: type[Executor] = ThreadPoolExecutor,
    **convertKwargs
) -> List[pathlib.Path]:
    """
    Save a collection of images concurrently to specified PNG file paths.

    Accepts a list of PIL.Image objects or a numpy array (first dimension indexes images).
    When given a numpy array, each image is converted to a PIL.Image using Image.fromarray
    with any extra keyword arguments passed via `convertKwargs`. The output file paths are supplied
    in `outpngs` and must correspond to the images.

    :param imgs: List of PIL.Image objects or numpy array with images along the first axis.
    :param outpngs: List of pathlib.Path objects where each image is saved.
    :param executorType: Executor class for parallel processing (default: ThreadPoolExecutor).
    :param convertKwargs: Optional keyword arguments for Image.fromarray.
    :return: List of pathlib.Path objects for the saved PNG files.
    
    **Example usage:**
    
    >>> import numpy as np, pathlib
    >>> imgs = np.random.randint(0, 255, (2, 100, 100, 3), dtype=np.uint8)
    >>> outpngs = [pathlib.Path(f"output/img_{i}.png") for i in range(2)]
    >>> saved = writePngs(imgs, outpngs)
    >>> for p in saved: print(p)
    output/img_0.png
    output/img_1.png
    """
    if isinstance(imgs, np.ndarray):
        imgsList = [Image.fromarray(img, **convertKwargs) for img in imgs.astype(np.uint8)]
    else:
        imgsList = [
            img if isinstance(img, Image.Image) else Image.fromarray(img, **convertKwargs)
            for img in imgs
        ]

    if isinstance(outPngs, pathlib.Path):
        if outPngs.is_dir():
            outpngs = [outPngs / f"{i}.png" for i in range(len(imgsList))]
        else:
            outpngs = [outPngs]
    else:
        outpngs = outPngs

    assert len(imgsList) == len(outpngs), f"Number of images and output paths do not match: {len(imgsList)} vs {len(outpngs)}"

    for outpng in outpngs:
        outpng.parent.mkdir(parents=True, exist_ok=True)

    with executorType() as executor:
        futures = [
            executor.submit(_saveWrapper, (img, outpng))
            for img, outpng in zip(imgsList, outpngs)
        ]
    return [f.result() for f in futures]
