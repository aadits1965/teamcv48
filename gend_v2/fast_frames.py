"""Output-identical, faster frame reader for GenD's detector.py (commit 387a422).

The original generator reads each selected frame with iio.imread(index=k), which re-opens and
re-seeks the video for every frame. This version decodes the video ONCE sequentially and caches
the first `prefetch_factor * num_frames` frame ids of the detector's own selection order
(max_spread_permutation_pq, mode='at_least'). Any further id falls back to the original reader.
Detection, face selection, alignment and saving are unchanged (authors' functions).
Only active for mode='at_least' without masks; otherwise delegates to the original generator.
"""
import av
import cv2
import imageio.v3 as iio
import detector as _det

_ORIG = _det.get_video_frames_generator


def n_frames(path):
    """Frame count exactly as the authors' reader gets it (iio.improps(...).shape[0], pyav plugin).
    Some DFD files carry no frame-count metadata (improps -> 0, the authors' reader then crashes);
    only in that case the count is obtained by demuxing the video packets.
    Returns (count, used_fallback)."""
    try:
        n = int(iio.improps(path, plugin="pyav").shape[0])
    except Exception:
        n = 0
    if n > 0:
        return n, False
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            n = sum(1 for pkt in c.demux(s) if pkt.size > 0)
    except Exception as e:
        print(f"Warning: cannot count frames of {path}: {e}")
        n = 0
    return n, True


def fast_generator(source_path, mask_path, stride=1, num_frames=32, mode="at_least", prefetch_factor=2):
    if mask_path is not None or mode != "at_least":
        yield from _ORIG(source_path, mask_path, stride=stride, num_frames=num_frames, mode=mode)
        return
    total, _ = n_frames(source_path)
    if total <= 0:
        print(f"Warning: Video {source_path} cannot be opened or has no frames!")
        return
    order = _det.max_spread_permutation_pq(total, start=total // 2)
    k = total if num_frames == -1 else num_frames
    wanted = set(order[: prefetch_factor * k])
    last = max(wanted)
    cache = {}
    try:
        with av.open(source_path) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for i, fr in enumerate(c.decode(s)):
                if i in wanted:
                    cache[i] = fr.to_ndarray(format="rgb24")
                if i >= last:
                    break
    except Exception as e:
        print(f"Warning: sequential decode failed for {source_path} ({e}); using original reader")
        cache = {}
    for fid in order:
        rgb = cache.pop(fid, None)
        if rgb is None:
            try:
                rgb = iio.imread(source_path, index=fid, plugin="pyav")
            except Exception as e:
                print(f"Warning: Failed to read frame {fid} of {source_path}. Skipping. {e}")
                continue
        yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), fid, None


def install():
    _det.get_video_frames_generator = fast_generator


def uninstall():
    _det.get_video_frames_generator = _ORIG


# ----------------------------------------------------------------------------------------------
# Mask-aware variant (used for DFD fakes, where the README passes manipulation masks).
# Mirrors the original generator: total = min(video frames, mask frames); frames whose mask cannot
# be read are skipped; empty-mask handling and face choice stay in the authors' process_video.
# The FF++ (no-mask) path above is unchanged.
# ----------------------------------------------------------------------------------------------
def _decode_wanted(path, wanted):
    cache = {}
    if not wanted:
        return cache
    last = max(wanted)
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            if i in wanted:
                cache[i] = fr.to_ndarray(format="rgb24")
            if i >= last:
                break
    return cache


def fast_generator_masks(source_path, mask_path, stride=1, num_frames=32, mode="at_least", prefetch_factor=3):
    if mask_path is None:
        yield from fast_generator(source_path, None, stride=stride, num_frames=num_frames, mode=mode)
        return
    if mode != "at_least":
        yield from _ORIG(source_path, mask_path, stride=stride, num_frames=num_frames, mode=mode)
        return
    video_frames, _ = n_frames(source_path)
    mask_frames, _ = n_frames(mask_path)
    total = min(video_frames, mask_frames)
    if total <= 0:
        print(f"Warning: {source_path} / {mask_path} cannot be opened or have no frames!")
        return
    order = _det.max_spread_permutation_pq(total, start=total // 2)
    k = total if num_frames == -1 else num_frames
    wanted = set(order[: prefetch_factor * k])
    try:
        cv_, cm_ = _decode_wanted(source_path, wanted), _decode_wanted(mask_path, wanted)
    except Exception as e:
        print(f"Warning: sequential decode failed for {source_path} ({e}); using original reader")
        cv_, cm_ = {}, {}
    for fid in order:
        rgb = cv_.pop(fid, None)
        if rgb is None:
            try:
                rgb = iio.imread(source_path, index=fid, plugin="pyav")
            except Exception as e:
                print(f"Warning: Failed to read frame {fid} of {source_path}. Skipping. {e}")
                continue
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mrgb = cm_.pop(fid, None)
        if mrgb is None:
            try:
                mrgb = iio.imread(mask_path, index=fid, plugin="pyav")
            except Exception as e:
                print(f"Warning: Failed to read mask frame {fid} of {mask_path}. Skipping. {e}")
                continue
        yield frame, fid, cv2.cvtColor(mrgb, cv2.COLOR_RGB2BGR)


def install_masks():
    _det.get_video_frames_generator = fast_generator_masks
