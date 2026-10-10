import gzip
import socket
import traceback
import urllib.request
import os
import tempfile
from io import BytesIO

import torch
import folder_paths
import nodes

from aiohttp import web
from comfy_api.latest import io, ComfyExtension, InputImpl
from server import PromptServer


# ============================================================
# Configuration
# ============================================================

# Default port used by the node. The actual port is selectable per node.
DEFAULT_REMOTE_PORT = 8188

# CLIPロード + VAE encode + H3 conditioning は
# モデルのロード状況によって時間がかかるため長めに設定
REQUEST_TIMEOUT = 600


# ============================================================
# Serialization
# ============================================================

def pack_data(obj):
    """
    Python object / Tensor / CONDITIONING / LATENT を
    torch.save() でシリアライズし、gzip圧縮する。

    モデル本体は含まれない。
    """

    buffer = BytesIO()

    torch.save(
        obj,
        buffer,
        pickle_protocol=5,
    )

    return gzip.compress(
        buffer.getvalue(),
        compresslevel=3,
    )


def unpack_data(data):
    """
    PC-Bから受信したデータを復元する。
    """

    raw = gzip.decompress(data)

    return torch.load(
        BytesIO(raw),
        map_location="cpu",
        weights_only=False,
    )


# ============================================================
# Reference video file transfer / decoding
# ============================================================

def _read_video_file(path):
    """Read the original reference video file without decoding it on PC-A."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference video file not found: {path}")

    with open(path, "rb") as f:
        data = f.read()

    return {
        "filename": os.path.basename(path),
        "extension": os.path.splitext(path)[1].lower(),
        "data": data,
    }


def _decode_reference_video(video_file):
    """Decode a transferred reference video using ComfyUI's built-in VIDEO implementation."""
    if not isinstance(video_file, dict):
        raise TypeError("Reference video payload must be a dict")

    filename = video_file.get("filename", "reference_video.mp4")
    extension = video_file.get("extension", "")
    file_data = video_file.get("data")

    if not isinstance(file_data, (bytes, bytearray)):
        raise TypeError("Reference video payload does not contain file bytes")

    suffix = extension if extension else os.path.splitext(filename)[1]
    if not suffix:
        suffix = ".mp4"

    temp_path = None
    try:
        # ComfyUI's built-in VideoFromFile expects a filesystem path.
        # The received file is therefore written to a temporary file, then
        # decoded by ComfyUI itself.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(file_data)
            temp_path = f.name

        video = InputImpl.VideoFromFile(temp_path)
        components = video.get_components()
        images = components.images

        if images is None:
            raise RuntimeError("ComfyUI video decoder returned no reference frames")

        if not isinstance(images, torch.Tensor):
            raise TypeError(
                f"ComfyUI video decoder returned unexpected frame type: {type(images).__name__}"
            )

        if images.ndim != 4 or images.shape[-1] != 3:
            raise RuntimeError(
                f"ComfyUI video decoder returned unexpected frame shape: {tuple(images.shape)}"
            )

        return images

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _debug_dump(name, obj, indent=2):
    prefix = " " * indent
    if obj is None:
        print(f"{prefix}{name}: None")
    elif isinstance(obj, torch.Tensor):
        print(
            f"{prefix}{name}: Tensor "
            f"shape={tuple(obj.shape)} dtype={obj.dtype} "
            f"device={obj.device} numel={obj.numel()} "
            f"bytes={obj.numel() * obj.element_size()}"
        )
    elif isinstance(obj, dict):
        print(f"{prefix}{name}: dict keys={list(obj.keys())}")
        for key, value in obj.items():
            if key == "data" and isinstance(value, (bytes, bytearray)):
                print(f"{' ' * (indent + 2)}{name}['data']: bytes len={len(value)}")
            else:
                _debug_dump(f"{name}[{key!r}]", value, indent + 2)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {type(obj).__name__} len={len(obj)}")
        for i, value in enumerate(obj):
            _debug_dump(f"{name}[{i}]", value, indent + 2)
    else:
        print(f"{prefix}{name}: type={type(obj).__name__} value={obj!r}")


def _debug_reference_data(label, data, debug=False):
    if not debug:
        return
    print(f"[RemoteMiniMaxH3][DEBUG] {label}")
    for key in ("ref_images", "ref_videos", "ref_video_audios", "ref_audios"):
        _debug_dump(key, data.get(key))
    print(f"[RemoteMiniMaxH3][DEBUG] End {label}")


def _debug_print(debug, message):
    if debug:
        print(f"[RemoteMiniMaxH3][DEBUG] {message}")


# ============================================================
# PC-B
#
# Remote execution endpoint
#
# PC-B側で実際に
#
#   CLIPロード
#   VAEロード
#   MiniMax H3 ImageToVideo
#
# を実行する。
# ============================================================

async def _remote_minimax_h3_process(request, reference=False):

    try:

        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            from comfy_aimdo import control as coctrl
            coctrl.set_dynamic_vram(True)

        # Allow Remote MiniMax H3 payloads up to 1 GiB.
        request._client_max_size = 1024 * 1024 * 1024

        request_body = await request.read()
        request_data = unpack_data(request_body)
        debug = bool(request_data.get("debug", False))

        _debug_print(debug, f"Request received: {len(request_body) / 1024 / 1024:.2f} MB")

        if reference:
            _debug_reference_data(
                "PC-B received Reference data (before video decode)",
                request_data,
                debug,
            )
            ref_videos = request_data.get("ref_videos") or {}
            decoded_videos = {}
            for key, video_file in ref_videos.items():
                _debug_print(debug, f"Decoding reference video: {key} ({video_file.get('filename', 'unknown')})")
                decoded_videos[key] = _decode_reference_video(video_file)
            request_data["ref_videos"] = decoded_videos
            _debug_reference_data("PC-B Reference data (after video decode)", request_data, debug)

        clip_name = request_data["clip_name"]
        vae_name = request_data["vae_name"]
        prompt = request_data["prompt"]
        width = int(request_data["width"])
        height = int(request_data["height"])
        length = int(request_data["length"])

        if debug:
            print(
                "[RemoteMiniMaxH3][DEBUG] "
                + ("Reference to Video" if reference else "Image to Video")
                + " request received."
            )
            print(f"  CLIP   : {clip_name}")
            print(f"  VAE    : {vae_name}")
            if reference:
                audio_vae_name = request_data["audio_vae_name"]
                print(f"  Audio VAE: {audio_vae_name}")
                print(f"  Ref size: {request_data.get('ref_image_size', 'match')}")
            print(f"  Size   : {width} x {height}")
            print(f"  Length : {length}")

        if reference:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        else:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

        clip = nodes.CLIPLoader().load_clip(clip_name)[0]
        _debug_print(debug, "CLIP loaded on PC-B.")

        vae = nodes.VAELoader().load_vae(vae_name)[0]
        _debug_print(debug, "VAE loaded on PC-B.")

        if reference:
            audio_vae = nodes.VAELoader().load_vae(
                request_data["audio_vae_name"]
            )[0]
            _debug_print(debug, "Audio VAE loaded on PC-B.")

            result = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                ref_image_size=request_data.get("ref_image_size", "match"),
                vae=vae,
                audio_vae=audio_vae,
                ref_images=request_data.get("ref_images"),
                ref_videos=request_data.get("ref_videos"),
                ref_video_audios=request_data.get("ref_video_audios"),
                ref_audios=request_data.get("ref_audios"),
            )
        else:
            result = MiniMaxH3ImageToVideo.execute(
                clip=clip,
                vae=vae,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                first_frame=request_data.get("first_frame"),
                last_frame=request_data.get("last_frame"),
            )

        conditioning = result[0]
        latent = result[1]

        _debug_print(
            debug,
            ("MiniMax H3 Reference to Video" if reference else "MiniMax H3 Image to Video")
            + " conditioning completed."
        )

        response_body = pack_data({
            "conditioning": conditioning,
            "latent": latent,
        })

        _debug_print(
            debug,
            f"Response size: {len(response_body) / 1024 / 1024:.2f} MB"
        )

        return web.Response(
            body=response_body,
            content_type="application/octet-stream",
        )

    except Exception:

        error_text = traceback.format_exc()

        print("[RemoteMiniMaxH3] ERROR on PC-B:")
        print(error_text)

        return web.Response(
            text=error_text,
            status=500,
        )


@PromptServer.instance.routes.post("/remote_minimax_h3")
async def remote_minimax_h3(request):
    return await _remote_minimax_h3_process(request, reference=False)


@PromptServer.instance.routes.post("/remote_minimax_h3_reference_to_video")
async def remote_minimax_h3_reference_to_video(request):
    return await _remote_minimax_h3_process(request, reference=True)


# ============================================================
# PC-A
#
# Remote MiniMax H3 Image to Video
#
# CLIP / VAEは「ファイル名」だけを指定する。
#
# PC-AではCLIPもVAEもロードしない。
# ============================================================

class RemoteMiniMaxH3ImageToVideo(
    io.ComfyNode
):

    @classmethod
    def define_schema(cls):

        # ----------------------------------------------------
        # Get model filename lists.
        #
        # These are only filenames.
        # No model is loaded here.
        # ----------------------------------------------------

        clip_list = (
            folder_paths.get_filename_list(
                "text_encoders"
            )
        )

        vae_list = (
            folder_paths.get_filename_list(
                "vae"
            )
        )

        # Avoid an empty Combo input
        # if the corresponding model folder
        # happens to be empty.
        if not clip_list:
            clip_list = [""]

        if not vae_list:
            vae_list = [""]

        # ----------------------------------------------------
        # Schema
        # ----------------------------------------------------

        return io.Schema(

            node_id=(
                "RemoteMiniMaxH3ImageToVideo"
            ),

            display_name=(
                "Remote MiniMax H3 Image to Video"
            ),

            category=(
                "model/conditioning/minimax"
            ),

            description=(
                "Run MiniMax H3 Image to Video "
                "conditioning on a remote ComfyUI PC. "
                "CLIP and VAE remain on the remote PC."
            ),

            inputs=[

                # --------------------------------------------
                # PC-B hostname / IP
                # --------------------------------------------

                io.String.Input(
                    "target_pc_name",
                    default="Z840",
                    multiline=False,
                ),

                io.Int.Input(
                    "remote_port",
                    default=DEFAULT_REMOTE_PORT,
                    min=1,
                    max=65535,
                    step=1,
                ),

                io.Boolean.Input(
                    "debug",
                    default=False,
                ),

                # --------------------------------------------
                # CLIP filename
                #
                # This is NOT a CLIP object.
                # It is only the filename used on PC-B.
                # --------------------------------------------

                io.Combo.Input(
                    "clip_name",
                    options=clip_list,
                    default=clip_list[0],
                ),

                # --------------------------------------------
                # VAE filename
                #
                # This is NOT a VAE object.
                # It is only the filename used on PC-B.
                # --------------------------------------------

                io.Combo.Input(
                    "vae_name",
                    options=vae_list,
                    default=vae_list[0],
                ),

                # --------------------------------------------
                # Prompt
                # --------------------------------------------

                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                ),

                # --------------------------------------------
                # Video parameters
                # --------------------------------------------

                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=nodes.MAX_RESOLUTION,
                    step=32,
                ),

                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=nodes.MAX_RESOLUTION,
                    step=32,
                ),

                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                ),

                # --------------------------------------------
                # First frame
                # --------------------------------------------

                io.Image.Input(
                    "first_frame",
                    optional=True,
                ),

                # --------------------------------------------
                # Last frame
                # --------------------------------------------

                io.Image.Input(
                    "last_frame",
                    optional=True,
                ),
            ],

            # ------------------------------------------------
            # Outputs
            # ------------------------------------------------

            outputs=[

                io.Conditioning.Output(
                    display_name="positive"
                ),

                io.Latent.Output(
                    display_name="av_latent"
                ),
            ],
        )

    # ========================================================
    # Execute
    #
    # PC-A側で実行される。
    # ========================================================

    @classmethod
    def execute(
        cls,

        target_pc_name,

        remote_port,

        debug,

        clip_name,

        vae_name,

        prompt,

        width,

        height,

        length,

        first_frame=None,

        last_frame=None,
    ):

        # ----------------------------------------------------
        # Resolve PC-B hostname
        # ----------------------------------------------------

        try:

            resolved_ip = socket.gethostbyname(
                target_pc_name
            )

        except socket.gaierror as e:

            raise RuntimeError(
                f"PC名 '{target_pc_name}' "
                f"をIPアドレスへ解決できません。\n"
                f"{e}"
            )

        # ----------------------------------------------------
        # Remote endpoint
        # ----------------------------------------------------

        url = (
            f"http://"
            f"{resolved_ip}:"
            f"{int(remote_port)}"
            f"/remote_minimax_h3"
        )

        # ----------------------------------------------------
        # Prepare request
        # ----------------------------------------------------
        #
        # Important:
        #
        # We send only:
        #
        #   CLIP filename
        #   VAE filename
        #   Prompt
        #   Width
        #   Height
        #   Length
        #   First frame
        #   Last frame
        #
        # The CLIP/VAE model itself is NOT sent.
        #

        request_data = {

            "clip_name": clip_name,

            "vae_name": vae_name,

            "prompt": prompt,

            "width": int(width),

            "height": int(height),

            "length": int(length),

            "first_frame": first_frame,

            "last_frame": last_frame,
            "debug": bool(debug),
        }

        request_body = pack_data(
            request_data
        )

        if debug:
            print("[RemoteMiniMaxH3][DEBUG] Sending request to PC-B:")
            print(f"  PC     : {target_pc_name}")
            print(f"  IP     : {resolved_ip}")
            print(f"  Port   : {int(remote_port)}")
            print(f"  CLIP   : {clip_name}")
            print(f"  VAE    : {vae_name}")
            print(f"  Payload: {len(request_body) / 1024 / 1024:.2f} MB")

        # ----------------------------------------------------
        # HTTP POST
        # ----------------------------------------------------

        request = urllib.request.Request(

            url,

            data=request_body,

            method="POST",

            headers={
                "Content-Type":
                    "application/octet-stream"
            },
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=REQUEST_TIMEOUT,
            ) as response:

                response_body = (
                    response.read()
                )

        except Exception as e:

            raise RuntimeError(
                "PC-BへのRemote MiniMax H3 "
                "通信に失敗しました。\n"
                f"URL: {url}\n"
                f"Error: {e}"
            )

        _debug_print(debug, "Response received:")
        _debug_print(debug, f"  Size: {len(response_body) / 1024 / 1024:.2f} MB")

        # ----------------------------------------------------
        # Decode response
        # ----------------------------------------------------

        result = unpack_data(
            response_body
        )

        conditioning = (
            result["conditioning"]
        )

        latent = (
            result["latent"]
        )

        # ----------------------------------------------------
        # Return to PC-A
        # ----------------------------------------------------

        return io.NodeOutput(
            conditioning,
            latent,
        )


# ============================================================
# PC-A
#
# Remote MiniMax H3 Reference to Video
#
# CLIP / VAE / Audio VAEは「ファイル名」だけを指定する。
#
# PC-Aではモデルをロードしない。
# Reference image / video / audio は必要なテンソルだけを
# PC-Bへ送る。
# ============================================================

class RemoteMiniMaxH3ReferenceToVideo(
    io.ComfyNode
):

    @classmethod
    def define_schema(cls):

        clip_list = folder_paths.get_filename_list("text_encoders")
        vae_list = folder_paths.get_filename_list("vae")

        if not clip_list:
            clip_list = [""]
        if not vae_list:
            vae_list = [""]

        return io.Schema(
            node_id="RemoteMiniMaxH3ReferenceToVideo",
            display_name="Remote MiniMax H3 Reference to Video",
            category="model/conditioning/minimax",
            description=(
                "Run MiniMax H3 Reference to Video conditioning "
                "on a remote ComfyUI PC. CLIP, video VAE and "
                "audio VAE remain on the remote PC."
            ),
            inputs=[
                io.String.Input("target_pc_name", default="Z840", multiline=False),
                io.Int.Input(
                    "remote_port",
                    default=DEFAULT_REMOTE_PORT,
                    min=1,
                    max=65535,
                    step=1,
                ),
                io.Boolean.Input(
                    "debug",
                    default=False,
                ),
                io.Combo.Input("clip_name", options=clip_list, default=clip_list[0]),
                io.Combo.Input("vae_name", options=vae_list, default=vae_list[0]),
                io.Combo.Input("audio_vae_name", options=vae_list, default=vae_list[0]),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Reference image (downscaled to 2048 short edge if larger)",
                        ),
                        prefix="ref_image_", min=0, max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.String.Input(
                            "ref_video",
                            tooltip="Reference video file path (read and transferred without decoding on PC-A)",
                        ),
                        prefix="ref_video_", min=0, max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack of the same-numbered reference video",
                        ),
                        prefix="ref_video_audio_", min=0, max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        prefix="ref_audio_", min=0, max=3,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="av_latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        target_pc_name,
        remote_port,
        debug,
        clip_name,
        vae_name,
        audio_vae_name,
        prompt,
        width,
        height,
        length,
        ref_image_size="match",
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ):

        try:
            resolved_ip = socket.gethostbyname(target_pc_name)
        except socket.gaierror as e:
            raise RuntimeError(
                f"PC名 '{target_pc_name}' をIPアドレスへ解決できません。\n{e}"
            )

        url = (
            f"http://{resolved_ip}:{int(remote_port)}"
            "/remote_minimax_h3_reference_to_video"
        )

        transferred_ref_videos = {}
        for key, path in (ref_videos or {}).items():
            if path:
                transferred_ref_videos[key] = _read_video_file(path)

        request_data = {
            "clip_name": clip_name,
            "vae_name": vae_name,
            "audio_vae_name": audio_vae_name,
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(length),
            "ref_image_size": ref_image_size,
            "ref_images": ref_images,
            "ref_videos": transferred_ref_videos,
            "ref_video_audios": ref_video_audios,
            "ref_audios": ref_audios,
            "debug": bool(debug),
        }

        _debug_reference_data(
            "PC-A sending Reference data (video files)",
            request_data,
            debug,
        )
        request_body = pack_data(request_data)

        if debug:
            print("[RemoteMiniMaxH3][DEBUG] Sending Reference to Video request:")
            print(f"  PC        : {target_pc_name}")
            print(f"  IP        : {resolved_ip}")
            print(f"  Port      : {int(remote_port)}")
            print(f"  CLIP      : {clip_name}")
            print(f"  VAE       : {vae_name}")
            print(f"  Audio VAE : {audio_vae_name}")
            print(f"  Size      : {width} x {height}")
            print(f"  Length    : {length}")
            print(f"  Ref size  : {ref_image_size}")
            print(f"  Payload   : {len(request_body) / 1024 / 1024:.2f} MB")

        request = urllib.request.Request(
            url,
            data=request_body,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )

        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                response_body = response.read()
        except Exception as e:
            raise RuntimeError(
                "PC-BへのRemote MiniMax H3 Reference to Video通信に失敗しました。\n"
                f"URL: {url}\nError: {e}"
            )

        _debug_print(debug, "Response received:")
        _debug_print(debug, f"  Size: {len(response_body) / 1024 / 1024:.2f} MB")

        result = unpack_data(response_body)

        return io.NodeOutput(
            result["conditioning"],
            result["latent"],
        )


# ============================================================
# ComfyUI extension registration
# ============================================================

class RemoteMiniMaxH3Extension(
    ComfyExtension
):

    async def get_node_list(
        self
    ):
        return [
            RemoteMiniMaxH3ImageToVideo,
            RemoteMiniMaxH3ReferenceToVideo,
        ]


async def comfy_entrypoint():
    return RemoteMiniMaxH3Extension()


# ============================================================
# Legacy / normal custom-node registration
# ============================================================

NODE_CLASS_MAPPINGS = {

    "RemoteMiniMaxH3ImageToVideo":
        RemoteMiniMaxH3ImageToVideo,

    "RemoteMiniMaxH3ReferenceToVideo":
        RemoteMiniMaxH3ReferenceToVideo,
}


NODE_DISPLAY_NAME_MAPPINGS = {

    "RemoteMiniMaxH3ImageToVideo":
        "Remote MiniMax H3 Image to Video",

    "RemoteMiniMaxH3ReferenceToVideo":
        "Remote MiniMax H3 Reference to Video",
}