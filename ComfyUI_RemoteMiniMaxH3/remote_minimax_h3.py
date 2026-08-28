import gzip
import socket
import traceback
import urllib.request
from io import BytesIO

import torch
import folder_paths
import nodes

from aiohttp import web
from comfy_api.latest import io, ComfyExtension
from server import PromptServer


# ============================================================
# Configuration
# ============================================================

REMOTE_PORT = 8188

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

@PromptServer.instance.routes.post("/remote_minimax_h3")
async def remote_minimax_h3(request):

    try:

        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            from comfy_aimdo import control as coctrl
            coctrl.set_dynamic_vram(True)

        # ----------------------------------------------------
        # Receive request
        # ----------------------------------------------------

        request_body = await request.read()

        request_data = unpack_data(
            request_body
        )

        clip_name = request_data["clip_name"]
        vae_name = request_data["vae_name"]

        prompt = request_data["prompt"]

        width = int(
            request_data["width"]
        )

        height = int(
            request_data["height"]
        )

        length = int(
            request_data["length"]
        )

        first_frame = request_data.get(
            "first_frame"
        )

        last_frame = request_data.get(
            "last_frame"
        )

        print(
            "[RemoteMiniMaxH3] "
            "Remote request received."
        )

        print(
            f"  CLIP   : {clip_name}"
        )

        print(
            f"  VAE    : {vae_name}"
        )

        print(
            f"  Size   : {width} x {height}"
        )

        print(
            f"  Length : {length}"
        )

        # ----------------------------------------------------
        # Import official MiniMax H3 implementation
        # ----------------------------------------------------

        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3ImageToVideo,
        )

        # ----------------------------------------------------
        # Load CLIP on PC-B
        # ----------------------------------------------------

        clip_loader = nodes.CLIPLoader()

        clip_result = clip_loader.load_clip(
            clip_name
        )

        clip = clip_result[0]

        print(
            "[RemoteMiniMaxH3] "
            "CLIP loaded on PC-B."
        )

        # ----------------------------------------------------
        # Load VAE on PC-B
        # ----------------------------------------------------

        vae_loader = nodes.VAELoader()

        vae_result = vae_loader.load_vae(
            vae_name
        )

        vae = vae_result[0]

        print(
            "[RemoteMiniMaxH3] "
            "VAE loaded on PC-B."
        )

        # ----------------------------------------------------
        # Execute the official H3 node
        # ----------------------------------------------------
        #
        # This is deliberately calling Comfy-Org's
        # implementation instead of reproducing the H3
        # conditioning algorithm ourselves.
        #
        # Therefore, if Comfy-Org changes the H3
        # implementation, this remote node follows it.
        #

        result = MiniMaxH3ImageToVideo.execute(
            clip,
            vae,
            prompt,
            width,
            height,
            length,
            first_frame,
            last_frame,
        )

        # Official H3 node returns:
        #
        #   result[0] = CONDITIONING
        #   result[1] = LATENT
        #

        conditioning = result[0]
        latent = result[1]

        print(
            "[RemoteMiniMaxH3] "
            "MiniMax H3 conditioning completed."
        )

        # ----------------------------------------------------
        # Return only calculation results
        # ----------------------------------------------------
        #
        # No CLIP weights
        # No VAE weights
        # No UNET weights
        #
        # are returned to PC-A.
        #

        response_data = {
            "conditioning": conditioning,
            "latent": latent,
        }

        response_body = pack_data(
            response_data
        )

        print(
            "[RemoteMiniMaxH3] "
            f"Response size: "
            f"{len(response_body) / 1024 / 1024:.2f} MB"
        )

        return web.Response(
            body=response_body,
            content_type=(
                "application/octet-stream"
            ),
        )

    except Exception:

        error_text = traceback.format_exc()

        print(
            "[RemoteMiniMaxH3] "
            "ERROR on PC-B:"
        )

        print(error_text)

        return web.Response(
            text=error_text,
            status=500,
        )


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
            f"{REMOTE_PORT}"
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
        }

        request_body = pack_data(
            request_data
        )

        print(
            "[RemoteMiniMaxH3] "
            "Sending request to PC-B:"
        )

        print(
            f"  PC     : {target_pc_name}"
        )

        print(
            f"  IP     : {resolved_ip}"
        )

        print(
            f"  CLIP   : {clip_name}"
        )

        print(
            f"  VAE    : {vae_name}"
        )

        print(
            f"  Payload: "
            f"{len(request_body) / 1024 / 1024:.2f} MB"
        )

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

        print(
            "[RemoteMiniMaxH3] "
            "Response received:"
        )

        print(
            f"  Size: "
            f"{len(response_body) / 1024 / 1024:.2f} MB"
        )

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
# ComfyUI extension registration
# ============================================================

class RemoteMiniMaxH3Extension(
    ComfyExtension
):

    async def get_node_list(
        self
    ):
        return [
            RemoteMiniMaxH3ImageToVideo
        ]


async def comfy_entrypoint():
    return RemoteMiniMaxH3Extension()


# ============================================================
# Legacy / normal custom-node registration
# ============================================================

NODE_CLASS_MAPPINGS = {

    "RemoteMiniMaxH3ImageToVideo":
        RemoteMiniMaxH3ImageToVideo,
}


NODE_DISPLAY_NAME_MAPPINGS = {

    "RemoteMiniMaxH3ImageToVideo":
        "Remote MiniMax H3 Image to Video",
}