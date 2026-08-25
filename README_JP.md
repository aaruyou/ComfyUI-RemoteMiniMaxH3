# ComfyUI Remote MiniMax H3

[English](README.md) | [日本語](README_JP.md)

**MiniMax H3 の CLIP / Text Encoder 処理を別PCへ分離し、メインPCのVRAM使用量を削減するComfyUIカスタムノードです。**

このカスタムノードは、LANで接続された2台のPCにMiniMax H3の処理を分散します。

巨大なMiniMax H3 Text Encoderや関連モデルをUNETと同じGPUにロードする代わりに、必要な入力を別PCのComfyUIへ送り、そこでH3の処理を行います。

そして、処理結果だけをメインPCへ返します。

**巨大なモデルファイルそのものをPC間で転送する必要はありません。**

---

## なぜ作ったのか？

MiniMax H3では非常に大きなText Encoderや関連モデルを使用します。

VRAM容量が限られたGPUで全てを1台にロードすると、

- VRAM不足
- 頻繁なモデルオフロード
- GPU ↔ CPU間のメモリ転送
- 大幅な処理速度低下
- ComfyUIの操作自体がカクつく
- UNETやアップスケール、Refinerを実行する余裕がなくなる

といった問題が発生する場合があります。

そこで、H3モデルを処理できる別PCがあるなら、**H3関連処理をそちらへ分離する**という方法を取ります。

### 分離前

```text
┌──────────────────────────────┐
│          PC-A                 │
│                              │
│  CLIP / Text Encoder         │
│  MiniMax H3 Text Encoder     │
│  Video VAE                   │
│  UNET                        │
│  Upscaler / Refiner          │
│                              │
│       VRAM OVERLOAD          │
└──────────────────────────────┘
```

### Remote MiniMax H3 使用時

```text
┌─────────────────────┐
│       PC-A          │
│                     │
│       UNET          │
│  Upscaler / Refiner │
│                     │
└──────────┬──────────┘
           │
           │ LAN
           │
           │ 処理データ
           │
           ▼
┌─────────────────────┐
│       PC-B          │
│                     │
│  CLIP               │
│  H3 Text Encoder    │
│  Video VAE          │
│                     │
└─────────────────────┘
```

これにより、PC-AのGPUを主にUNETやサンプリング処理へ使えるようになります。

---

## 仕組み

メインPC（PC-A）から、必要な情報をリモート側のComfyUI（PC-B）へ送信します。

PC-Bでは、

1. 指定されたCLIPをロード
2. 指定されたVAEをロード
3. MiniMax H3の処理を実行
4. 必要なCONDITIONING / LATENT等の処理結果を生成
5. 結果をPC-Aへ返送

という処理を行います。

巨大なモデルファイルはPC-Bに残ったままです。

```text
PC-A
  │
  │ Prompt
  │ CLIP/VAE names
  │ First Frame
  │ Last Frame
  │ Width / Height
  │ Length
  │
  ▼
PC-B
  │
  │ CLIP
  │ H3 Text Encoder
  │ Video VAE
  │ MiniMax H3 processing
  │
  ▼
PC-A
  │
  │ Conditioning / Latent
  ▼
UNET / Sampler
```

---

## ネットワーク通信量

この方式の大きなメリットの一つは、**巨大なモデルそのものをPC間で転送しない**ことです。

実際のテストでは、

```text
PC-A → PC-B : 4.60 MB
PC-B → PC-A : 15.01 MB

合計        : 19.61 MB
```

でした。

テストで使用したMiniMax H3 Text EncoderはPC-B上で約**25.9GB staged**となりましたが、PC-Aへ返された結果は約**15MB**でした。

そのため、高速なローカルネットワーク環境で実用的な構成を作ることができます。

---

## 構成例

例えば次のような構成です。

```text
PC-A
GPU: RTX 3090 24GB

担当:
- UNET
- Sampling
- Latent Upscaling
- Refiner
```

```text
PC-B
GPU: MiniMax H3をロードできる十分なVRAM

担当:
- CLIP
- MiniMax H3 Text Encoder
- MiniMax H3 Video VAE
- H3 preprocessing
```

2台のPCでそれぞれComfyUIを起動します。

PC-B側のComfyUIは、PC-Aから通信できるように

```text
--listen
```

を付けて起動します。

---

## VRAMとパフォーマンス

このプロジェクトの主目的は、単純にGPUの演算能力を増やすことではなく、**VRAM使用量を削減すること**です。

H3関連の巨大なモデルをPC-Bへ移すことで、PC-AではUNET処理に利用できるVRAMを大きく確保できます。

実際のテストでは、H3関連処理をPC-Bへ分離した後、24GBのRTX 3090で時分割サンプリングを行い、VRAM使用量が約

```text
~17 GB
```

に収まりました。

これによりVRAM圧迫、モデルオフロード、メモリ転送による性能低下を大幅に抑えられる可能性があります。

---

## シード違いの生成

CLIP / Text Encoderによる結果は、通常サンプリングのシードには依存しません。

そのため、同じPromptやその他の入力条件で複数のシードを試す場合、**高コストなH3 conditioning処理を再利用し、シード依存のサンプリングだけをPC-A側で繰り返す**ことができます。

概念的には、

```text
                 ┌─ Seed 1 ─┐
                 ├─ Seed 2 ─┤
H3 Conditioning ─┼─ Seed 3 ─┼─ UNET / Sampling
                 ├─ Seed 4 ─┤
                 └─ Seed 5 ─┘
```

という構成です。

大量のシードを試す用途で特に有効です。

---

## 必要環境

- ComfyUI
- 両方のPCにMiniMax H3対応環境
- ネットワーク接続された2台のPC
- `--listen` を付けて起動したPC-B側ComfyUI
- MiniMax H3モデルをロードできる十分なPC-B側VRAM
- PC-AとPC-B間のネットワーク通信

両PCには、互換性のあるComfyUI環境と必要なMiniMax H3モデルを用意してください。

---

## インストール

このリポジトリをComfyUIの `custom_nodes` ディレクトリへ配置します。

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI_RemoteMiniMaxH3/
        ├── __init__.py
        └── ...
```

2台のComfyUIインスタンスでノードを使用する場合は、両方のPCへインストールしてください。

インストール後、ComfyUIを再起動します。

---

## 設定

このノードには主に以下の入力があります。

| Input | 説明 |
|---|---|
| `PC Name` | リモート側ComfyUIを実行しているPCのホスト名またはIPアドレス |
| `CLIP File Name` | MiniMax H3で使用するCLIPモデル |

例：

```text
PC Name:
Z840

CLIP:
MiniMax\qwen3vl_32b_minimax_h3_int8_convrot.safetensors
```

対応するVAEはリモート側のH3処理で扱われます。

---

## 注意事項

このプロジェクトは主に**ローカル / プライベートネットワークでの使用**を想定しています。

適切なネットワークセキュリティ対策なしに、リモート側のComfyUIサーバーをインターネットへ直接公開しないでください。

---

## プロジェクトの状態

**Experimental / Work in Progress**

このプロジェクトは、MiniMax H3を複数PCで実行する際のVRAM不足を解決するために作成しました。

現在、以下の2台構成で動作を確認しています。

- PC-A：UNET / Samplingを担当
- PC-B：CLIP / H3 Text Encoder / Video VAEを担当
- PC間では処理結果のみを転送

使用するハードウェア、ComfyUIのバージョン、モデルのバージョン、ネットワーク環境によって結果は異なる場合があります。

---

## 開発のきっかけ

始まりは非常に単純でした。

> **「1台のGPUでは全部をロードするのにVRAMが足りない。」**

巨大なモデルそのものをPC間で移動させるのではなく、それぞれのGPUが得意な処理を担当するように分担できないか、と考えました。

その結果、MiniMax H3を2台のPCへ分散する構成になりました。

```text
        PC-A                         PC-B
┌─────────────────┐          ┌─────────────────┐
│      UNET       │          │      CLIP       │
│    Sampling     │◄────────►│  H3 Text Encoder│
│    Upscaling    │   LAN    │   Video VAE     │
│     Refiner     │          │                 │
└─────────────────┘          └─────────────────┘
```

**Instead of moving the model, move the result.**

**モデルを移動させるのではなく、結果を移動させる。**

---

## 注意事項

本プロジェクトは、MiniMax H3のワークフローを複数PCに分散し、VRAM使用量を抑えることを主な目的として作成しました。

PC間で転送されるデータ量は、大容量のモデルウェイトそのものを転送する場合よりも小さくなることを想定しています。実際の転送量はワークフローやComfyUI側の実装によって異なります。

本プロジェクトは**現状有姿（as-is）**で提供されます。

個人的・実験的な用途を主な目的としており、継続的なメンテナンスやサポートを保証するものではありません。将来のComfyUIバージョンとの互換性も保証されません。

## License

本プロジェクトは **GNU General Public License v3.0 (GPL-3.0)** の下で公開します。

本プロジェクトは、ComfyUIに含まれるMiniMax H3実装（`comfy_extras.nodes_minimax_h3`）に依存しています。ComfyUIおよびその各コンポーネントには、それぞれのライセンスが適用されます。

本プロジェクトにはMiniMax H3のモデルウェイトは含まれておらず、再配布も行いません。MiniMax H3のモデルには、それぞれのライセンスが適用されます。

GPL-3.0の全文は [`LICENSE`](LICENSE) を参照してください。

## 謝辞

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- MiniMax H3および関連するモデル・ソフトウェア
