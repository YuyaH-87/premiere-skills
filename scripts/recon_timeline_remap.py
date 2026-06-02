#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
再構成WAVベースSRTの時刻を、Premiereタイムライン時刻へ per-clip 変換するツール。

背景（2026-05-31 アルトラ登山靴レビュー）:
  WAV不在のため A1 トラックの各クリップ [in,out] を素材mp4から切り出して連結し
  「再構成WAV」を作って Whisper にかけた。だが再構成WAVの時間軸は
  「素材の実fps(≈23.81/23.84, VFR)」で、Premiereのタイムラインは「シーケンスfps(23.0)」。
  16805f は 23.0fps で 730.65秒だが、再構成WAVは素材実時間で705秒。
  そのため再構成WAV秒をそのままSRTにすると一律約3.6%圧縮され、テロップが進むほど早く出て
  末尾約25秒が無字幕になる。

  本ツールは各クリップで「再構成WAVサンプル位置 ↔ タイムラインframe」の区分線形写像を作り、
  SRTの各時刻を timeline秒 = timeline_frame / SEQ_FPS に変換する。
  VFRの実サンプル数ベースなので素材境界も厳密に合う。

使い方:
  python recon_timeline_remap.py <調整後.xml> <入力SRT(再構成WAV時刻)> <出力SRT>

注意:
  SAMP は再構成WAV作成時に ffmpeg で抽出した 16kHz モノラル音声のサンプル数。
  クランプ（min/max）も作成時と同一にしてあり、Whisperが見た再構成WAVと
  サンプル位置が一致する。素材を再抽出した場合は SAMP を更新すること。
"""
import sys
import re
import xml.etree.ElementTree as ET

SEQ_FPS = 23.0          # シーケンスfps（XML timebase=23 ntsc=FALSE → 16805f=730.65s=12:10 と一致）
SR = 16000              # 再構成WAVのサンプルレート

# 再構成WAV作成時の素材抽出音声サンプル数（16kHz mono, pcm_s16le）
#   src1.wav=4860664 bytes, src2.wav=31966628 bytes → (bytes-44)//2
SAMP = {
    "file-130": (4860664 - 44) // 2,   # 2430310  (撮影素材1.mp4)
    "file-131": (31966628 - 44) // 2,  # 15983292 (撮影素材2.mp4)
}
# XML <file><duration>（フレーム）
FDUR = {"file-130": 3617, "file-131": 23814}


def build_clip_map(xml_path):
    root = ET.parse(xml_path).getroot()
    seq = root.find(".//sequence")
    seqdur = int(seq.findtext("duration"))
    a1 = seq.find("./media/audio").findall("track")[0]
    clips = []
    cum = 0  # 再構成WAV上の累積サンプル位置
    for c in a1.findall("clipitem"):
        s = int(c.findtext("start"))
        e = int(c.findtext("end"))
        i = int(c.findtext("in"))
        o = int(c.findtext("out"))
        fid = c.find("file").get("id")
        n = SAMP[fid]
        fd = FDUR[fid]
        s_smp = round(i / fd * n)
        e_smp = round(o / fd * n)
        # 作成時と同一のクランプ
        s_smp = max(0, min(s_smp, n))
        e_smp = max(s_smp, min(e_smp, n))
        seg = e_smp - s_smp
        clips.append(
            dict(r0=cum, r1=cum + seg, tl0=s, tl1=e, fid=fid)
        )
        cum += seg
    return clips, cum, seqdur


def make_recon_to_tlframe(clips, total_samples, seqdur):
    def recon_sec_to_tl_sec(sec):
        rs = sec * SR
        if rs >= total_samples:
            # 単語列を超えた末尾補完など → シーケンス終端へクランプ
            return seqdur / SEQ_FPS
        # 該当クリップ探索（線形でも337件なら十分高速だが二分探索）
        lo, hi = 0, len(clips) - 1
        idx = len(clips) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            c = clips[mid]
            if rs < c["r0"]:
                hi = mid - 1
            elif rs >= c["r1"]:
                lo = mid + 1
            else:
                idx = mid
                break
        c = clips[idx]
        seg = c["r1"] - c["r0"]
        frac = (rs - c["r0"]) / seg if seg > 0 else 0.0
        tlf = c["tl0"] + frac * (c["tl1"] - c["tl0"])
        return tlf / SEQ_FPS

    return recon_sec_to_tl_sec


TIME_RE = re.compile(
    r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)"
)


def parse_time(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def fmt_time(sec):
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def remap_srt(in_srt, out_srt, conv, seqdur):
    with open(in_srt, encoding="utf-8-sig") as f:
        text = f.read()
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    end_limit = seqdur / SEQ_FPS
    MIN_DUR = 0.6  # 最小表示秒
    EOF_BUF = 1.0  # 最終行のみシーケンス終端を超えて確保してよい余裕
    items = []  # (nst, nen, body, orig_st)
    for b in blocks:
        lines = b.splitlines()
        if len(lines) < 2:
            continue
        m = TIME_RE.search(lines[1])
        if not m:
            continue
        st = parse_time(*m.group(1, 2, 3, 4))
        en = parse_time(*m.group(5, 6, 7, 8))
        nst = conv(st)
        nen = conv(en)
        if nen > end_limit:
            nen = end_limit
        body = "\n".join(lines[2:])
        items.append([nst, nen, body, st])
    # 最小表示時間の確保（次エントリ開始を侵食しない／最終行のみ終端超過可）
    n = len(items)
    for i, it in enumerate(items):
        nst, nen = it[0], it[1]
        if nen - nst < MIN_DUR:
            want = nst + MIN_DUR
            if i < n - 1:
                nen = min(want, items[i + 1][0])  # 次行を侵食しない
            else:
                nen = min(want, end_limit + EOF_BUF)  # 最終行は終端+余裕まで
            it[1] = max(nen, nst + 0.001)
    out_lines = []
    max_drift = 0.0
    for idx, (nst, nen, body, st) in enumerate(items, 1):
        max_drift = max(max_drift, abs(nst - st))
        out_lines.append(f"{idx}\n{fmt_time(nst)} --> {fmt_time(nen)}\n{body}")
    idx = len(items)
    # Premiere 日本語版: UTF-8 BOM + CRLF
    with open(out_srt, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n\n".join(out_lines) + "\n\n")
    return idx, max_drift


def main():
    xml_path, in_srt, out_srt = sys.argv[1], sys.argv[2], sys.argv[3]
    clips, total, seqdur = build_clip_map(xml_path)
    print(f"クリップ数: {len(clips)}  再構成総サンプル: {total} (= {total/SR:.3f}s)")
    print(f"シーケンス: {seqdur}f = {seqdur/SEQ_FPS:.3f}s @ {SEQ_FPS}fps")
    conv = make_recon_to_tlframe(clips, total, seqdur)
    # 検算
    for chk in (0.0, 50.23, 56.99, 350.0, 680.29, total / SR):
        print(f"  recon {chk:7.2f}s -> timeline {conv(chk):7.2f}s")
    n, md = remap_srt(in_srt, out_srt, conv, seqdur)
    print(f"\n変換完了: {n} エントリ  最大シフト: {md:.2f}s")
    print(f"出力: {out_srt}")


if __name__ == "__main__":
    main()
