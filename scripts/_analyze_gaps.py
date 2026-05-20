"""大ギャップ箇所の word-level 検証 + 各案の効果シミュレーション"""
import json, sys, statistics
sys.path.insert(0, 'scripts')
from whisper_to_srt import (
    build_src_to_tl_map, apply_corrections, _normalize_for_match,
)

with open('output/srt/test_sozai_20260520/test_sozai_20260520.segments.json', encoding='utf-8') as f:
    seg_list = json.load(f)
with open('output/srt/test_sozai_20260520/test_sozai_20260520.lines.txt', encoding='utf-8') as f:
    raw_text = f.read()

words = []
for seg in seg_list:
    sw = seg.get('words', [])
    if not sw:
        text = apply_corrections(seg['text'])
        if text:
            words.append({'word': text, 'start': seg['start'], 'end': seg['end']})
        continue
    rc = ''.join(w['word'] for w in sw)
    co = apply_corrections(rc)
    if rc == co:
        for w in sw:
            if w['word']:
                words.append({'word': w['word'], 'start': w['start'], 'end': w['end']})
    else:
        ss = sw[0]['start']; se = sw[-1]['end']; dur = max(se - ss, 0.01); n = len(co)
        for i, c in enumerate(co):
            words.append({'word': c, 'start': ss + dur*i/n, 'end': ss + dur*(i+1)/n})

char_to_word = []
for widx, w in enumerate(words):
    for _ in _normalize_for_match(w['word']):
        char_to_word.append(widx)
total_chars = len(char_to_word)
whisper_norm = ''.join(_normalize_for_match(w['word']) for w in words)
lines = [ln.strip() for ln in raw_text.split('\n') if ln.strip()]
lines_total = sum(len(_normalize_for_match(apply_corrections(l))) for l in lines)
ratio = total_chars / lines_total

entries_src = []
pos = 0; min_pos = 0; cum = 0
for line in lines:
    ln_norm = _normalize_for_match(apply_corrections(line))
    L = len(ln_norm)
    if L == 0: continue
    if pos >= total_chars:
        cum += L; continue
    ideal = int(cum * ratio)
    exp = max(pos, min(ideal, total_chars - 1))
    if L >= 6:
        idx = whisper_norm.find(ln_norm[:6], max(min_pos, exp-300), min(total_chars, exp+300+6))
    else:
        idx = -1
    if idx >= 0 and idx != pos:
        pos = idx
    elif idx < 0 and ideal > pos + 300:
        pos = min(ideal, total_chars - 1)
    end_pos = min(pos + L, total_chars) - 1
    sw_i = char_to_word[pos]; ew = char_to_word[end_pos]
    entries_src.append((words[sw_i]['start'], words[ew]['end'], apply_corrections(line), sw_i, ew))
    pos += L; min_pos = pos; cum += L

clip_map, fps, tl_dur = build_src_to_tl_map('output/cut/test_sozai_20260520_カット済み.xml')

def find_clip(t):
    for i, (si, so, _, _) in enumerate(clip_map):
        if si <= t < so: return i
    return -1

entries_tl = []
for ss, se, t, sw_i, ew in entries_src:
    ci = find_clip(ss)
    if ci < 0: continue
    si, so, ts, te = clip_map[ci]
    tl_s = ts + (ss - si)
    eci = find_clip(se)
    if eci < 0 or eci != ci:
        tl_e = te - 1.0/fps
    else:
        es_si, _, es_ts, _ = clip_map[eci]
        tl_e = es_ts + (se - es_si)
    entries_tl.append({'tl_s': tl_s, 'tl_e': tl_e, 'src_s': ss, 'src_e': se, 'sw': sw_i, 'ew': ew, 't': t})

# 大ギャップ TOP 20 word-level 分析
big_gaps_data = []
for i in range(len(entries_tl) - 1):
    cur = entries_tl[i]; nxt = entries_tl[i+1]
    e_orig = cur['tl_e']
    e_new = max(e_orig, min(nxt['tl_s'] - 0.080, e_orig + 2.0))
    tl_gap = nxt['tl_s'] - e_new
    src_gap = nxt['src_s'] - cur['src_e']
    inter_words = words[cur['ew']+1 : nxt['sw']]
    max_word_gap = 0.0
    if inter_words:
        prev_end = cur['src_e']
        for w in inter_words:
            g = w['start'] - prev_end
            if g > max_word_gap: max_word_gap = g
            prev_end = w['end']
        g_tail = nxt['src_s'] - prev_end
        if g_tail > max_word_gap: max_word_gap = g_tail
    else:
        max_word_gap = src_gap
    big_gaps_data.append({
        'idx': i, 'tl_gap': tl_gap, 'src_gap': src_gap,
        'n_inter': len(inter_words), 'max_word_gap': max_word_gap,
        'cur_text': cur['t'][:25], 'nxt_text': nxt['t'][:25],
    })

big = sorted([g for g in big_gaps_data if g['tl_gap'] >= 2.0], key=lambda x: -x['tl_gap'])[:20]
print('=== 2s超ギャップ TOP 20 (word-level) ===')
print(f'{"idx":>3} {"tl_gap":>7} {"src_gap":>7} {"nW":>3} {"maxWgap":>8}  cur >> nxt')
for g in big:
    print(f"{g['idx']:>3} {g['tl_gap']:>6.2f}s {g['src_gap']:>6.2f}s {g['n_inter']:>3} {g['max_word_gap']:>7.2f}s  {g['cur_text']} >> {g['nxt_text']}")

print()
print('=== 各案のシミュレーション ===')

def simulate(label, calc_e_new):
    new_gaps = []
    durs = []
    for i in range(len(entries_tl)):
        cur = entries_tl[i]
        if i+1 < len(entries_tl):
            nxt = entries_tl[i+1]
            e_new = calc_e_new(cur, nxt, i)
            new_gaps.append(max(0, nxt['tl_s'] - e_new))
        else:
            e_new = cur['tl_e'] + 2.0
        durs.append(max(1/fps, e_new - cur['tl_s']))
    buckets = [(0, 0.1, '0-100ms'), (0.1, 0.5, '100-500ms'), (0.5, 1.0, '500ms-1s'),
               (1.0, 2.0, '1-2s'), (2.0, 3.0, '2-3s'), (3.0, 5.0, '3-5s'), (5.0, 999, '5s+')]
    print(f'\n[{label}]')
    for lo, hi, lab in buckets:
        c = sum(1 for g in new_gaps if lo <= g < hi)
        bar = '#' * (c // 4)
        print(f'  {lab:>10}: {c:>3}  {bar}')
    print(f'  gap 中央値={statistics.median(new_gaps)*1000:.0f}ms, 平均={statistics.mean(new_gaps):.2f}s')
    print(f'  dur 中央値={statistics.median(durs):.2f}s, 平均={statistics.mean(durs):.2f}s, 最大={max(durs):.2f}s')

simulate('現状 max_extend=2.0s', lambda c, n, i: max(c['tl_e'], min(n['tl_s']-0.080, c['tl_e']+2.0)))
simulate('案A: max_extend=4.0s', lambda c, n, i: max(c['tl_e'], min(n['tl_s']-0.080, c['tl_e']+4.0)))
simulate('案A2: max_extend=8.0s', lambda c, n, i: max(c['tl_e'], min(n['tl_s']-0.080, c['tl_e']+8.0)))

def case_c(c, n, i):
    gap = n['tl_s'] - c['tl_e']
    if gap <= 5.0:
        return max(c['tl_e'], n['tl_s'] - 0.080)
    return max(c['tl_e'], min(n['tl_s'] - 0.080, c['tl_e'] + 2.0))
simulate('案C: gap<=5s全接続/超は2s打切', case_c)

def case_d(c, n, i, thr=1.5):
    inter = words[c['ew']+1 : n['sw']]
    if not inter:
        return max(c['tl_e'], n['tl_s'] - 0.080)
    prev_end = c['src_e']
    biggest = 0.0; biggest_at_src = c['src_e']
    for w in inter:
        g = w['start'] - prev_end
        if g > biggest:
            biggest = g; biggest_at_src = prev_end
        prev_end = w['end']
    tail = n['src_s'] - prev_end
    if tail > biggest:
        biggest = tail; biggest_at_src = prev_end
    if biggest <= thr:
        return max(c['tl_e'], n['tl_s'] - 0.080)
    tl_cut = c['tl_e'] + (biggest_at_src - c['src_e']) + 0.3
    return max(c['tl_e'], min(n['tl_s'] - 0.080, tl_cut))

simulate('案D: word-level 1.5s 無音検出', lambda c, n, i: case_d(c, n, i, 1.5))
simulate('案E: word-level 2.0s 無音検出', lambda c, n, i: case_d(c, n, i, 2.0))
simulate('案F: word-level 1.0s 無音検出', lambda c, n, i: case_d(c, n, i, 1.0))
