"""Render matched native and artificial-wait mechanism pairs on one time axis."""

import argparse
from html import escape
import json
from math import ceil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'held', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    pairs = [json.loads(p.read_bytes()) for p in (args.native, args.held)]
    assert pairs[0]['soft_ssd_hold_ms'] == 0 and pairs[1]['soft_ssd_hold_ms'] > 0
    assert len({p['schedule_sha256'] for p in pairs}) == len({p['capacity_bytes'] for p in pairs}) == 1
    assert len({p[a]['audit']['runtime_worker_sha256'] for p in pairs for a in ('shallow', 'deep')}) == 1
    final = max(p[a]['phases'][-1]['native_finish_ms'] for p in pairs for a in ('shallow', 'deep'))
    step = 200
    limit = ceil(final / step) * step

    def x(ms):
        return round(160 + float(ms) / limit * 782, 2)

    def line(a, b, y, fill, width=1):
        return f'<path d="M{x(a)} {y} H{x(b)}" stroke="{fill}" stroke-width="{width}"/>'

    def label(x_, y, value, size=14, **attrs):
        extra = ' '.join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return f'<text x="{x_}" y="{y}" font-size="{size}" {extra}>{escape(value)}</text>'

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 976 649">',
           '<title id="title">R10：额外一次 SSD 路径等待如何改变排序</title>',
           '<desc id="desc">同一输入、同一执行器的无延迟和有延迟配对。每臂有前驱、目标、后继三行；图中展示 SSD 读取、人工等待、上传就绪和实际续算。</desc>',
           '<rect width="976" height="649" fill="white"/>',
           '<g fill="#16253a" font-family="Segoe UI, Microsoft YaHei, Noto Sans CJK SC, sans-serif">',
           label(24, 34, '共享前缀，暂存 82.25 MiB：原生路径与受控 SSD 等待', 22, font_weight='700')]
    for arm_index, (condition, key) in enumerate(((i, a) for i in (0, 1) for a in ('shallow', 'deep'))):
        pair, y0 = pairs[condition], 64 + arm_index * 130
        arm = pair[key]
        tag = '原生搬运' if condition == 0 else f'读盘后人为等待 {pair["soft_ssd_hold_ms"]} ms'
        svg += [f'<rect x="18" y="{y0}" width="940" height="119" rx="5" fill="#f4f7fa"/>',
                label(34, y0 + 24, f'{tag}  ·  目标边界 {arm["actions"][1]:,} token', 15, font_weight='600')]
        for t in range(0, limit + 1, step):
            svg.append(f'<path d="M{x(t)} {y0 + 32} v76" stroke="#d2dce7"/>')
        for i, phase in enumerate(arm['phases']):
            y = y0 + 46 + i * 26
            svg += [label(37, y + 5, ('前驱', '目标', '后继')[i]),
                    line(phase['relative_head_marker_suffix_ms'], phase['native_finish_ms'], y, '#185e9f', 8)]
            for e in arm['event_graph']:
                if e['request'] != i:
                    continue
                t, name = e['relative_head_marker_ms'], e['event']
                if name in ('disk-complete', 'disk-soft-hold'):
                    svg.append(line(t - e['interval_us'] / 1000, t, y - 4, '#4c5261' if name == 'disk-complete' else '#bd580a', 4))
                if name == 'upload-ready':
                    svg.append(f'<path d="M{x(t)} {y-9} v18" stroke="#087b62" stroke-width="2.5"/>')
    for t in range(0, limit + 1, step):
        svg.append(label(x(t), 594, str(t), 12, text_anchor='middle'))
    svg += [label(160, 617, '相对前驱首个计算信号提交的设备事件时间（ms）', 13),
            label(24, 639, '━ 续算', 13, fill='#185e9f'), label(143, 639, '━ SSD 读盘', 13, fill='#4c5261'),
            label(275, 639, '━ 人为等待', 13, fill='#bd580a'), label(407, 639, '│ 上传就绪', 13, fill='#087b62'),
            '</g></svg>']
    with args.out.open('x', encoding='utf8', newline='\n') as stream:
        print('\n'.join(svg), file=stream)


if __name__ == '__main__':
    main()
