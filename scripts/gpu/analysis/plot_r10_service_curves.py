"""Render service holdout curves using each arm's actual offered load and its own source block."""

import argparse
from collections import defaultdict
from html import escape
from itertools import product
import json
from math import isfinite
from pathlib import Path


POLICIES = {'local-cost': ('#1b62a5', ''), 'entry-replay': ('#b84c15', '7 4')}
GROUPS = tuple(product(('oasst', 'bailian'), ('short', 'mixed')))
ANCHORS = ('low', 'shoulder', 'overload')
METRICS = (('p95_visible_ttft_ms', '可见文本 TTFT P95（ms）'),
           ('p95_frame_tpot_ms', '解码帧 TPOT P95（ms）'),
           ('completed_per_s', '完成吞吐（请求/s）'),
           ('goodput_per_s', 'SLO 有效吞吐（请求/s）'))


def series(report, family, profile, metric):
    points = defaultdict(dict)
    totals = defaultdict(int)
    for anchor in ANCHORS:
        cell = report['curves'][f'{family}/{profile}/{anchor}']
        for k in ('registered_pairs', 'comparable_pairs', 'failed_pairs', 'different_output_pairs'):
            totals[k] += cell[k]
        block_sets = [{r['block'] for r in cell['policy_points'][p]} for p in POLICIES]
        if block_sets[0] != block_sets[1]:
            raise ValueError('two policies describe different sets of comparable blocks')
        for policy in POLICIES:
            rows = cell['policy_points'][policy]
            if len({r['block'] for r in rows}) != len(rows) or len(rows) != cell['comparable_pairs']:
                raise ValueError('one comparable point per policy and source block is required')
            for row in rows:
                x, y = row['offered_per_s'], row[metric]
                if not isfinite(x) or x <= 0 or (y is not None and (not isfinite(y) or y < 0)):
                    raise ValueError('invalid measured arrival rate or metric')
                if y is not None:
                    points[row['block'], policy][anchor] = x, y
    return points, totals


def render(report):
    if set(report['curves']) != {'/'.join(g + (a,)) for g, a in product(GROUPS, ANCHORS)} or len(report['pairs']) != 24:
        raise ValueError('requires the complete, audited 24-pair service summary')
    width, height = 1296, 150 + 240 * len(METRICS)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">R10 本机单槽服务留出曲线</title>',
           '<desc id="desc">按真实提交率绘制各源块的可见文本 TTFT、解码帧 TPOT、完成吞吐和有效吞吐。断线表示缺少可比观测。</desc>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           '<g font-family="Segoe UI, Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#1c2834">']

    def txt(x, y, value, size=14, extra=''):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(value))}</text>')

    txt(25, 33, 'R10 本机单槽服务留出曲线', 23, 'font-weight="700"')
    txt(25, 62, '横轴：客户端实测提交率（请求/s）。同线为同一个输入块；缺测与不可比点不连线。')
    for i, (policy, (color, dash)) in enumerate(POLICIES.items()):
        x = 790 + 230 * i
        svg.append(f'<line x1="{x}" y1="27" x2="{x + 28}" y2="27" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
        txt(x + 35, 33, policy)
    for col, (family, profile) in enumerate(GROUPS):
        col0 = col * 320 + 14
        txt(col0 + 14, 95, ('OASST' if family == 'oasst' else '百炼') + ('／短输出' if profile == 'short' else '／混合输出'), 17)
        for row, (metric, title) in enumerate(METRICS):
            paths, counts = series(report, family, profile, metric)
            x0, y0, pw, ph = col0 + 52, 131 + 240 * row, 239, 154
            flat = [p for path in paths.values() for p in path.values()]
            xr = max((x for x, _ in flat), default=1.) * 1.05
            yr = max(1., max((y for _, y in flat), default=1.) * 1.12)
            position = lambda p: (x0 + p[0] * pw / xr, y0 + ph - p[1] * ph / yr)
            txt(col0 + 14, y0 - 10, title)
            for step in (0, .5, 1):
                x, y = x0 + pw * step, y0 + ph * (1 - step)
                svg.append(f'<path d="M{x0} {y} h{pw} M{x} {y0} v{ph}" stroke="#dfe6ee"/>')
                txt(x0 - 6, y + 4, f'{yr * step:.3g}', 10, 'text-anchor="end"')
                txt(x, y0 + ph + 16, f'{xr * step:.3g}', 10, 'text-anchor="middle"')
            if not flat:
                txt(x0 + 15, y0 + 32, '没有可比观测')
            for (block, policy), locations in sorted(paths.items()):
                color, dash = POLICIES[policy]
                for a, b in zip(ANCHORS, ANCHORS[1:]):
                    if a in locations and b in locations:
                        (xa, ya), (xb, yb) = [position(locations[k]) for k in (a, b)]
                        svg.append(f'<path d="M{xa:.2f} {ya:.2f} L{xb:.2f} {yb:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}"/>')
                for anchor, p in locations.items():
                    x, y = position(p)
                    svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>'
                               f'{escape(block)} / {policy} / {anchor}: {p[0]:.4g}, {p[1]:.4g}</title></circle>')
            txt(col0 + 14, y0 + ph + 37,
                f"可比 {counts['comparable_pairs']}/{counts['registered_pairs']} 对；失败 {counts['failed_pairs']}，输出不一致 {counts['different_output_pairs']}", 11)
    txt(25, height - 13, '有限开环回放；各面板刻度按该组数据设置。完整失败记录和单请求尾部见试验汇总。', 12)
    return '\n'.join(svg + ['</g></svg>']) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    graphic = render(json.loads(a.summary.read_bytes()))
    with a.out.open('x', encoding='utf-8', newline='\n') as f:
        f.write(graphic)


if __name__ == '__main__':
    main()
