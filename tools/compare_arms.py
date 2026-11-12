"""对比 256 vs 1024 两臂的层扫描曲线与选层逻辑。"""
import json, sys, numpy as np, torch, argparse
sys.path.insert(0, 'src')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from utils import extract_boxed, last_assistant_text


def load(d, split):
    H = torch.load(f'{d}/{split}_hidden_no_reasoning.pt', map_location='cpu', weights_only=True).float().numpy()
    lab = json.load(open(f'{d}/{split}_labels_no_reasoning.json'))
    outs = json.load(open(f'{d}/{split}_no_tool_outputs.json'))
    items = outs if isinstance(outs, list) else list(outs.values())
    box = {it.get('id'): bool((extract_boxed(last_assistant_text(it)) or '').strip()) for it in items}
    meta = lab['task_meta']
    mask = np.array([box.get(m['id'], False) for m in meta])
    y = np.array([m['tool_necessary'] for m in meta])
    return H, y, meta, mask


def scan(d, filt):
    Htr, ytr, mtr, ktr = load(d, 'train')
    Hte, yte, mte, kte = load(d, 'test')
    if filt:
        Htr, ytr, mtr = Htr[ktr], ytr[ktr], [m for m, k in zip(mtr, ktr) if k]
        Hte, yte, mte = Hte[kte], yte[kte], [m for m, k in zip(mte, kte) if k]
    rows = []
    for L in range(Htr.shape[1]):
        sc = StandardScaler().fit(Htr[:, L, :])
        clf = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Htr[:, L, :]), ytr)
        tr = roc_auc_score(ytr, clf.predict_proba(sc.transform(Htr[:, L, :]))[:, 1])
        te = roc_auc_score(yte, clf.predict_proba(sc.transform(Hte[:, L, :]))[:, 1])
        acc = clf.score(sc.transform(Hte[:, L, :]), yte)
        rows.append(dict(layer=L, train=tr, test=te, acc=acc, n_tr=len(ytr), n_te=len(yte)))
    return rows, yte, mte, None


def strat(rows, d, filt, peak):
    Htr, ytr, _, ktr = load(d, 'train')
    Hte, yte, mte, kte = load(d, 'test')
    if filt:
        Htr, ytr = Htr[ktr], ytr[ktr]
        Hte, yte, mte = Hte[kte], yte[kte], [m for m, k in zip(mte, kte) if k]
    sc = StandardScaler().fit(Htr[:, peak, :])
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Htr[:, peak, :]), ytr)
    p = clf.predict_proba(sc.transform(Hte[:, peak, :]))[:, 1]
    out = {}
    for diff in ['easy', 'medium', 'hard']:
        idx = [i for i, m in enumerate(mte) if m['difficulty'] == diff]
        yy, pp = yte[idx], p[idx]
        a = roc_auc_score(yy, pp) if len(set(yy)) > 1 else float('nan')
        out[diff] = (a, len(idx), int(yy.sum()))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_md', default=None)
    ap.add_argument('--out_png', default=None)
    args = ap.parse_args()

    arms = [('256', 'probe_data/scaled'), ('1024', 'probe_data/scaled1024')]
    lines = []
    curves = {}
    for tag, d in arms:
        for filt in (False, True):
            rows, _, _, _ = scan(d, filt)
            key = f'{tag}{"F" if filt else "U"}'
            curves[key] = rows
            best = max(rows, key=lambda r: r['test'])
            # 仓库逻辑：按 train AUROC 选层（小样本下必饱和）
            repo = max(rows, key=lambda r: (r['train'], -r['layer']))
            st = strat(rows, d, filt, best['layer'])
            name = f'max_new_tokens={tag} / {"parseable-only" if filt else "unfiltered"}'
            lines.append(f'| {name} | {rows[0]["n_tr"]}/{rows[0]["n_te"]} | L{best["layer"]} | **{best["test"]:.3f}** | {best["acc"]:.3f} | L{repo["layer"]} ({repo["test"]:.3f}) |')
            lines.append(f'|  ↳ 分难度 @L{best["layer"]} | easy {st["easy"][0]:.3f} (n={st["easy"][1]},+{st["easy"][2]}) | medium {st["medium"][0]:.3f} (n={st["medium"][1]},+{st["medium"][2]}) | hard {st["hard"][0]:.3f} | | |')

    md = ('| 臂 / 口径 | n (train/test) | 峰值层 | test AUROC | test acc | 仓库 train-acc 选层 |\n'
          '|---|---|---|---|---|---|\n' + '\n'.join(lines))
    print(md)
    if args.out_md:
        open(args.out_md, 'w').write(md)

    if args.out_png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        style = {'256U': ('#9aa5b1', '--', '256 tok · unfiltered'),
                 '256F': ('#9aa5b1', '-', '256 tok · parseable'),
                 '1024U': ('#d1495b', '--', '1024 tok · unfiltered'),
                 '1024F': ('#d1495b', '-', '1024 tok · parseable')}
        for k, rows in curves.items():
            c, ls, lab = style[k]
            ax.plot([r['layer'] for r in rows], [r['test'] for r in rows], ls, color=c, lw=1.8, label=lab)
        ax.axhline(0.89, color='#2e7d32', lw=1, ls=':', alpha=.8)
        ax.axhline(0.96, color='#2e7d32', lw=1, ls=':', alpha=.8)
        ax.text(0.3, 0.893, 'paper 0.89', fontsize=7, color='#2e7d32')
        ax.text(0.3, 0.963, 'paper 0.96', fontsize=7, color='#2e7d32')
        ax.set_xlabel('Transformer layer')
        ax.set_ylabel('Test AUROC')
        ax.set_title('Tool-necessity probe: layer scan, 135 train / 135 test')
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=.22)
        ax.legend(fontsize=7.5, loc='lower right')
        fig.tight_layout()
        fig.savefig(args.out_png)
        print('\n[saved]', args.out_png)
