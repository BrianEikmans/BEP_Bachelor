"""
generate_graphs.py  --  Training visualisation for report and presentation.

Usage:
    python generate_graphs.py                      # reads logs/training.log
    python generate_graphs.py --log path/to/file
    python generate_graphs.py --out figures/

Produces 12 figures:
    01_score_progression.png       avg/min/max score across curriculum
    02_win_rate.png                win rate per phase with threshold
    03_loss_curve.png              training loss
    04_hand_efficiency.png         hands + discards used
    05_hand_type_heatmap.png       most-played hand type (heatmap)
    06_epsilon_decay.png           exploration schedule
    07_phase_summary_bars.png      final+peak score and win rate bars
    08_learning_speed.png          episodes used vs budget per phase
    09_score_distribution.png      violin plot of score distribution
    10_hand_type_evolution.png     stacked bar of hand type frequency
    11_state_vector_diagram.png    visual state vector breakdown
    12_curriculum_overview.png     one-page summary for presentation
"""

import re, os, argparse
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np

BG       = "#0f1117"
BG_PANEL = "#1a1d27"
GRID     = "#2a2d3a"
TEXT     = "#e8e8f0"
ACCENT   = "#7eb8f7"

PHASE_COLORS = {
    1:"#5b7fa6", 2:"#6aaa8e", 3:"#a0c878", 4:"#e8c55a",
    5:"#f09060", 6:"#e07090", 7:"#c070e0", 8:"#70c0e0", 9:"#f07850",
}
PHASE_LABELS = {
    1:"Ph1\nScore",    2:"Ph2\nBlind 300", 3:"Ph3\nBlind 450",
    4:"Ph4\nDiscards", 5:"Ph5\nBlind 600", 6:"Ph6\n1 Joker",
    7:"Ph7\n2 Jokers", 8:"Ph8\n3 Jokers",  9:"Ph9\n4 Jokers",
}
HAND_ORDER  = ["High Card","One Pair","Two Pair","Three of a Kind",
               "Straight","Flush","Full House","Four of a Kind","Straight Flush"]
HAND_COLORS = ["#5b7fa6","#6aaa8e","#a0c878","#e8c55a","#f09060",
               "#e07090","#c070e0","#70c0e0","#f07850"]
BLIND_MAP   = {2:300,3:450,4:450,5:600,6:700,7:850,8:1000,9:1200}
BUDGETS     = {1:20000,2:5000,3:5000,4:20000,5:25000,6:15000,7:15000,8:20000,9:20000}

def style(fig, axes=None):
    fig.patch.set_facecolor(BG)
    axlist = []
    if axes is not None:
        try:
            axlist = list(np.array(axes).flat)
        except Exception:
            axlist = [axes]
    for ax in axlist:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

def save(fig, name, out_dir):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved {p}")

def smooth(v, w=5):
    a = np.array(v, dtype=float)
    return np.convolve(a, np.ones(w)/w, "same") if len(a) >= w else a

def phase_dividers(ax, groups, label=True):
    phases = sorted(groups)
    for ph in phases[1:]:
        ax.axvline(groups[ph][0]["episode"], color=GRID, lw=1, ls="--", alpha=0.8)
    if label:
        for ph in phases:
            rows = groups[ph]
            mid  = (rows[0]["episode"]+rows[-1]["episode"])/2
            lo, hi = ax.get_ylim()
            ax.text(mid, lo+(hi-lo)*0.96, f"Ph{ph}",
                    color=PHASE_COLORS.get(ph,TEXT), fontsize=7,
                    ha="center", va="top", fontweight="bold")

def parse_log(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(
                r"Ph(\d+) Ep\s+(\d+)\s+\|\s+Score\s+([\d.]+).*?"
                r"\(min\s+(\d+)\s+max\s+(\d+)\s+std\s+([\d.]+)\).*?"
                r"Reward\s+([\d.]+).*?Loss\s+([\d.]+)[~^v].*?"
                r"Hands\s+([\d.]+)/(\d+)", line)
            if not m: continue
            row = dict(
                phase=int(m.group(1)), episode=int(m.group(2)),
                avg_score=float(m.group(3)), min_score=int(m.group(4)),
                max_score=int(m.group(5)), std_score=float(m.group(6)),
                reward=float(m.group(7)), loss=float(m.group(8)),
                hands=float(m.group(9))
            )
            for pat, key in [(r"Win%\s+([\d.]+)","win_rate"),
                              (r"Discards\s+([\d.]+)/","discards"),
                              (r"Eps\s+([\d.]+)","epsilon")]:
                mm = re.search(pat, line)
                row[key] = float(mm.group(1))/(100 if key=="win_rate" else 1) if mm else None
            for pat, key in [(r"Most played:\s+(.+?)\s+\|","most_played"),
                              (r"Best scored:\s+(.+?)(?:\s*$|\r)","best_scored")]:
                mm = re.search(pat, line)
                row[key] = mm.group(1).strip() if mm else None
            rows.append(row)
    return rows

def group_by_phase(rows):
    g = defaultdict(list)
    for r in rows: g[r["phase"]].append(r)
    return dict(g)

# ── 01: Score progression ─────────────────────────────────────────────────────
def fig_score(groups, out_dir):
    fig, ax = plt.subplots(figsize=(14,5)); style(fig, ax)
    drawn = set()
    for ph, rows in sorted(groups.items()):
        col = PHASE_COLORS.get(ph, ACCENT)
        eps = [r["episode"] for r in rows]
        ax.fill_between(eps,[r["min_score"] for r in rows],[r["max_score"] for r in rows],alpha=0.10,color=col)
        ax.plot(eps, smooth([r["avg_score"] for r in rows]), color=col, lw=1.8, label=f"Phase {ph}")
        if ph in BLIND_MAP and BLIND_MAP[ph] not in drawn:
            ax.axhline(BLIND_MAP[ph],color=col,lw=0.8,ls=":",alpha=0.5)
            ax.text(rows[0]["episode"],BLIND_MAP[ph]+15,f"blind {BLIND_MAP[ph]}",color=col,fontsize=7,alpha=0.7)
            drawn.add(BLIND_MAP[ph])
    phase_dividers(ax, groups)
    ax.set_xlabel("Episode",fontsize=9); ax.set_ylabel("Score",fontsize=9)
    ax.set_title("Score Progression Across Curriculum",fontsize=12,fontweight="bold")
    ax.legend(loc="upper left",fontsize=7,facecolor=BG_PANEL,edgecolor=GRID,labelcolor=TEXT,ncol=3)
    fig.tight_layout(); save(fig,"01_score_progression.png",out_dir)

# ── 02: Win rate ──────────────────────────────────────────────────────────────
def fig_winrate(groups, out_dir):
    p2 = {ph:r for ph,r in groups.items() if any(x["win_rate"] for x in r)}
    if not p2: return
    fig, ax = plt.subplots(figsize=(14,4.5)); style(fig, ax)
    for ph, rows in sorted(p2.items()):
        wr = [r for r in rows if r["win_rate"] is not None]
        col = PHASE_COLORS.get(ph,ACCENT)
        y = smooth([r["win_rate"]*100 for r in wr])
        ax.plot([r["episode"] for r in wr], y, color=col, lw=1.8, label=f"Phase {ph}")
        ax.fill_between([r["episode"] for r in wr], 0, y, color=col, alpha=0.06)
    ax.axhline(75, color=TEXT, lw=1, ls="--", alpha=0.4)
    ax.text(list(p2.values())[0][0]["episode"]+100, 76.5, "75% early-stop threshold", color="#aaa", fontsize=7.5)
    ax.set_ylim(0,105)
    ax.set_xlabel("Episode",fontsize=9); ax.set_ylabel("Win Rate %",fontsize=9)
    ax.set_title("Win Rate per Phase",fontsize=12,fontweight="bold")
    phase_dividers(ax, p2)
    ax.legend(loc="lower right",fontsize=7.5,facecolor=BG_PANEL,edgecolor=GRID,labelcolor=TEXT)
    fig.tight_layout(); save(fig,"02_win_rate.png",out_dir)

# ── 03: Loss ──────────────────────────────────────────────────────────────────
def fig_loss(groups, out_dir):
    fig, ax = plt.subplots(figsize=(14,4)); style(fig, ax)
    for ph, rows in sorted(groups.items()):
        ax.plot([r["episode"] for r in rows], smooth([r["loss"] for r in rows],7),
                color=PHASE_COLORS.get(ph,ACCENT), lw=1.4, alpha=0.9)
    phase_dividers(ax, groups)
    ax.set_xlabel("Episode",fontsize=9); ax.set_ylabel("Smooth L1 Loss",fontsize=9)
    ax.set_title("Training Loss Curve",fontsize=12,fontweight="bold")
    fig.tight_layout(); save(fig,"03_loss_curve.png",out_dir)

# ── 04: Hand efficiency ───────────────────────────────────────────────────────
def fig_efficiency(groups, out_dir):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,4.5)); style(fig,[ax1,ax2])
    for ph, rows in sorted(groups.items()):
        col = PHASE_COLORS.get(ph,ACCENT)
        ax1.plot([r["episode"] for r in rows],smooth([r["hands"] for r in rows]),color=col,lw=1.4,label=f"Ph{ph}")
        dr = [r for r in rows if r["discards"] is not None]
        if dr: ax2.plot([r["episode"] for r in dr],smooth([r["discards"] for r in dr]),color=col,lw=1.4,label=f"Ph{ph}")
    for ax,title,ylab,ylim in [
        (ax1,"Hands Used\n(lower = winning faster)","Avg Hands Used / 4",4.5),
        (ax2,"Discards Used\n(phases 4+)","Avg Discards Used / 3",3.5)]:
        ax.set_ylim(0,ylim); ax.set_title(title,fontsize=10,fontweight="bold")
        ax.set_xlabel("Episode",fontsize=9); ax.set_ylabel(ylab,fontsize=9)
        ax.legend(fontsize=7,facecolor=BG_PANEL,edgecolor=GRID,labelcolor=TEXT,ncol=2)
    fig.tight_layout(); save(fig,"04_hand_efficiency.png",out_dir)

# ── 05: Hand type heatmap ─────────────────────────────────────────────────────
def fig_heatmap(groups, out_dir):
    phases = sorted(groups)
    pc = {ph:Counter(r["most_played"] for r in groups[ph] if r["most_played"]) for ph in phases}
    hts = [h for h in HAND_ORDER if any(pc.get(ph,{}).get(h,0)>0 for ph in phases)]
    mat = np.array([[pc.get(ph,{}).get(ht,0)/(sum(pc.get(ph,{}).values()) or 1)*100
                     for ph in phases] for ht in hts])
    fig, ax = plt.subplots(figsize=(11,5)); style(fig, ax)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(phases))); ax.set_xticklabels([PHASE_LABELS.get(p,f"Ph{p}") for p in phases],fontsize=8,color=TEXT)
    ax.set_yticks(range(len(hts))); ax.set_yticklabels(hts,fontsize=8.5,color=TEXT)
    for i in range(len(hts)):
        for j in range(len(phases)):
            v = mat[i,j]
            if v>2: ax.text(j,i,f"{v:.0f}%",ha="center",va="center",fontsize=7,
                            color="black" if v>40 else TEXT,fontweight="bold")
    cb = fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
    cb.set_label("% of log windows",color=TEXT,fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(),color=TEXT)
    cb.ax.yaxis.set_tick_params(color=TEXT,labelsize=7)
    ax.set_title("Most-Played Hand Type by Phase",fontsize=11,fontweight="bold")
    fig.tight_layout(); save(fig,"05_hand_type_heatmap.png",out_dir)

# ── 06: Epsilon decay ─────────────────────────────────────────────────────────
def fig_epsilon(groups, out_dir):
    all_rows = sorted([r for v in groups.values() for r in v], key=lambda r:r["episode"])
    er = [r for r in all_rows if r["epsilon"] is not None]
    if not er: return
    fig, ax = plt.subplots(figsize=(14,3.5)); style(fig, ax)
    ex=[r["episode"] for r in er]; ey=[r["epsilon"] for r in er]
    ax.plot(ex,ey,color=ACCENT,lw=1.6); ax.fill_between(ex,0,ey,alpha=0.15,color=ACCENT)
    for ph,val in {5:0.35,6:0.30,8:0.35}.items():
        if ph in groups:
            x = groups[ph][0]["episode"]
            ax.annotate(f"reset->{val}", xy=(x,val), xytext=(x+300,val+0.07),
                arrowprops=dict(arrowstyle="->",color=PHASE_COLORS.get(ph,TEXT),lw=1.2),
                color=PHASE_COLORS.get(ph,TEXT), fontsize=7.5)
    phase_dividers(ax,groups,label=True)
    ax.set_ylim(0,1.05); ax.set_xlabel("Episode",fontsize=9); ax.set_ylabel("Epsilon",fontsize=9)
    ax.set_title("Exploration Schedule (Epsilon Decay)",fontsize=12,fontweight="bold")
    fig.tight_layout(); save(fig,"06_epsilon_decay.png",out_dir)

# ── 07: Phase summary bars ────────────────────────────────────────────────────
def fig_phase_bars(groups, out_dir):
    phases = sorted(groups)
    fs = [groups[ph][-1]["avg_score"] for ph in phases]
    ps = [max(r["avg_score"] for r in groups[ph]) for ph in phases]
    wr = []
    for ph in phases:
        w = [r["win_rate"] for r in groups[ph] if r["win_rate"] is not None]
        wr.append(max(w)*100 if w else 0)
    x = np.arange(len(phases)); cols = [PHASE_COLORS.get(p,ACCENT) for p in phases]
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5)); style(fig,[ax1,ax2])
    ax1.bar(x-0.2,fs,0.35,color=cols,alpha=0.9,edgecolor=BG,label="Final avg")
    ax1.bar(x+0.2,ps,0.35,color=cols,alpha=0.5,edgecolor=BG,hatch="//",label="Peak avg")
    drawn=set()
    for j,ph in enumerate(phases):
        if ph in BLIND_MAP and BLIND_MAP[ph] not in drawn:
            ax1.plot([j-0.35,j+0.55],[BLIND_MAP[ph]]*2,color=cols[j],lw=1.5,ls=":",alpha=0.7)
            drawn.add(BLIND_MAP[ph])
    ax1.set_xticks(x); ax1.set_xticklabels([f"Ph{p}" for p in phases],color=TEXT,fontsize=8)
    ax1.set_ylabel("Score",fontsize=9)
    ax1.set_title("Final vs Peak Score\n(dotted = blind target)",fontsize=10,fontweight="bold")
    ax1.legend(facecolor=BG_PANEL,edgecolor=GRID,labelcolor=TEXT,fontsize=8)
    bars2 = ax2.bar(x,wr,color=cols,edgecolor=BG,alpha=0.9)
    ax2.axhline(75,color=TEXT,lw=1,ls="--",alpha=0.4)
    for bar,val in zip(bars2,wr):
        if val>0: ax2.text(bar.get_x()+bar.get_width()/2,val+1.5,f"{val:.0f}%",
                           ha="center",va="bottom",color=TEXT,fontsize=8,fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels([f"Ph{p}" for p in phases],color=TEXT,fontsize=8)
    ax2.set_ylim(0,110); ax2.set_ylabel("Peak Win Rate %",fontsize=9)
    ax2.set_title("Peak Win Rate per Phase",fontsize=10,fontweight="bold")
    fig.tight_layout(); save(fig,"07_phase_summary_bars.png",out_dir)

# ── 08: Learning speed ────────────────────────────────────────────────────────
def fig_learning_speed(groups, out_dir):
    phases = sorted(groups)
    eps_used = [groups[ph][-1]["episode"]-groups[ph][0]["episode"]+50 for ph in phases]
    budgets  = [BUDGETS.get(ph,eps_used[i]) for i,ph in enumerate(phases)]
    x = np.arange(len(phases)); cols = [PHASE_COLORS.get(p,ACCENT) for p in phases]
    fig, ax = plt.subplots(figsize=(11,4.5)); style(fig, ax)
    ax.bar(x,[b for b in budgets],color=[c+"44" for c in cols],width=0.6,
           edgecolor=cols,linewidth=1.5,label="Budget")
    bars = ax.bar(x,eps_used,color=cols,width=0.6,alpha=0.85,label="Actual")
    for bar,used,bud in zip(bars,eps_used,budgets):
        pct=used/bud*100
        ax.text(bar.get_x()+bar.get_width()/2,used+200,f"{used:,}\n({pct:.0f}%)",
                ha="center",va="bottom",color=TEXT,fontsize=7.5,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"Ph{p}" for p in phases],color=TEXT,fontsize=9)
    ax.set_ylabel("Episodes",fontsize=9)
    ax.set_title("Episodes Used vs Budget\n(% = fraction consumed; <100% = early stop)",
                 fontsize=10,fontweight="bold")
    ax.legend(facecolor=BG_PANEL,edgecolor=GRID,labelcolor=TEXT,fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"{int(v):,}"))
    fig.tight_layout(); save(fig,"08_learning_speed.png",out_dir)

# ── 09: Score distribution violin ────────────────────────────────────────────
def fig_violin(groups, out_dir):
    phases = sorted(groups)
    all_samples = []
    for ph in phases:
        rows = groups[ph]
        s=[]
        for r in rows: s.extend(np.random.normal(r["avg_score"],max(r["std_score"],1),8).tolist())
        all_samples.append(s)
    fig, ax = plt.subplots(figsize=(13,5)); style(fig, ax)
    parts = ax.violinplot(all_samples, positions=range(len(phases)), showmedians=True, showextrema=True)
    for i,(body,ph) in enumerate(zip(parts["bodies"],phases)):
        body.set_facecolor(PHASE_COLORS.get(ph,ACCENT)); body.set_alpha(0.6)
    parts["cmedians"].set_color(TEXT); parts["cmedians"].set_linewidth(2)
    for k in ["cbars","cmins","cmaxes"]: parts[k].set_color(GRID)
    drawn=set()
    for j,ph in enumerate(phases):
        if ph in BLIND_MAP and BLIND_MAP[ph] not in drawn:
            ax.axhline(BLIND_MAP[ph],color=PHASE_COLORS.get(ph,TEXT),lw=0.8,ls=":",alpha=0.5)
            drawn.add(BLIND_MAP[ph])
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([f"Phase {p}" for p in phases],color=TEXT,fontsize=8)
    ax.set_ylabel("Score",fontsize=9)
    ax.set_title("Score Distribution per Phase\n(width=density, line=median)",fontsize=11,fontweight="bold")
    fig.tight_layout(); save(fig,"09_score_distribution.png",out_dir)

# ── 10: Hand type stacked bars ────────────────────────────────────────────────
def fig_hand_evolution(groups, out_dir):
    phases = sorted(groups)
    pc = {ph:Counter(r["most_played"] for r in groups[ph] if r["most_played"]) for ph in phases}
    data = {ht:np.array([pc.get(ph,{}).get(ht,0)/(sum(pc.get(ph,{}).values()) or 1)*100
                         for ph in phases]) for ht in HAND_ORDER}
    x = np.arange(len(phases))
    fig, ax = plt.subplots(figsize=(13,5)); style(fig, ax)
    bottom = np.zeros(len(phases)); handles=[]
    for ht,col in zip(HAND_ORDER,HAND_COLORS):
        ax.bar(x,data[ht],bottom=bottom,color=col,alpha=0.85,edgecolor=BG,linewidth=0.5)
        handles.append(mpatches.Patch(color=col,label=ht))
        bottom += data[ht]
    ax.set_xticks(x); ax.set_xticklabels([PHASE_LABELS.get(p,f"Ph{p}") for p in phases],color=TEXT,fontsize=8)
    ax.set_ylabel("% of log windows",fontsize=9); ax.set_ylim(0,100)
    ax.set_title("Hand Type Distribution Across Curriculum",fontsize=11,fontweight="bold")
    ax.legend(handles=handles,loc="upper right",fontsize=7,facecolor=BG_PANEL,edgecolor=GRID,labelcolor=TEXT,ncol=2)
    fig.tight_layout(); save(fig,"10_hand_type_evolution.png",out_dir)

# ── 11: State vector diagram ──────────────────────────────────────────────────
def fig_state_diagram(out_dir):
    blocks = [
        ("Per-card\nfeatures",40,"#5b7fa6"),
        ("Deck count\nvector",52,"#6aaa8e"),
        ("Scalars",4,"#a0c878"),
        ("Draw\nfeatures",15,"#e8c55a"),
        ("Score\ncontext",21,"#f09060"),
        ("Joker\nfeatures",30,"#e07090"),
        ("Run\ncontext",5,"#c070e0"),
    ]
    total = sum(b[1] for b in blocks)
    fig, ax = plt.subplots(figsize=(14,3.5)); style(fig, ax)
    ax.set_xlim(0,total); ax.set_ylim(0,1); ax.axis("off")
    x=0
    for name,size,col in blocks:
        ax.add_patch(plt.Rectangle((x,0.1),size,0.7,color=col,alpha=0.85,linewidth=2,edgecolor=BG))
        cx=x+size/2
        ax.text(cx,0.48,name,ha="center",va="center",color="white",fontsize=8.5,fontweight="bold")
        ax.text(cx,0.92,f"{size}",ha="center",va="center",color=col,fontsize=9,fontweight="bold")
        ax.text(cx,0.02,f"[{x}:{x+size}]",ha="center",va="bottom",color="#888",fontsize=6.5)
        x+=size
    ax.text(total/2,1.08,f"State Vector -- {total} floats",
            ha="center",va="center",color=TEXT,fontsize=13,fontweight="bold")
    # Score context sub-breakdown
    sc_x = sum(b[1] for b in blocks[:4])
    sub = [("HT est\nx12",12,"#f09060"),("Needed",1,"#e08050"),("Best\nplay",1,"#d07040"),
           ("Draw\nx3",3,"#c06030"),("Discard\nfeats x2",2,"#b05020"),("Joker\nstrat x2",2,"#a04010")]
    sy=-0.34; sx=sc_x
    for sn,ss,sc in sub:
        ax.add_patch(plt.Rectangle((sx,sy),ss,0.22,color=sc,alpha=0.7,linewidth=1,edgecolor=BG))
        ax.text(sx+ss/2,sy+0.11,sn,ha="center",va="center",color="white",fontsize=5.5,fontweight="bold")
        sx+=ss
    ax.annotate("",xy=(sc_x+21/2,0.1),xytext=(sc_x+21/2,sy+0.22),
                arrowprops=dict(arrowstyle="-",color="#888",lw=1,linestyle="dashed"))
    fig.tight_layout(rect=[0,0.12,1,1])
    save(fig,"11_state_vector_diagram.png",out_dir)

# ── 12: Curriculum overview (presentation) ────────────────────────────────────
def fig_overview(groups, out_dir):
    fig = plt.figure(figsize=(16,9)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.95,"Balatro DQN -- Training Curriculum Overview",
             ha="center",color=TEXT,fontsize=17,fontweight="bold")
    gs = gridspec.GridSpec(2,3,figure=fig,left=0.06,right=0.97,
                           top=0.88,bottom=0.07,hspace=0.48,wspace=0.35)
    def mini_ax(pos):
        ax=fig.add_subplot(pos); ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=TEXT,labelsize=7)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(color=GRID,lw=0.4,alpha=0.5); ax.set_axisbelow(True)
        return ax
    phases = sorted(groups)
    # Score
    ax1=mini_ax(gs[0,0])
    for ph,rows in sorted(groups.items()):
        ax1.plot([r["episode"] for r in rows],smooth([r["avg_score"] for r in rows]),
                 color=PHASE_COLORS.get(ph,ACCENT),lw=1.4)
    ax1.set_title("Score",fontsize=9,fontweight="bold"); ax1.set_xlabel("Episode",fontsize=7)
    # Win rate
    ax2=mini_ax(gs[0,1])
    for ph,rows in sorted(groups.items()):
        wr=[r for r in rows if r["win_rate"] is not None]
        if wr: ax2.plot([r["episode"] for r in wr],smooth([r["win_rate"]*100 for r in wr]),
                        color=PHASE_COLORS.get(ph,ACCENT),lw=1.4)
    ax2.axhline(75,color=TEXT,lw=0.8,ls="--",alpha=0.4); ax2.set_ylim(0,105)
    ax2.set_title("Win Rate %",fontsize=9,fontweight="bold"); ax2.set_xlabel("Episode",fontsize=7)
    # Peak win rate bars
    ax3=mini_ax(gs[0,2]); ax3.grid(axis="y",color=GRID,lw=0.4,alpha=0.5)
    wr_peaks=[max([r["win_rate"] for r in groups[ph] if r["win_rate"] is not None],default=0)*100 for ph in phases]
    ax3.bar(range(len(phases)),wr_peaks,color=[PHASE_COLORS.get(p,ACCENT) for p in phases],alpha=0.85)
    ax3.axhline(75,color=TEXT,lw=0.8,ls="--",alpha=0.4)
    ax3.set_xticks(range(len(phases))); ax3.set_xticklabels([f"Ph{p}" for p in phases],fontsize=7,color=TEXT)
    ax3.set_ylim(0,110); ax3.set_title("Peak Win Rate",fontsize=9,fontweight="bold")
    # Hand type heatmap (full width)
    ax4=mini_ax(gs[1,:]); ax4.grid(False)
    pc={ph:Counter(r["most_played"] for r in groups[ph] if r["most_played"]) for ph in phases}
    hts=[h for h in HAND_ORDER if any(pc.get(ph,{}).get(h,0)>0 for ph in phases)]
    mat=np.array([[pc.get(ph,{}).get(ht,0)/(sum(pc.get(ph,{}).values()) or 1)*100
                   for ph in phases] for ht in hts])
    im=ax4.imshow(mat,aspect="auto",cmap="YlOrRd",vmin=0,vmax=mat.max())
    ax4.set_xticks(range(len(phases))); ax4.set_xticklabels([PHASE_LABELS.get(p,f"Ph{p}") for p in phases],fontsize=8,color=TEXT)
    ax4.set_yticks(range(len(hts))); ax4.set_yticklabels(hts,fontsize=8,color=TEXT)
    for i in range(len(hts)):
        for j in range(len(phases)):
            v=mat[i,j]
            if v>3: ax4.text(j,i,f"{v:.0f}%",ha="center",va="center",
                             fontsize=7,color="black" if v>50 else TEXT)
    ax4.set_title("Most-Played Hand Type by Phase",fontsize=9,fontweight="bold")
    cb=fig.colorbar(im,ax=ax4,fraction=0.015,pad=0.01)
    cb.ax.yaxis.set_tick_params(color=TEXT,labelsize=6); plt.setp(cb.ax.yaxis.get_ticklabels(),color=TEXT)
    save(fig,"12_curriculum_overview.png",out_dir)

# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(groups):
    print(); print("="*75); print(f"{'TRAINING SUMMARY':^75}"); print("="*75)
    print(f"  {'Ph':<4}{'Eps':>8}{'Score end':>10}{'Peak':>9}{'Win% end':>9}{'Win% peak':>10}{'Conv?':>7}")
    print("-"*75)
    for ph in sorted(groups):
        rows=groups[ph]; n=rows[-1]["episode"]-rows[0]["episode"]+50
        budget=BUDGETS.get(ph,n); se=rows[-1]["avg_score"]; sp=max(r["avg_score"] for r in rows)
        wr=[r["win_rate"] for r in rows if r["win_rate"] is not None]
        we=f"{wr[-1]*100:.0f}%" if wr else "--"; wp=f"{max(wr)*100:.0f}%" if wr else "--"
        conv="YES" if n<budget*0.95 else " no"
        print(f"  {ph:<4}{n:>8,}{se:>10.0f}{sp:>9.0f}{str(we):>9}{str(wp):>10}{conv:>7}")
    print("="*75)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate training graphs")
    parser.add_argument("--log", default="logs/training.log")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    if not os.path.exists(args.log):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs","training.log")
        if os.path.exists(alt): args.log=alt
        else: print(f"Log not found: {args.log}"); return
    os.makedirs(args.out, exist_ok=True)
    print(f"Parsing {args.log} ...")
    rows=parse_log(args.log); groups=group_by_phase(rows)
    print(f"Parsed {len(rows)} data points, phases {sorted(groups)}")
    print_summary(groups)
    print("Generating 12 figures...")
    fig_score(groups,args.out); fig_winrate(groups,args.out); fig_loss(groups,args.out)
    fig_efficiency(groups,args.out); fig_heatmap(groups,args.out); fig_epsilon(groups,args.out)
    fig_phase_bars(groups,args.out); fig_learning_speed(groups,args.out)
    fig_violin(groups,args.out); fig_hand_evolution(groups,args.out)
    fig_state_diagram(args.out); fig_overview(groups,args.out)
    print(f"\nAll 12 figures saved to {args.out}/")

if __name__ == "__main__":
    main()
