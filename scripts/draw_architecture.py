"""Draw architecture diagrams for both Causal and Bidirectional variants."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Colors — muted academic palette
C_QUERY      = "#AED6F1"
C_PASSAGE    = "#F5CBA7"
C_EOS        = "#A9DFBF"
C_HIDDEN     = "#FADBD8"
C_HIDDEN_EOS = "#6DBE8B"
C_CLASSIFIER = "#D2B4DE"
C_LABEL      = "#F1948A"
C_BACKBONE   = "#F8F9F9"
C_ATTN       = "#5C6BC0"

BOX_H = 0.6
BOX_W = 0.9
ROUND = 0.08

# Token layout (shared by both diagrams)
TOKEN_INFO = [
    ("Q₁", C_QUERY, False),
    ("Q₂", C_QUERY, False),
    ("···", None, False),
    ("P₁", C_PASSAGE, False),
    ("···", None, False),
    ("⟨eos⟩", C_EOS, True),
    ("P₂", C_PASSAGE, False),
    ("···", None, False),
    ("⟨eos⟩", C_EOS, True),
    ("···", None, False),
    ("Pₙ", C_PASSAGE, False),
    ("···", None, False),
    ("⟨eos⟩", C_EOS, True),
]

N = len(TOKEN_INFO)
XS = np.linspace(0.5, 13.5, N)
Y_INPUT = 0.3
Y_EMBED = 1.7
Y_HIDDEN = 4.5
Y_CLASSIFIER = 6.0
Y_LABEL = 7.3

EOS_INDICES = [i for i, (_, _, is_eos) in enumerate(TOKEN_INFO) if is_eos]
REAL_INDICES = [i for i, (_, color, _) in enumerate(TOKEN_INFO) if color is not None]


def draw_box(ax, x, y, text, color, fontsize=8.5, bold=False, w=BOX_W, h=BOX_H):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={ROUND}",
        facecolor=color, edgecolor="#555555", linewidth=1.0,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight,
            fontfamily="monospace" if text.startswith(("h", "E")) else "sans-serif")


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def draw_bracket(ax, x_start, x_end, y, label, color):
    mid = (x_start + x_end) / 2
    ax.plot([x_start, x_start, x_end, x_end], [y, y - 0.15, y - 0.15, y],
            color=color, lw=1.5, solid_capstyle="round")
    ax.text(mid, y - 0.45, label, ha="center", va="center",
            fontsize=9, color=color, weight="bold")


def draw_common(ax):
    """Draw input tokens, embeddings, hidden states, classifier, and labels."""
    # Input tokens
    for i, (tok, color, is_eos) in enumerate(TOKEN_INFO):
        if color is None:
            ax.text(XS[i], Y_INPUT, tok, ha="center", va="center", fontsize=16, color="#888")
        else:
            draw_box(ax, XS[i], Y_INPUT, tok, color, fontsize=9)

    # Embeddings
    for i, (tok, color, is_eos) in enumerate(TOKEN_INFO):
        if color is None:
            ax.text(XS[i], Y_EMBED, tok, ha="center", va="center", fontsize=16, color="#888")
        else:
            draw_box(ax, XS[i], Y_EMBED,
                     f"E({tok})" if not is_eos else "E(eos)", color, fontsize=7)
            draw_arrow(ax, XS[i], Y_INPUT + BOX_H / 2 + 0.02,
                       XS[i], Y_EMBED - BOX_H / 2 - 0.02, color="#AAAAAA", lw=0.8)

    # Hidden states
    for i, (tok, color, is_eos) in enumerate(TOKEN_INFO):
        if color is None:
            ax.text(XS[i], Y_HIDDEN, tok, ha="center", va="center", fontsize=16, color="#888")
        else:
            if is_eos:
                eos_num = EOS_INDICES.index(i) + 1
                label = f"h_eos{eos_num}" if eos_num <= 2 else "h_eosN"
                draw_box(ax, XS[i], Y_HIDDEN, label, C_HIDDEN_EOS, fontsize=7.5, bold=True)
            else:
                draw_box(ax, XS[i], Y_HIDDEN, f"h{i + 1}", C_HIDDEN, fontsize=7.5)

    # Classifier + labels
    classifier_labels = ["W·h₁+b", "W·h₂+b", "W·hₙ+b"]
    label_texts = ["ŷ₁", "ŷ₂", "ŷₙ"]
    for idx, eos_idx in enumerate(EOS_INDICES):
        draw_arrow(ax, XS[eos_idx], Y_HIDDEN + BOX_H / 2 + 0.02,
                   XS[eos_idx], Y_CLASSIFIER - BOX_H / 2 - 0.02, color="#2E7D32", lw=2.0)
        draw_box(ax, XS[eos_idx], Y_CLASSIFIER, classifier_labels[idx],
                 C_CLASSIFIER, fontsize=8.5, w=1.1)
        draw_arrow(ax, XS[eos_idx], Y_CLASSIFIER + BOX_H / 2 + 0.02,
                   XS[eos_idx], Y_LABEL - BOX_H / 2 - 0.02, color="#C62828", lw=2.0)
        mid_y = (Y_CLASSIFIER + Y_LABEL) / 2
        ax.text(XS[eos_idx] + 0.65, mid_y, "σ(·)",
                ha="center", va="center", fontsize=10, color="#C62828",
                style="italic", fontfamily="serif")
        draw_box(ax, XS[eos_idx], Y_LABEL, label_texts[idx], C_LABEL, fontsize=11, bold=True)

    # Brackets
    draw_bracket(ax, XS[0] - 0.45, XS[1] + 0.45, Y_INPUT - 0.5, "Query", "#1565C0")
    draw_bracket(ax, XS[3] - 0.45, XS[5] + 0.45, Y_INPUT - 0.5, "Passage 1", "#BF360C")
    draw_bracket(ax, XS[6] - 0.45, XS[8] + 0.45, Y_INPUT - 0.5, "Passage 2", "#BF360C")
    draw_bracket(ax, XS[10] - 0.45, XS[12] + 0.45, Y_INPUT - 0.5, "Passage N", "#BF360C")


def draw_causal_attention(ax):
    """Draw representative causal attention arrows (uniform sampling)."""
    # Pick a few representative (source, target) pairs across the sequence
    # to show the causal pattern without cluttering
    # Each arrow: target attends to source (arrow from target to source)
    pairs = [
        # Within query
        (1, 0),
        # Passage 1 tokens attend to query
        (3, 0), (3, 1),
        # EOS1 attends to query + passage 1
        (5, 0), (5, 3),
        # Passage 2 attends to earlier tokens
        (6, 1), (6, 5),
        # EOS2 attends across
        (8, 0), (8, 5), (8, 6),
        # Passage N attends to earlier
        (10, 5), (10, 8),
        # EOSN attends across all
        (12, 0), (12, 5), (12, 8), (12, 10),
    ]
    for tgt, src in pairs:
        dist = abs(tgt - src)
        rad = 0.12 + dist * 0.012
        ax.annotate(
            "", xy=(XS[src], Y_EMBED + BOX_H / 2 + 0.03),
            xytext=(XS[tgt], Y_EMBED + BOX_H / 2 + 0.03),
            arrowprops=dict(
                arrowstyle="-|>", color=C_ATTN, lw=0.8, alpha=0.45,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )

    # Legend
    ax.annotate("", xy=(0.3, Y_EMBED + BOX_H / 2 + 1.6),
                xytext=(1.8, Y_EMBED + BOX_H / 2 + 1.6),
                arrowprops=dict(arrowstyle="-|>", color=C_ATTN, lw=1.2,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(2.7, Y_EMBED + BOX_H / 2 + 1.6, "attends to (causal)",
            fontsize=8, color=C_ATTN, va="center")


def draw_bidirectional_attention(ax):
    """Draw representative bidirectional attention arrows (uniform sampling)."""
    # Pairs showing both forward AND backward attention
    pairs = [
        # Forward (like causal)
        (1, 0),
        (3, 1),
        (5, 0), (5, 3),
        (8, 5), (8, 6),
        (12, 8), (12, 10),
        # Backward (unique to bidirectional — later tokens attending to earlier is same,
        # but earlier tokens attending to later is new)
        (0, 3),
        (1, 5),
        (3, 8),
        (5, 10), (5, 12),
        (6, 12),
        (8, 12),
    ]
    for tgt, src in pairs:
        dist = abs(tgt - src)
        rad = 0.12 + dist * 0.012
        # Always curve upward: positive rad when src is to the right, negative when left
        if src > tgt:
            arc_rad = -rad
        else:
            arc_rad = rad
        ax.annotate(
            "", xy=(XS[src], Y_EMBED + BOX_H / 2 + 0.03),
            xytext=(XS[tgt], Y_EMBED + BOX_H / 2 + 0.03),
            arrowprops=dict(
                arrowstyle="-|>", color=C_ATTN, lw=0.8, alpha=0.45,
                connectionstyle=f"arc3,rad={arc_rad}",
            ),
        )

    # Legend (double-headed arrow)
    ax.annotate("", xy=(0.3, Y_EMBED + BOX_H / 2 + 1.6),
                xytext=(1.8, Y_EMBED + BOX_H / 2 + 1.6),
                arrowprops=dict(arrowstyle="<|-|>", color=C_ATTN, lw=1.2,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(2.7, Y_EMBED + BOX_H / 2 + 1.6, "attends to (bidirectional)",
            fontsize=8, color=C_ATTN, va="center")


def draw_diagram(mode, title, backbone_label, output_name):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1.5, 8.5)
    ax.axis("off")

    # Backbone box
    backbone_rect = mpatches.FancyBboxPatch(
        (XS[0] - 0.8, Y_EMBED - 0.6), XS[-1] - XS[0] + 1.6, Y_HIDDEN - Y_EMBED + 1.2,
        boxstyle="round,pad=0.2",
        facecolor=C_BACKBONE, edgecolor="#7986CB", linewidth=2.5, alpha=0.4,
    )
    ax.add_patch(backbone_rect)
    ax.text((XS[0] + XS[-1]) / 2, (Y_EMBED + Y_HIDDEN) / 2, backbone_label,
            ha="center", va="center", fontsize=14, weight="bold", color="#3949AB", alpha=0.55)

    draw_common(ax)

    if mode == "causal":
        draw_causal_attention(ax)
    else:
        draw_bidirectional_attention(ax)

    ax.text((XS[0] + XS[-1]) / 2, 8.2, title,
            ha="center", va="center", fontsize=15, weight="bold")

    plt.tight_layout()
    plt.savefig(f"assets/{output_name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(f"assets/{output_name}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved assets/{output_name}.png and assets/{output_name}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    draw_diagram(
        mode="causal",
        title="Causal LM: Hidden State Classification",
        backbone_label="Frozen Causal LM  (Qwen3-0.6B)",
        output_name="architecture_causal",
    )
    draw_diagram(
        mode="bidirectional",
        title="Bidirectional Variant: Hidden State Classification",
        backbone_label="Frozen Bidirectional Encoder  (Qwen3-Embedding-0.6B)",
        output_name="architecture_bidirectional",
    )
