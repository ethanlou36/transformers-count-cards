from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import FancyBboxPatch

from .deck_tracker import _BAR_ORDER, Frame, ShoeViz

_FELT = "#0b6b3a"
_BAR = "#2a78d6"
_HIGHLIGHT = "#eb6834"
_CHIP = "#eda100"
_TRUE_TICK = "#1a1a19"
_INK = "#0b0b0b"
_MAXBAR = 16
_DISPLAY = {r: ("10" if r == "T" else r) for r in _BAR_ORDER}


def _bars(ax, values, title, true_values=None, key=False):
    ax.clear()
    y = range(len(_BAR_ORDER))
    ax.barh(y, values, color=_BAR, height=0.72, zorder=2, label="model's guess")
    if true_values is not None:
        for i, tv in enumerate(true_values):
            ax.plot([tv, tv], [i - 0.4, i + 0.4], color=_TRUE_TICK, lw=2.4, zorder=3)
    ax.set_xlim(0, _MAXBAR)
    ax.set_ylim(-0.6, len(_BAR_ORDER) - 0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [_DISPLAY[r] for r in _BAR_ORDER], fontsize=14, fontweight="bold", color=_INK
    )
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.set_title(title, fontsize=15, fontweight="bold", color=_INK, pad=10)
    ax.invert_yaxis()
    if key and true_values is not None:
        ax.text(
            0.98,
            0.52,
            "blue = model's guess",
            transform=ax.transAxes,
            ha="right",
            fontsize=11.5,
            color=_BAR,
            fontweight="bold",
        )
        ax.text(
            0.98,
            0.46,
            "black line = what's really left",
            transform=ax.transAxes,
            ha="right",
            fontsize=11.5,
            color=_TRUE_TICK,
            fontweight="bold",
        )


def _card(ax, x, y, rank, highlight=False, facedown=False):
    w, h = 0.62, 0.92
    face = "#d9d9d9" if facedown else "white"
    edge = _HIGHLIGHT if highlight else "#333333"
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=3 if highlight else 1.4,
            edgecolor=edge,
            facecolor=face,
            mutation_aspect=1.0,
            zorder=4,
        )
    )
    if not facedown:
        ax.text(
            x + w / 2,
            y + h / 2,
            _DISPLAY.get(rank, rank),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#b00000" if rank in ("A", "T") else "#111111",
            zorder=5,
        )


def _table(ax, frame: Frame):
    ax.clear()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 8, 6, color=_FELT, zorder=0))
    ax.text(
        0.2,
        5.7,
        f"Hand {frame.hand_number}",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color="#d7ffe6",
    )
    ax.text(
        4, 5.6, "Dealer", ha="center", fontsize=13, fontweight="bold", color="white"
    )
    for i, r in enumerate(frame.dealer_cards):
        _card(
            ax,
            0.6 + i * 0.75,
            4.2,
            r,
            highlight=(
                frame.dealt_to == "dealer"
                and i == len(frame.dealer_cards) - 1
                and r == frame.just_dealt
            ),
            facedown=(r == "?"),
        )
    ax.text(4, 2.5, "You", ha="center", fontsize=13, fontweight="bold", color="white")
    for i, r in enumerate(frame.player_cards):
        _card(
            ax,
            0.6 + i * 0.75,
            1.1,
            r,
            highlight=(frame.dealt_to == "player" and i == len(frame.player_cards) - 1),
        )


def _chips(ax, frame: Frame, show_annotation: bool):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax.text(0.2, 1.5, "Model's bet:", fontsize=13, fontweight="bold", color=_INK)

    xs = [2.7 + i * 0.42 for i in range(frame.bet)]
    ax.scatter(
        xs,
        [1.0] * frame.bet,
        s=520,
        marker="o",
        facecolor=_CHIP,
        edgecolor="#8a5b00",
        linewidth=1.4,
        zorder=3,
    )
    ax.text(
        2.7 + max(frame.bet, 1) * 0.42 + 0.25,
        1.0,
        f"{frame.bet}x",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=_INK,
    )
    if show_annotation and frame.bet > 1:
        ax.text(
            0.2,
            0.35,
            "deck is rich in high cards → model bets more",
            fontsize=12.5,
            style="italic",
            color=_HIGHLIGHT,
            fontweight="bold",
        )


def _fig():
    fig = plt.figure(figsize=(12.5, 7.4), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.15, 1],
        height_ratios=[5, 1.3],
        hspace=0.28,
        wspace=0.16,
        left=0.03,
        right=0.96,
        top=0.82,
        bottom=0.06,
    )
    table = fig.add_subplot(gs[0, 0])
    bars = fig.add_subplot(gs[0, 1])
    chips = fig.add_subplot(gs[1, :])
    return fig, (table, bars, chips)


def _draw(axes, frame: Frame, shoeviz: ShoeViz, seen_high_bet: list[bool]):
    table, bars, chips = axes
    _table(table, frame)
    _bars(
        bars,
        frame.model_remaining,
        "How many cards of each rank are left",
        true_values=frame.true_remaining,
        key=True,
    )
    annotate = frame.bet > 1 and not seen_high_bet[0]
    if annotate:
        seen_high_bet[0] = True
    _chips(chips, frame, show_annotation=(frame.bet > 1))


def render_animation(
    shoeviz: ShoeViz,
    out_prefix: Path,
    fps: int = 3,
    title: str | None = None,
    caption: str | None = None,
) -> dict[str, Path]:
    from matplotlib.animation import FuncAnimation

    fig, axes = _fig()
    seen = [False]
    frames = shoeviz.frames + [shoeviz.frames[-1]] * 5

    fig.suptitle(
        title or "Watch the model count the deck",
        y=0.975,
        fontsize=18,
        fontweight="bold",
        color=_INK,
    )
    if caption:
        fig.text(
            0.5,
            0.90,
            caption,
            ha="center",
            fontsize=12.5,
            color=_HIGHLIGHT,
            fontweight="bold",
            wrap=True,
        )
    fig.text(
        0.5,
        0.015,
        "the blue bars are decoded from the model's internal activity as the "
        "hand is played",
        ha="center",
        fontsize=10.5,
        color="#52514e",
    )

    def update(i):
        _draw(axes, frames[i], shoeviz, seen)
        return []

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    gif = out_prefix.with_suffix(".gif")
    mp4 = out_prefix.with_suffix(".mp4")
    anim.save(gif, writer=PillowWriter(fps=fps))
    outputs = {"gif": gif}
    try:
        anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=2400))
        outputs["mp4"] = mp4
    except (FileNotFoundError, RuntimeError):
        pass
    plt.close(fig)
    return outputs


def render_filmstrip(shoeviz: ShoeViz, out_path: Path, every: int = 5) -> Path:
    picks = shoeviz.frames[::every]
    if picks[-1] is not shoeviz.frames[-1]:
        picks = [*list(picks), shoeviz.frames[-1]]
    fig, axes = plt.subplots(
        1, len(picks), figsize=(1.9 * len(picks), 4.4), facecolor="white"
    )
    for col, frame in enumerate(picks):
        _bars(
            axes[col],
            frame.model_remaining,
            f"card {frame.cards_seen}",
            true_values=frame.true_remaining,
        )
        if col > 0:
            axes[col].set_yticklabels([])
    fig.suptitle(
        "The model's belief (blue) tracks the real deck (black lines) as cards "
        "are dealt",
        fontsize=15,
        fontweight="bold",
        color=_INK,
    )
    fig.text(
        0.5,
        0.005,
        "blue bars decoded from the model's internal activity · black lines = "
        "what's really left",
        ha="center",
        fontsize=9.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_infinite_contrast(
    true_remaining, finite_belief, infinite_belief, cards_seen: int, out_path: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), facecolor="white")
    _bars(axes[0], finite_belief, "Normal model", true_values=true_remaining, key=True)
    _bars(
        axes[1],
        infinite_belief,
        "Infinite-deck model",
        true_values=true_remaining,
        key=True,
    )
    fig.suptitle(
        f"After {cards_seen} cards: remove the reason to count and the "
        "ability disappears",
        fontsize=15,
        fontweight="bold",
        color=_INK,
    )
    fig.text(
        0.5,
        0.02,
        "blue = each model's guess, black line = what's really left · the "
        "infinite-deck model never learns which cards are gone",
        ha="center",
        fontsize=10,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
