"""
Visualization Module
Creates performance charts and analytical reports.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_performance_comparison(
    results_dict: Dict[str, Dict],
    output_path: str = "results/performance_comparison.png",
) -> None:
    """
    Plot performance comparison between configurations (fine-grained vs coarse-grained).
    Replicates the type of analysis in the paper.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Multi-Agent LLM Trading System Performance Comparison\n"
        "(arXiv:2602.23330 - Fine-Grained vs Coarse-Grained Tasks)",
        fontsize=13,
        fontweight="bold",
    )

    colors = {"fine_grained": "#2196F3", "coarse_grained": "#FF9800", "baseline": "#4CAF50"}

    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    for config_name, results in results_dict.items():
        monthly_returns = results.get("monthly_returns", {})
        if monthly_returns:
            returns_series = pd.Series(monthly_returns)
            if not returns_series.empty:
                cum_returns = (1 + returns_series).cumprod()
                color = colors.get(config_name, "#9E9E9E")
                ax1.plot(
                    range(len(cum_returns)),
                    cum_returns.values,
                    label=config_name.replace("_", " ").title(),
                    color=color,
                    linewidth=2,
                )

    ax1.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
    ax1.set_title("Cumulative Returns (Monthly Rebalancing)", fontweight="bold")
    ax1.set_xlabel("Months")
    ax1.set_ylabel("Cumulative Return (1 = base)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Sharpe Ratio Comparison
    ax2 = axes[0, 1]
    config_names = list(results_dict.keys())
    sharpe_values = [
        results_dict[c].get("performance", {}).get("sharpe_ratio", 0)
        for c in config_names
    ]
    bar_colors = [colors.get(c, "#9E9E9E") for c in config_names]
    bars = ax2.bar(
        [c.replace("_", "\n").title() for c in config_names],
        sharpe_values,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, val in zip(bars, sharpe_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_title("Sharpe Ratio Comparison", fontweight="bold")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Risk-Return Scatter
    ax3 = axes[1, 0]
    for config_name, results in results_dict.items():
        perf = results.get("performance", {})
        ann_vol = perf.get("annual_volatility", 0) * 100
        ann_ret = perf.get("annual_return", 0) * 100
        color = colors.get(config_name, "#9E9E9E")
        ax3.scatter(
            ann_vol, ann_ret,
            color=color,
            s=150,
            zorder=5,
            label=config_name.replace("_", " ").title(),
            edgecolors="black",
            linewidth=0.8,
        )
        ax3.annotate(
            config_name.replace("_", " "),
            (ann_vol, ann_ret),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax3.set_title("Risk-Return Profile", fontweight="bold")
    ax3.set_xlabel("Annual Volatility (%)")
    ax3.set_ylabel("Annual Return (%)")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Key Metrics Table
    ax4 = axes[1, 1]
    ax4.axis("off")

    metrics_to_show = [
        ("Annual Return", "annual_return", "{:.1%}"),
        ("Annual Vol", "annual_volatility", "{:.1%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.1%}"),
        ("Calmar Ratio", "calmar_ratio", "{:.3f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
    ]

    col_labels = ["Metric"] + [c.replace("_", " ").title() for c in config_names]
    table_data = []
    for label, key, fmt in metrics_to_show:
        row = [label]
        for c in config_names:
            val = results_dict[c].get("performance", {}).get(key, 0)
            row.append(fmt.format(val))
        table_data.append(row)

    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax4.set_title("Performance Metrics Summary", fontweight="bold", pad=20)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distribution(
    scores_history: List[Dict],
    output_path: str = "results/score_distribution.png",
) -> None:
    """Plot distribution of attractiveness scores over time."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PM Agent Score Distribution Analysis", fontweight="bold")

    all_scores = []
    for period_data in scores_history:
        scores = list(period_data.get("scores", {}).values())
        all_scores.extend(scores)

    if all_scores:
        # Score histogram
        ax1 = axes[0]
        ax1.hist(all_scores, bins=20, color="#2196F3", alpha=0.7, edgecolor="black")
        ax1.axvline(x=70, color="green", linestyle="--", label="Long threshold (70)")
        ax1.axvline(x=30, color="red", linestyle="--", label="Short threshold (30)")
        ax1.set_title("Distribution of Attractiveness Scores")
        ax1.set_xlabel("Score (0-100)")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Score over time
        ax2 = axes[1]
        if len(scores_history) > 1:
            dates = [d.get("date", f"T{i}") for i, d in enumerate(scores_history)]
            avg_scores = [
                np.mean(list(d.get("scores", {50: 50}).values()))
                for d in scores_history
            ]
            top_scores = [
                np.percentile(list(d.get("scores", {50: 50}).values()), 90)
                for d in scores_history
            ]
            bot_scores = [
                np.percentile(list(d.get("scores", {50: 50}).values()), 10)
                for d in scores_history
            ]
            x = range(len(dates))
            ax2.plot(x, avg_scores, label="Average", color="#2196F3", linewidth=2)
            ax2.fill_between(x, bot_scores, top_scores, alpha=0.2, color="#2196F3",
                            label="10th-90th percentile")
            ax2.axhline(y=70, color="green", linestyle="--", alpha=0.5)
            ax2.axhline(y=30, color="red", linestyle="--", alpha=0.5)
            ax2.set_title("Score Evolution Over Rebalancing Periods")
            ax2.set_xlabel("Rebalancing Period")
            ax2.set_ylabel("Attractiveness Score")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
