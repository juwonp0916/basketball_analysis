"""
SVG-based Court Renderer - Unified visualization for frontend and backend

Uses the same coordinate system as the frontend SVG:
- X domain: [0, 50] (left to right, corresponds to 15m court width)
- Y domain: [47, 0] (bottom to top, corresponds to 14m half-court length)
- Origin at TOP-LEFT in SVG coordinates
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from typing import List, Tuple, Optional


class CourtRenderer:
    """Renders basketball half-court using the same SVG layout as frontend"""

    # SVG viewBox dimensions (matches frontend exactly)
    SVG_WIDTH = 50.0
    SVG_HEIGHT = 47.0

    # FIBA court dimensions in meters
    COURT_WIDTH_M = 15.0
    COURT_HALF_LENGTH_M = 14.0

    # Colors matching frontend (dark theme)
    BG_COLOR = '#1f2937'  # Dark gray background
    LINE_COLOR = '#374151'  # Medium gray lines
    BASKET_COLOR = '#ef4444'  # Red basket
    MADE_COLOR = '#10b981'  # Green for made shots
    MISS_COLOR = '#ef4444'  # Red for missed shots

    @staticmethod
    def meters_to_svg(x_m: float, y_m: float) -> Tuple[float, float]:
        """
        Convert FIBA court coordinates (meters) to SVG coordinates

        FIBA coordinates:
        - X: [0, 15]m (left to right)
        - Y: [0, 14]m (baseline to half-court)

        SVG coordinates (frontend):
        - X: [0, 50] (left to right)
        - Y: [0, 47] (top to bottom in image, but baseline at top)

        The frontend uses yDomain=[47, 0] which inverts rendering,
        so baseline (Y=0m) appears at top (Y=0 SVG).
        """
        svg_x = (x_m / CourtRenderer.COURT_WIDTH_M) * CourtRenderer.SVG_WIDTH
        # Y is NOT flipped - baseline (0m) maps to SVG Y=0 (top)
        svg_y = (y_m / CourtRenderer.COURT_HALF_LENGTH_M) * CourtRenderer.SVG_HEIGHT

        return svg_x, svg_y

    @staticmethod
    def draw_court(ax: plt.Axes, show_dimensions: bool = False):
        """
        Draw the basketball half-court using SVG coordinates

        Args:
            ax: Matplotlib axes to draw on
            show_dimensions: Whether to show dimension labels
        """
        # Set axis limits to match SVG viewBox
        ax.set_xlim(0, CourtRenderer.SVG_WIDTH)
        ax.set_ylim(CourtRenderer.SVG_HEIGHT, 0)  # Invert Y axis (0 at top)
        ax.set_aspect('equal')

        # Background
        ax.set_facecolor(CourtRenderer.BG_COLOR)

        # Court boundary (0, 0, 50, 47)
        court_boundary = patches.Rectangle(
            (0, 0), CourtRenderer.SVG_WIDTH, CourtRenderer.SVG_HEIGHT,
            linewidth=2, edgecolor=CourtRenderer.LINE_COLOR, facecolor='none'
        )
        ax.add_patch(court_boundary)

        # Penalty box (paint) - (17, 0, 16, 19)
        penalty_box = patches.Rectangle(
            (17, 0), 16, 19,
            linewidth=2, edgecolor=CourtRenderer.LINE_COLOR, facecolor='none'
        )
        ax.add_patch(penalty_box)

        # Free throw circle - center (25, 19), radius 6
        ft_circle = patches.Circle(
            (25, 19), 6,
            linewidth=2, edgecolor=CourtRenderer.LINE_COLOR, facecolor='none'
        )
        ax.add_patch(ft_circle)

        # 3-point arc - path: M 3 0 L 3 14 Q 25 43 47 14 L 47 0
        # Convert SVG path to matplotlib path
        arc_vertices = [
            (3, 0),   # Start at left corner
            (3, 14),  # Straight line down
            (25, 43), # Quadratic control point
            (47, 14), # Curve to right side
            (47, 0),  # Straight line up to right corner
        ]
        arc_codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO]
        arc_path = Path(arc_vertices, arc_codes)
        arc_patch = patches.PathPatch(
            arc_path, linewidth=2, edgecolor=CourtRenderer.LINE_COLOR, facecolor='none'
        )
        ax.add_patch(arc_patch)

        # Basket (hoop) - circle at (25, 5.25), radius 0.75
        basket = patches.Circle(
            (25, 5.25), 0.75,
            linewidth=2, edgecolor=CourtRenderer.BASKET_COLOR, facecolor='none'
        )
        ax.add_patch(basket)

        # Backboard - line from (22, 4) to (28, 4)
        ax.plot([22, 28], [4, 4], color=CourtRenderer.LINE_COLOR, linewidth=2)

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Optional dimension labels
        if show_dimensions:
            ax.text(
                CourtRenderer.SVG_WIDTH / 2, CourtRenderer.SVG_HEIGHT - 1,
                'FIBA Half Court',
                ha='center', va='top', fontsize=10, color='#6b7280'
            )

    @staticmethod
    def plot_shot(
        ax: plt.Axes,
        x_m: float,
        y_m: float,
        is_made: bool,
        shot_number: Optional[int] = None,
        alpha: float = 0.8
    ):
        """
        Plot a single shot on the court

        Args:
            ax: Matplotlib axes
            x_m: X coordinate in meters (FIBA)
            y_m: Y coordinate in meters (FIBA)
            is_made: True if shot was made
            shot_number: Optional shot number to display
            alpha: Transparency (0-1)
        """
        # Convert to SVG coordinates
        svg_x, svg_y = CourtRenderer.meters_to_svg(x_m, y_m)

        # Choose color
        color = CourtRenderer.MADE_COLOR if is_made else CourtRenderer.MISS_COLOR

        # Draw shot marker
        shot_marker = patches.Circle(
            (svg_x, svg_y), 1.2,
            facecolor=color, edgecolor='white', linewidth=1, alpha=alpha
        )
        ax.add_patch(shot_marker)

        # Optional shot number
        if shot_number is not None:
            ax.text(
                svg_x, svg_y, str(shot_number),
                ha='center', va='center', fontsize=6,
                color='white', weight='bold'
            )

    @staticmethod
    def create_shot_chart(
        shots: List[Tuple[float, float, bool]],
        title: str = "Shot Chart",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 9.4),
        show_stats: bool = True
    ) -> Optional[plt.Figure]:
        """
        Create a complete shot chart

        Args:
            shots: List of (x_m, y_m, is_made) tuples
            title: Chart title
            save_path: Optional path to save the figure
            figsize: Figure size (width, height) in inches
            show_stats: Whether to show statistics overlay

        Returns:
            Figure object (if not saving)
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Draw court
        CourtRenderer.draw_court(ax)

        # Plot shots
        for i, (x_m, y_m, is_made) in enumerate(shots, 1):
            CourtRenderer.plot_shot(ax, x_m, y_m, is_made, shot_number=i)

        # Title
        ax.set_title(title, fontsize=16, color='white', pad=20)

        # Statistics overlay
        if show_stats and shots:
            made_count = sum(1 for _, _, is_made in shots if is_made)
            total_count = len(shots)
            pct = (made_count / total_count * 100) if total_count > 0 else 0

            stats_text = f"{made_count}/{total_count} ({pct:.1f}%)"
            ax.text(
                2, 2, stats_text,
                fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
            )

            # Legend
            legend_y = 6
            # Made shots
            made_marker = patches.Circle(
                (2, legend_y), 0.8,
                facecolor=CourtRenderer.MADE_COLOR, edgecolor='white'
            )
            ax.add_patch(made_marker)
            ax.text(4, legend_y, f"Made: {made_count}", fontsize=9, color='white', va='center')

            # Missed shots
            miss_marker = patches.Circle(
                (2, legend_y + 3), 0.8,
                facecolor=CourtRenderer.MISS_COLOR, edgecolor='white'
            )
            ax.add_patch(miss_marker)
            ax.text(
                4, legend_y + 3, f"Missed: {total_count - made_count}",
                fontsize=9, color='white', va='center'
            )

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, facecolor=CourtRenderer.BG_COLOR)
            plt.close(fig)
            return None
        else:
            return fig


# Convenience function
def create_shot_chart(
    shots: List[Tuple[float, float, bool]],
    save_path: str,
    title: str = "Shot Chart"
):
    """
    Quick function to create and save a shot chart

    Args:
        shots: List of (x_meters, y_meters, is_made) tuples
        save_path: Path to save the chart
        title: Chart title
    """
    CourtRenderer.create_shot_chart(
        shots=shots,
        title=title,
        save_path=save_path
    )


if __name__ == "__main__":
    # Test the renderer
    import random

    print("Testing CourtRenderer...")

    # Generate some random shots
    test_shots = []
    for _ in range(20):
        x_m = random.uniform(2, 13)  # Random X in court
        y_m = random.uniform(1, 12)  # Random Y in court
        is_made = random.random() > 0.5
        test_shots.append((x_m, y_m, is_made))

    # Create shot chart
    create_shot_chart(
        shots=test_shots,
        save_path="test_shot_chart.png",
        title="Test Shot Chart"
    )

    print("✓ Test shot chart saved to test_shot_chart.png")
