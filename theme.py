# =============================================================================
# theme.py — Healthcare Visual Identity Configuration
# =============================================================================
# This file centralises every colour used across the Medical Report Summarizer
# project.  Importing THEME from here means you only ever change a colour in
# ONE place and the whole app updates automatically.
#
# Usage example:
#   from theme import THEME
#   print(THEME["primary"])   # → "#1A73E8"
# =============================================================================

THEME: dict[str, str] = {
    # ------------------------------------------------------------------
    # Background
    # Used as the base canvas colour for all pages / panels.
    # ------------------------------------------------------------------
    "background": "#FFFFFF",      # Bright White

    # ------------------------------------------------------------------
    # Primary / Buttons
    # Main action colour: submit buttons, links, active nav items.
    # ------------------------------------------------------------------
    "primary": "#1A73E8",         # Oceanic Blue

    # ------------------------------------------------------------------
    # Success / Citations
    # Displayed next to successfully extracted citations and OK banners.
    # ------------------------------------------------------------------
    "success": "#2E7D32",         # Healing Green

    # ------------------------------------------------------------------
    # Headers
    # Section titles, sidebar headings, modal titles.
    # ------------------------------------------------------------------
    "header": "#6A1B9A",          # Professional Purple

    # ------------------------------------------------------------------
    # Warnings / Disclaimers
    # Medical disclaimers, error messages, destructive-action buttons.
    # ------------------------------------------------------------------
    "warning": "#C62828",         # Vitality Red

    # ------------------------------------------------------------------
    # Highlights
    # Soft background behind key terms, tooltips, and hover states.
    # ------------------------------------------------------------------
    "highlight": "#FFF9C4",       # Gentle Yellow
}


# ------------------------------------------------------------------
# Convenience aliases — use whichever naming style feels natural
# ------------------------------------------------------------------
BACKGROUND_COLOR  = THEME["background"]   # #FFFFFF  — Bright White
PRIMARY_COLOR     = THEME["primary"]      # #1A73E8  — Oceanic Blue
SUCCESS_COLOR     = THEME["success"]      # #2E7D32  — Healing Green
HEADER_COLOR      = THEME["header"]       # #6A1B9A  — Professional Purple
WARNING_COLOR     = THEME["warning"]      # #C62828  — Vitality Red
HIGHLIGHT_COLOR   = THEME["highlight"]    # #FFF9C4  — Gentle Yellow


# ------------------------------------------------------------------
# Quick self-test — run `python theme.py` to print the palette
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n[THEME]  Medical Report Summarizer - Colour Palette\n")
    labels = {
        "background": "Bright White      (Background)",
        "primary":    "Oceanic Blue      (Primary / Buttons)",
        "success":    "Healing Green     (Success / Citations)",
        "header":     "Professional Purple (Headers)",
        "warning":    "Vitality Red      (Warnings / Disclaimers)",
        "highlight":  "Gentle Yellow     (Highlights)",
    }
    for key, label in labels.items():
        print(f"  {label:45s}  ->  {THEME[key]}")
    print()
