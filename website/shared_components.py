"""
Shared UI components for the SolarSoundBytes website
"""

def get_emoji_title(include_team=False):
    """
    Returns the consistent emoji title for SolarSoundBytes
    
    Args:
        include_team (bool): Whether to include "Team" at the end
    
    Returns:
        str: The emoji title string
    """
    base_title = "☀️Solar🔊Sound🍔Bytes"
    if include_team:
        return f"{base_title}🫂Team"
    return base_title

def render_emoji_title_header(include_team=False, size="h1", center=True):
    """
    Renders the emoji title as an HTML header
    
    Args:
        include_team (bool): Whether to include "Team" at the end
        size (str): HTML header size (h1, h2, h3, etc.)
        center (bool): Whether to center the title
    
    Returns:
        str: HTML string for the title
    """
    title = get_emoji_title(include_team)
    alignment = "text-align: center" if center else "text-align: left"
    
    return f'<{size} style="{alignment}">{title}</{size}>'

def get_emoji_link_text():
    """
    Returns the emoji title formatted for use in links
    
    Returns:
        str: The emoji title for links
    """
    return get_emoji_title(include_team=True) 