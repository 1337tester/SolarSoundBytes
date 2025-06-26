# Using the Shared Emoji Title Components

The `shared_components.py` file provides utilities to use the consistent emoji title "☀️Solar🔊Sound🍔Bytes" across all pages.

## Usage Examples

### For pages in the root `website/` directory:

```python
import streamlit as st
from shared_components import get_emoji_title, render_emoji_title_header, get_emoji_link_text

# Use in title
st.markdown(f"<h1>Welcome to {get_emoji_title()}</h1>", unsafe_allow_html=True)

# Use in links 
st.markdown(f'<a href="/about_us">{get_emoji_link_text()}</a>', unsafe_allow_html=True)

# Use with "Team" included
st.title(f"{get_emoji_title(include_team=True)}")
```

### For pages in the `website/pages/` directory:

```python
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_components import get_emoji_title

# Use in your page
st.title(f"{get_emoji_title()} Dashboard")
```

## Available Functions

- `get_emoji_title(include_team=False)` - Returns the base emoji title
- `get_emoji_link_text()` - Returns emoji title with "Team" for links  
- `render_emoji_title_header()` - Returns formatted HTML header

This ensures consistency across all pages and makes it easy to update the emoji title in one place. 