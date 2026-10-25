"""
MiniMessage Parser Module

Parses MiniMessage format strings into Minecraft standard JSON text components.
Reference: https://docs.advntr.dev/minimessage/format.html

Example usage:
    >>> from datatypes import parse_mini_message
    >>> result = parse_mini_message("<green>Mini Message")
    >>> print(result)
    {'text': 'Mini Message', 'color': 'green'}
"""

import re
import json
from loguru import logger
from typing import Any, Optional, Union
from dataclasses import dataclass, field


# Minecraft color constants
COLOR_NAMES = {
    "black": "#000000",
    "dark_blue": "#0000AA",
    "dark_green": "#00AA00",
    "dark_aqua": "#00AAAA",
    "dark_red": "#AA0000",
    "dark_purple": "#AA00AA",
    "gold": "#FFAA00",
    "gray": "#555555",
    "grey": "#555555",
    "dark_gray": "#555555",
    "dark_grey": "#555555",
    "blue": "#5555FF",
    "green": "#55FF55",
    "aqua": "#55FFFF",
    "red": "#FF5555",
    "light_purple": "#FF55FF",
    "yellow": "#FFFF55",
    "white": "#FFFFFF",
}

# Decoration mappings to Minecraft JSON format
DECORATION_MAP = {
    "bold": "bold",
    "b": "bold",
    "italic": "italic",
    "em": "italic",
    "i": "italic",
    "underlined": "underlined",
    "u": "underlined",
    "strikethrough": "strikethrough",
    "st": "strikethrough",
    "obfuscated": "obfuscated",
    "obf": "obfuscated",
}

# Pride flag colors (since v4.18.0)
PRIDE_FLAGS = {
    # Standard pride flags
    "pride": ["#FF0018", "#FFA52C", "#FFFF41", "#008018", "#0000F9", "#86007D"],
    "progress": ["#FF0018", "#FF8C00", "#FFED00", "#008018", "#0000F9", "#86007D"],
    
    # Specific identity flags
    "trans": ["#5BCEFA", "#F5A9B8", "#FFFFFF", "#F5A9B8", "#5BCEFA"],
    "bi": ["#D60270", "#9B59B6", "#0038A8"],
    "pan": ["#FF218C", "#FFD800", "#21B1FF"],
    "nb": ["#FCF434", "#FFFFFF", "#9C59D1", "#2C2C2C"],
    "lesbian": ["#D62800", "#FF9A56", "#FFFFFF", "#FF9A56", "#D62800"],
    "ace": ["#000000", "#A3A3A3", "#81007D"],
    "agender": ["#000000", "#A3A3A3", "#FFFFFF", "#6F732F", "#FFFFFF", "#A3A3A3", "#000000"],
    "demisexual": ["#000000", "#A3A3A3", "#596DD8", "#8F6FC4", "#CF592E"],
    "genderqueer": ["#B57EDC", "#FFFFFF", "#496304"],
    "genderfluid": ["#FF75A9", "#FFFFFF", "#FFE133", "#000000", "#8F59D1"],
    "intersex": ["#FFE216", "#FFA500"],
    "aro": ["#3DA642", "#78C46C", "#9F4995", "#000000", "#3A3A3A"],
    "baker": ["#FFFFFF", "#F19CBB", "#FFE4E1", "#F19CBB", "#FFFFFF"],
    "philly": ["#FF218C", "#FF8C00", "#FFED00", "#008018", "#0000F9", "#86007D"],
    "queer": ["#7B7B7B", "#AAAAAA", "#FFFFFF", "#AAAAAA", "#7B7B7B"],
    "gay": ["#FF218C", "#FF8C00", "#FFED00", "#008018", "#0000F9", "#86007D"],
    "bigender": ["#FF75A9", "#F5A9B8", "#FFFFFF", "#9C59D1", "#CF592E"],
    "demigender": ["#9C59D1", "#496304", "#FFFFFF", "#FFFFFF", "#496304", "#9C59D1"],
}

# Rainbow color sequence (6 colors)
RAINBOW_COLORS = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF", "#EE82EE"]


@dataclass
class TextComponent:
    """Represents a Minecraft text component."""
    text: str = ""
    color: Optional[str] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underlined: Optional[bool] = None
    strikethrough: Optional[bool] = None
    obfuscated: Optional[bool] = None
    click_event: Optional[dict] = None
    hover_event: Optional[dict] = None
    insertion: Optional[str] = None
    keybind: Optional[str] = None
    translate: Optional[str] = None
    with_: Optional[list] = None
    extra: list = field(default_factory=list)
    font: Optional[str] = None
    shadow_color: Optional[str] = None
    
    # Animation support for gradient/rainbow/transition/pride
    gradient_colors: Optional[list[str]] = None
    animation_type: Optional[str] = None  # "gradient", "rainbow", "transition", "pride"
    animation_phase: Optional[float] = None
    animation_reversed: bool = False
    
    # Fallback support for unsupported tags
    fallback_text: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to Minecraft JSON component format."""
        result = {}
        
        if self.text:
            result["text"] = self.text
        
        if self.color is not None:
            result["color"] = self.color
        
        if self.bold is not None:
            result["bold"] = self.bold
        
        if self.italic is not None:
            result["italic"] = self.italic
        
        if self.underlined is not None:
            result["underlined"] = self.underlined
        
        if self.strikethrough is not None:
            result["strikethrough"] = self.strikethrough
        
        if self.obfuscated is not None:
            result["obfuscated"] = self.obfuscated
        
        if self.click_event is not None:
            result["clickEvent"] = self.click_event
        
        if self.hover_event is not None:
            result["hoverEvent"] = self.hover_event
        
        if self.insertion is not None:
            result["insertion"] = self.insertion
        
        if self.keybind is not None:
            result["keybind"] = self.keybind
        
        if self.translate is not None:
            result["translate"] = self.translate
            if self.with_:
                result["with"] = self.with_
        
        if self.font is not None:
            result["font"] = self.font
        
        if self.shadow_color is not None:
            result["shadow_color"] = self.shadow_color
        
        if self.extra:
            result["extra"] = self.extra
            # When extra is present, always include text field (even empty)
            # to properly close the JSON structure
            if "text" not in result:
                result["text"] = ""
        
        # Add gradient info for animation effects (rainbow, gradient, transition, pride)
        # This will be expanded into individual color-per-character by _expand_animation_effects()
        if self.animation_type is not None and self.gradient_colors:
            result["gradient"] = {
                "colors": self.gradient_colors,
                "type": self.animation_type,
            }
            if self.animation_phase is not None:
                result["gradient"]["phase"] = self.animation_phase
            if self.animation_reversed:
                result["gradient"]["reversed"] = True

        return result
    
    def copy_styles(self) -> 'TextComponent':
        """Create a copy with only styles (no text or extra)."""
        return TextComponent(
            text="",
            color=self.color,
            bold=self.bold,
            italic=self.italic,
            underlined=self.underlined,
            strikethrough=self.strikethrough,
            obfuscated=self.obfuscated,
            click_event=self.click_event,
            hover_event=self.hover_event,
            insertion=self.insertion,
            keybind=self.keybind,
            translate=self.translate,
            with_=self.with_.copy() if self.with_ else None,
            font=self.font,
            shadow_color=self.shadow_color,
            gradient_colors=self.gradient_colors.copy() if self.gradient_colors else None,
            animation_type=self.animation_type,
            animation_phase=self.animation_phase,
            animation_reversed=self.animation_reversed,
            fallback_text=self.fallback_text,
        )


class MiniMessageParser:
    """
    Parser for MiniMessage format strings.
    
    Converts MiniMessage tagged text into Minecraft JSON text components.
    Uses a tokenization approach for cleaner parsing.
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize the parser.
        
        Args:
            strict: If True, enforce strict mode (all tags must be closed, no <reset>)
        """
        self.strict = strict
    
    def parse(self, input_string: str) -> dict:
        """
        Parse a MiniMessage string into a Minecraft JSON text component.
        
        Args:
            input_string: The MiniMessage formatted string
            
        Returns:
            A dictionary representing the Minecraft JSON text component
        """
        # Tokenize the input
        tokens = self._tokenize(input_string)
        # Parse tokens into component
        component, _ = self._parse_tokens(tokens, 0, TextComponent())
        result = component.to_dict()
        
        # Expand animation effects (gradient, rainbow, transition, pride)
        # into individual color-per-character format
        result = self._expand_animation_effects(result)
        
        return result
    
    def _expand_animation_effects(self, component: dict) -> dict:
        """
        Expand animation effects (gradient, rainbow, transition, pride)
        into individual color-per-character components.
        
        This converts the gradient metadata into actual text components
        with individual colors for each character.
        
        Args:
            component: The parsed component dict
            
        Returns:
            The component with animation effects expanded
        """
        # Check if this component itself has gradient (root level)
        if "gradient" in component and "text" in component:
            text_content = component.get("text", "")
            if text_content:
                expanded = self._expand_text_with_colors(text_content, component)
                if expanded:
                    # Remove gradient key and text, keep other styles
                    new_component = {}
                    for key, value in component.items():
                        if key not in ("gradient", "text"):
                            new_component[key] = value
                    new_component["extra"] = expanded
                    new_component["text"] = ""
                    return new_component
        
        # Check if this component has extra items with animation effects
        if "extra" in component:
            expanded_extra = []
            for extra_item in component["extra"]:
                # Recursively process nested extra items
                if isinstance(extra_item, dict):
                    extra_item = self._expand_animation_effects(extra_item)
                    expanded_extra.append(extra_item)
                else:
                    expanded_extra.append(extra_item)
            
            if expanded_extra:
                component["extra"] = expanded_extra
        
        return component
    
    def _expand_text_with_colors(self, text: str, component: dict) -> list:
        """
        Expand a text string into individual character components with colors.
        
        Args:
            text: The text to expand
            component: The parent component containing gradient info
            
        Returns:
            List of text components, one per character
        """
        gradient_info = component.get("gradient", {})
        colors = gradient_info.get("colors", ["#FFFFFF"])
        phase = gradient_info.get("phase", 0)
        reversed_ = gradient_info.get("reversed", False)
        
        # Get decoration styles from parent
        styles = {}
        for style_key in ["bold", "italic", "underlined", "strikethrough", "obfuscated", "font"]:
            if style_key in component:
                styles[style_key] = component[style_key]
        
        # Calculate colors for each character
        result = []
        num_colors = len(colors)
        text_len = len(text)
        
        for i, char in enumerate(text):
            if text_len == 1:
                color_idx = 0
            else:
                # Calculate color index based on position
                offset = (phase / (num_colors * text_len)) if phase else 0
                position = i / (text_len - 1) if text_len > 1 else 0
                if reversed_:
                    position = 1 - position
                
                color_idx = int((position + offset) * num_colors) % num_colors
            
            char_component = {"text": char, "color": colors[color_idx]}
            
            # Add styles
            for style_key, style_value in styles.items():
                if style_value is not None:
                    char_component[style_key] = style_value
            
            result.append(char_component)
        
        return result
    
    def _tokenize(self, text: str) -> list:
        """
        Tokenize MiniMessage string into text and tag tokens.
        
        Args:
            text: The MiniMessage string
            
        Returns:
            List of tuples (token_type, content) where token_type is 'text' or 'tag'
        """
        tokens = []
        i = 0
        current_text = []
        
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text) and text[i + 1] in '<\\\'"':
                # Escape sequence
                current_text.append(text[i + 1])
                i += 2
            elif text[i] == '<':
                # Flush current text
                if current_text:
                    tokens.append(('text', ''.join(current_text)))
                    current_text = []
                
                # Find tag end
                tag_end = self._find_tag_end(text, i)
                if tag_end != -1:
                    tag_content = text[i + 1:tag_end - 1]
                    tokens.append(('tag', tag_content))
                    i = tag_end
                else:
                    current_text.append('<')
                    i += 1
            else:
                current_text.append(text[i])
                i += 1
        
        # Flush remaining text
        if current_text:
            tokens.append(('text', ''.join(current_text)))
        
        return tokens
    
    def _find_tag_end(self, text: str, start: int) -> int:
        """Find the closing > of a tag, handling nested brackets and quotes."""
        depth = 0
        in_quotes = False
        quote_char = None
        
        for i in range(start, len(text)):
            if text[i] in '"\'':
                if not in_quotes:
                    in_quotes = True
                    quote_char = text[i]
                elif text[i] == quote_char:
                    in_quotes = False
                    quote_char = None
            elif not in_quotes:
                if text[i] == '<':
                    depth += 1
                elif text[i] == '>':
                    depth -= 1
                    if depth == 0:
                        return i + 1
        return -1
    
    def _parse_tokens(self, tokens: list, index: int, parent: TextComponent, stop_at_end_tag: Optional[str] = None) -> tuple[TextComponent, int]:
        """
        Parse tokens into a TextComponent.
        
        Args:
            tokens: List of tokens
            index: Current token index
            parent: Parent component to inherit styles from
            stop_at_end_tag: If set, stop parsing when encountering this end tag
            
        Returns:
            Tuple of (TextComponent, next_token_index)
        """
        result = parent.copy_styles()
        segments = []
        current_text_parts = []
        
        while index < len(tokens):
            token_type, token_content = tokens[index]
            
            if token_type == 'text':
                current_text_parts.append(token_content)
                index += 1
            
            elif token_type == 'tag':
                # Flush current text first
                if current_text_parts:
                    text_str = ''.join(current_text_parts)
                    if text_str:
                        # Add as a segment with current styles
                        segment = result.copy_styles()
                        segment.text = text_str
                        segments.append(segment.to_dict())
                    current_text_parts = []
                
                # Parse tag
                tag_name, tag_args = self._parse_tag(token_content)
                
                # Check for end tag
                if tag_name.startswith('/'):
                    end_tag_name = tag_name[1:]
                    if stop_at_end_tag and end_tag_name.lower() == stop_at_end_tag.lower():
                        index += 1
                        break
                    index += 1
                    continue
                
                # Check for auto-closing tag
                if token_content.endswith('/'):
                    self._apply_tag(result, tag_name, tag_args)
                    # Handle fallback for auto-closing unsupported tags
                    if result.fallback_text:
                        segments.append({"text": result.fallback_text})
                        result.fallback_text = None  # Reset
                    index += 1
                    continue
                
                # Check for invert decoration
                if tag_name.startswith('!'):
                    self._apply_invert_decoration(result, tag_name[1:])
                    index += 1
                    continue
                
                # It's a start tag - find matching end tag
                styled_component = result.copy_styles()
                self._apply_tag(styled_component, tag_name, tag_args)
                
                # Check if this is a fallback tag (unsupported)
                if styled_component.fallback_text:
                    # For unsupported tags, output as plain text
                    fallback_str = styled_component.fallback_text
                    
                    # Parse inner content
                    inner_result, index = self._parse_tokens(tokens, index + 1, styled_component, tag_name)
                    
                    # Prepend fallback tag to inner content
                    if inner_result.text:
                        inner_result.text = fallback_str + inner_result.text
                    elif inner_result.extra:
                        # Add fallback as first extra element
                        inner_result.extra.insert(0, {"text": fallback_str})
                    else:
                        # No inner content, just show the tag
                        inner_result.text = fallback_str
                    
                    if inner_result.text or inner_result.extra:
                        segments.append(inner_result.to_dict())
                else:
                    # Parse inner content
                    inner_result, index = self._parse_tokens(tokens, index + 1, styled_component, tag_name)
                    
                    # Add inner result as segment
                    if inner_result.text or inner_result.extra:
                        segments.append(inner_result.to_dict())
            
            else:
                index += 1
        
        # Flush remaining text
        if current_text_parts:
            text_str = ''.join(current_text_parts)
            if text_str:
                segment = result.copy_styles()
                segment.text = text_str
                segments.append(segment.to_dict())
        
        # Combine segments intelligently
        if len(segments) == 1:
            # Single segment - use it directly
            single = TextComponent()
            single.text = segments[0].get("text", "")
            single.color = segments[0].get("color")
            single.bold = segments[0].get("bold")
            single.italic = segments[0].get("italic")
            single.underlined = segments[0].get("underlined")
            single.strikethrough = segments[0].get("strikethrough")
            single.obfuscated = segments[0].get("obfuscated")
            single.click_event = segments[0].get("clickEvent")
            single.hover_event = segments[0].get("hoverEvent")
            single.insertion = segments[0].get("insertion")
            single.keybind = segments[0].get("keybind")
            single.translate = segments[0].get("translate")
            single.with_ = segments[0].get("with")
            single.font = segments[0].get("font")
            single.shadow_color = segments[0].get("shadow_color")
            # Preserve animation properties
            if "gradient" in segments[0]:
                single.gradient_colors = segments[0]["gradient"].get("colors")
                single.animation_type = segments[0]["gradient"].get("type")
                single.animation_phase = segments[0]["gradient"].get("phase")
                single.animation_reversed = segments[0]["gradient"].get("reversed", False)
            if "extra" in segments[0]:
                single.extra = segments[0]["extra"]
            result = single
        elif len(segments) > 1:
            # Multiple segments - combine into one with extras
            result.extra = segments
        
        return result, index
    
    def _parse_tag(self, tag_content: str) -> tuple[str, list[str]]:
        """
        Parse tag content into name and arguments.
        
        Args:
            tag_content: The content inside the tag brackets
            
        Returns:
            Tuple of (tag_name, list_of_arguments)
        """
        tag_content = tag_content.strip().rstrip('/')
        
        if ':' not in tag_content:
            return tag_content.lower(), []
        
        parts = []
        current = []
        in_quotes = False
        quote_char = None
        
        for char in tag_content:
            if char in '"\'':
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current.append(char)
            elif not in_quotes:
                if char == ':':
                    parts.append(''.join(current))
                    current = []
                else:
                    current.append(char)
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        if not parts:
            return tag_content.lower(), []
        
        return parts[0].lower(), parts[1:]
    
    def _apply_tag(self, component: TextComponent, tag_name: str, args: list[str]) -> None:
        """Apply a tag to a component."""
        
        # Color tags - check for hex color first (tag name starts with #)
        if tag_name.startswith('#'):
            component.color = tag_name.upper()
        
        elif tag_name in COLOR_NAMES:
            component.color = tag_name
        
        elif tag_name in ('color', 'colour', 'c'):
            if args:
                color_arg = args[0].strip('"\'')
                if color_arg.startswith('#'):
                    component.color = color_arg.upper()
                elif color_arg.lower() in COLOR_NAMES:
                    component.color = color_arg.lower()
        
        # Shadow color
        elif tag_name == 'shadow':
            if args:
                shadow_color = args[0].strip('"\'')
                if len(args) > 1:
                    alpha = float(args[1])
                    component.shadow_color = f"{shadow_color}:{alpha}"
                else:
                    component.shadow_color = shadow_color
        
        # Decorations
        elif tag_name in DECORATION_MAP:
            decor = DECORATION_MAP[tag_name]
            setattr(component, decor, True)
        
        # Click event
        elif tag_name == 'click' and len(args) >= 2:
            action = args[0].lower()
            value = ':'.join(args[1:]).strip('"\'')
            component.click_event = {"action": action, "value": value}
        
        # Hover event
        elif tag_name == 'hover' and len(args) >= 2:
            action = args[0].lower()
            value = ':'.join(args[1:]).strip('"\'')
            
            if action == 'show_text':
                # Parse hover text recursively
                hover_parser = MiniMessageParser(strict=self.strict)
                hover_result = hover_parser.parse(value)
                component.hover_event = {"action": "show_text", "contents": hover_result}
            elif action == 'show_item':
                component.hover_event = {"action": "show_item", "contents": {"type": value}}
            elif action == 'show_entity':
                component.hover_event = {"action": "show_entity", "contents": {"type": value}}
        
        # Keybind
        elif tag_name == 'key' and args:
            component.keybind = args[0].strip('"\'')
        
        # Translatable
        elif tag_name in ('lang', 'tr', 'translate') and args:
            component.translate = args[0].strip('"\'')
            if len(args) > 1:
                component.with_ = [arg.strip('"\'') for arg in args[1:]]
        
        # Lang with fallback
        elif tag_name in ('lang_or', 'tr_or', 'translate_or') and len(args) >= 2:
            component.translate = args[0].strip('"\'')
            component.text = args[1].strip('"\'')
            if len(args) > 2:
                component.with_ = [arg.strip('"\'') for arg in args[2:]]
        
        # Insertion
        elif tag_name == 'insert' and args:
            component.insertion = args[0].strip('"\'')
        
        # Font
        elif tag_name == 'font' and args:
            component.font = args[0].strip('"\'')
        
        # Newline
        elif tag_name in ('newline', 'br'):
            component.text = '\n'
        
        # Selector
        elif tag_name in ('selector', 'sel') and args:
            selector = args[0].strip('"\'')
            separator = args[1].strip('"\'') if len(args) > 1 else None
            selector_comp = {"selector": selector}
            if separator:
                selector_comp["separator"] = separator
            component.extra.append(selector_comp)
        
        # Score
        elif tag_name == 'score' and len(args) >= 2:
            component.extra.append({
                "score": {"name": args[0].strip('"\''), "objective": args[1].strip('"\'')}
            })
        
        # NBT
        elif tag_name in ('nbt', 'data') and len(args) >= 3:
            nbt_type = args[0].lower()
            nbt_id = args[1].strip('"\'')
            nbt_path = args[2].strip('"\'')
            interpret = 'interpret' in args
            
            nbt_data = {"nbt": nbt_path}
            if interpret:
                nbt_data["interpret"] = True
            
            if nbt_type == 'block':
                nbt_data["block"] = nbt_id
            elif nbt_type == 'entity':
                nbt_data["entity"] = nbt_id
            elif nbt_type == 'storage':
                nbt_data["storage"] = nbt_id
            
            component.extra.append(nbt_data)
        
        # Gradient
        elif tag_name == 'gradient':
            colors = []
            phase = None
            
            for arg in args:
                arg = arg.strip('"\'')
                # Check for phase (numeric value)
                try:
                    phase = float(arg)
                    continue
                except ValueError:
                    pass
                
                # Resolve color
                if arg.startswith('#'):
                    colors.append(arg.upper())
                elif arg.lower() in COLOR_NAMES:
                    colors.append(COLOR_NAMES[arg.lower()])
            
            # Default to white if no colors specified
            if not colors:
                colors = ["#FFFFFF"]
            
            component.animation_type = "gradient"
            component.gradient_colors = colors
            if phase is not None:
                component.animation_phase = phase
        
        # Rainbow
        elif tag_name == 'rainbow':
            component.animation_type = "rainbow"
            component.gradient_colors = RAINBOW_COLORS.copy()
            component.animation_phase = None
            component.animation_reversed = False
            
            # Handle phase and reversal
            for arg in args:
                arg = arg.strip('"\'')
                # Check for reverse flag
                if arg.startswith('!'):
                    component.animation_reversed = True
                    arg = arg[1:]
                
                # Check for phase
                try:
                    component.animation_phase = float(arg)
                except ValueError:
                    pass
        
        # Pride
        elif tag_name == 'pride':
            flag_type = "pride"
            phase = None
            
            for arg in args:
                arg = arg.strip('"\'')
                # Check for phase (numeric value)
                try:
                    phase = float(arg)
                    continue
                except ValueError:
                    pass
                
                # Check for flag type
                if arg.lower() in PRIDE_FLAGS:
                    flag_type = arg.lower()
            
            component.animation_type = "pride"
            component.gradient_colors = PRIDE_FLAGS.get(flag_type, PRIDE_FLAGS["pride"]).copy()
            if phase is not None:
                component.animation_phase = phase
        
        # Transition
        elif tag_name == 'transition':
            colors = []
            phase = None
            
            for arg in args:
                arg = arg.strip('"\'')
                # Check for phase (numeric value)
                try:
                    phase = float(arg)
                    continue
                except ValueError:
                    pass
                
                # Resolve color
                if arg.startswith('#'):
                    colors.append(arg.upper())
                elif arg.lower() in COLOR_NAMES:
                    colors.append(COLOR_NAMES[arg.lower()])
            
            if not colors:
                colors = ["#FFFFFF"]
            
            component.animation_type = "transition"
            component.gradient_colors = colors
            if phase is not None:
                component.animation_phase = phase
        
        # Sprite
        elif tag_name == 'sprite' and args:
            sprite_id = args[-1].strip('"\'')
            atlas = args[0].strip('"\'') if len(args) > 1 else "minecraft:blocks"
            component.extra.append({"sprite": {"atlas": atlas, "sprite": sprite_id}})
        
        # Head
        elif tag_name == 'head' and args:
            head_arg = args[0].strip('"\'')
            outer_layer = True
            if len(args) > 1:
                outer_layer = args[1].strip('"\'').lower() == 'true'
            
            head_data = {"outer_layer": outer_layer}
            if '-' in head_arg:
                head_data["uuid"] = head_arg
            elif '/' in head_arg:
                head_data["texture"] = head_arg
            else:
                head_data["name"] = head_arg
            
            component.extra.append({"head": head_data})
        
        # Reset
        elif tag_name == 'reset':
            component.color = None
            component.bold = None
            component.italic = None
            component.underlined = None
            component.strikethrough = None
            component.obfuscated = None
            component.click_event = None
            component.hover_event = None
            component.insertion = None
            component.keybind = None
            component.translate = None
            component.with_ = None
            component.font = None
            component.shadow_color = None
            component.gradient_colors = None
            component.animation_type = None
            component.animation_phase = None
            component.animation_reversed = False
        
        else:
            # Unknown/unsupported tag - use fallback
            tag_full = f"<{tag_name}"
            if args:
                tag_full += ":" + ":".join(args)
            tag_full += ">"
            component.fallback_text = tag_full
            logger.warning(f"[MiniMessage] Unsupported tag '{tag_name}' - falling back to plain text")
    
    def _apply_invert_decoration(self, component: TextComponent, decor_name: str) -> None:
        """Apply inverted decoration."""
        if decor_name in DECORATION_MAP:
            decor = DECORATION_MAP[decor_name]
            current = getattr(component, decor)
            setattr(component, decor, not current if current is not None else True)


def parse_mini_message(input_string: str, strict: bool = False) -> dict:
    """
    Parse a MiniMessage string into Minecraft JSON text component.
    
    Args:
        input_string: The MiniMessage formatted string
        strict: If True, enforce strict mode parsing
        
    Returns:
        A dictionary representing the Minecraft JSON text component
        
    Examples:
        >>> parse_mini_message("<green>Mini Message")
        {'text': 'Mini Message', 'color': 'green'}
        
        >>> parse_mini_message("<yellow>Hello <blue>World!</blue>")
        {'text': '', 'extra': [{'text': 'Hello ', 'color': 'yellow', 'extra': [{'text': 'World!', 'color': 'blue'}]}]}
    """
    parser = MiniMessageParser(strict=strict)
    return parser.parse(input_string)


def mini_message_to_json(input_string: str, **kwargs) -> str:
    """
    Convert MiniMessage string to Minecraft JSON text component string.
    
    Args:
        input_string: The MiniMessage formatted string
        **kwargs: Additional arguments passed to json.dumps()
        
    Returns:
        JSON string representation of the Minecraft text component
    """
    result = parse_mini_message(input_string)
    return json.dumps(result, ensure_ascii=False, **kwargs)


if __name__ == "__main__":
    # Test examples from MiniMessage spec
    test_cases = [
        # Basic colors
        ("<green>Mini Message", "Simple color tag"),
        ("<yellow>Hello <blue>World</blue>!", "Nested color tags"),
        ("<red>This is a <green>test!", "Unclosed nested tag"),
        ("<#00ff00>R G B!", "Hex color"),
        
        # Decorations
        ("<bold>Bold text</bold>", "Bold decoration"),
        ("<underlined>This is <bold>important</bold>!", "Multiple decorations"),
        
        # Events
        ("<click:run_command:/seed>Click me</click>", "Click event"),
        ("<hover:show_text:'<red>Tooltip'>Hover me</hover>", "Hover event"),
        
        # Verbose color
        ("<color:yellow>Hello</color>", "Verbose color tag"),
        ("<color:#FF5555>Red text</color>", "Verbose hex color"),
        
        # Shadow color
        ("<shadow:yellow>Shadow text</shadow>", "Shadow color"),
        ("<shadow:#FF5555:0.5>Shadow with alpha</shadow>", "Shadow with alpha"),
        
        # Rainbow
        ("<rainbow>Rainbow text</rainbow>", "Rainbow animation"),
        ("<rainbow:2>Rainbow with phase</rainbow>", "Rainbow with phase"),
        ("<rainbow:!2>Rainbow reversed</rainbow>", "Rainbow reversed"),
        
        # Gradient
        ("<gradient:#5e4fa2:#f79459>Gradient text</gradient>", "Gradient with two colors"),
        ("<gradient:#5e4fa2:#f79459:red>Gradient with three colors</gradient>", "Gradient with three colors"),
        ("<gradient:green:blue>Gradient with named colors</gradient>", "Gradient with named colors"),
        
        # Transition
        ("<transition:#00ff00:#ff0000:0>Transition text</transition>", "Color transition"),
        ("<transition:white:black:red>Transition multi-color</transition>", "Multi-color transition"),
        
        # Pride flags
        ("<pride>Pride text</pride>", "Pride flag default"),
        ("<pride:trans>Trans pride</pride>", "Trans pride flag"),
        ("<pride:bi>Bi pride</pride>", "Bi pride flag"),
        ("<pride:pan>Pan pride</pride>", "Pan pride flag"),
        ("<pride:lesbian>Lesbian pride</pride>", "Lesbian pride flag"),
        ("<pride:gay>Gay pride</pride>", "Gay pride flag"),
    ]
    
    print("=" * 70)
    print("MiniMessage Parser Tests")
    print("=" * 70)
    
    for test_input, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input:  {test_input}")
        try:
            result = parse_mini_message(test_input)
            print(f"  Output: {json.dumps(result, ensure_ascii=False)}")
        except Exception as e:
            print(f"  Error:  {e}")
    
    print("\n" + "=" * 70)
