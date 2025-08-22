from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import OpenImageIO as oiio



def create_single_tile(tile_size, color, x_coord, y_coord, font_path=None):
    """
    Creates a single square tile image with a given color and two lines of centered text.

    Args:
        tile_size (int): The sidelength of the square tile in pixels.
        color (tuple): A tuple of three floats (R, G, B) representing the color (0.0-1.0).
        x_coord (int): The X coordinate to display on the tile.
        y_coord (int): The Y coordinate to display on the tile.
        font_path (str, optional): The path to a TrueType font file (.ttf).
                                   If None, a default PIL font will be used.
    Returns:
        PIL.Image.Image: The created tile image with text.
    """
    # Convert float color (0.0-1.0) to 8-bit integer color (0-255)
    int_color = tuple(int(max(0, min(1, c)) * 255) for c in color) # Ensure color components are clamped

    img = Image.new('RGB', (tile_size, tile_size), int_color)
    draw = ImageDraw.Draw(img)

    text_line1 = f"x = {x_coord}"
    text_line2 = f"y = {y_coord}"

    text_fill_color = (255, 255, 255)

    # --- Dynamic Font Size Adjustment ---
    # Start with a relatively large font size and shrink if needed
    font_size = int(tile_size * 0.25) # Initial guess for font size
    max_font_size = int(tile_size * 0.25) # Don't exceed this

    font = None
    max_iterations = 100 # Prevent infinite loops in font size reduction

    for _ in range(max_iterations):
        current_font_path = font_path
        current_font_size = max(1, font_size) # Ensure font size is at least 1

        try:
            if current_font_path and os.path.exists(current_font_path):
                font = ImageFont.truetype(current_font_path, current_font_size)
            else:
                # Fallback to default font (size argument might not always work perfectly)
                font = ImageFont.load_default()
                # For default font, try to scale if load_default(size=...) is supported and works
                try:
                    scaled_font = ImageFont.load_default(size=current_font_size)
                    if draw.textbbox((0, 0), text_line1, font=scaled_font)[2] > 0: # Check if usable
                        font = scaled_font
                except Exception:
                    pass # Stick with original default font

            if font is None: # Last resort if no font could be loaded
                font = ImageFont.load_default()

            # Measure text dimensions
            bbox1 = draw.textbbox((0, 0), text_line1, font=font)
            text_width1 = bbox1[2] - bbox1[0]
            text_height1 = bbox1[3] - bbox1[1]

            bbox2 = draw.textbbox((0, 0), text_line2, font=font)
            text_width2 = bbox2[2] - bbox2[0]
            text_height2 = bbox2[3] - bbox2[1]

            # Calculate total height needed for both lines plus some padding
            # Let's assume a small gap between lines (e.g., 0.1 * text_height)
            line_gap = int(text_height1 * 0.2) # 20% of line height
            total_text_height = text_height1 + text_height2 + line_gap

            # Check if text fits vertically and horizontally
            if (total_text_height < tile_size * 0.9) and \
               (text_width1 < tile_size * 0.9) and \
               (text_width2 < tile_size * 0.9):
                break # Font size is good, break out of loop
            else:
                font_size -= 1 # Reduce font size
                if font_size <= 0: # Prevent infinite loop if text can never fit
                    font_size = 1 # Smallest possible font size
                    break

        except Exception as e:
            # Handle cases where font loading or textbbox fails
            print(f"Error during font sizing: {e}. Reducing font size and retrying.")
            font_size -= 1
            if font_size <= 0:
                font_size = 1
                break # Cannot make font smaller, stop

    # Final check: if font_size became 0 or less, ensure it's at least 1
    if font_size <= 0:
        font_size = 1
        # Reload font with minimum size if needed
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
            try:
                scaled_font = ImageFont.load_default(size=font_size)
                if draw.textbbox((0, 0), text_line1, font=scaled_font)[2] > 0:
                    font = scaled_font
            except Exception:
                pass


    # Re-measure with final font size to ensure accurate positioning
    bbox1 = draw.textbbox((0, 0), text_line1, font=font)
    text_width1 = bbox1[2] - bbox1[0]
    text_height1 = bbox1[3] - bbox1[1]

    bbox2 = draw.textbbox((0, 0), text_line2, font=font)
    text_width2 = bbox2[2] - bbox2[0]
    text_height2 = bbox2[3] - bbox2[1]

    # Calculate positions for centering
    # Line 1: centered horizontally, midpoint at 1/3 tile height
    x1 = (tile_size - text_width1) / 2
    y1 = (tile_size / 3) - (text_height1 / 2)

    # Line 2: centered horizontally, midpoint at 2/3 tile height
    x2 = (tile_size - text_width2) / 2
    y2 = (tile_size * 2 / 3) - (text_height2 / 2)

    # Draw the text
    draw.text((x1, y1), text_line1, fill=text_fill_color, font=font)
    draw.text((x2, y2), text_line2, fill=text_fill_color, font=font)

    return img

def generate_interpolated_grid_image(tile_size, count, font_path=None):
    """
    Generates a large image composed of 'count' x 'count' tiles,
    with colors bilinearly interpolated from corners and text indicating tile index.

    Args:
        tile_size (int): The sidelength of each individual square tile in pixels.
        count (int): The number of tiles per side of the large grid (e.g., if count=3,
                     it's a 3x3 grid of tiles).
        font_path (str, optional): Path to a TrueType font file for the tile text.
                                   If None, a default PIL font will be used.

    Returns:
        PIL.Image.Image: The generated large grid image.
    """
    if count <= 0:
        raise ValueError("Count must be a positive integer.")

    total_image_size = count * tile_size
    main_img = Image.new('RGB', (total_image_size, total_image_size))

    # Corner colors (R, G, B) as floats (0.0-1.0)
    corner_colors = {
        "top_left": (1.0, 0.0, 0.0),    # Red
        "top_right": (1.0, 0.0, 1.0),   # Purple
        "bottom_left": (0.0, 1.0, 0.0), # Green
        "bottom_right": (0.0, 0.0, 1.0) # Blue
    }

    # Handle the edge case where count is 1
    if count == 1:
        # If count is 1, there's only one tile, which is the top-left corner
        tile_color = corner_colors["top_left"]
        tile_image = create_single_tile(tile_size, tile_color, 0, 0, font_path=font_path)
        main_img.paste(tile_image, (0, 0))
        return main_img

    for y_tile in range(count):
        for x_tile in range(count):
            # Calculate normalized coordinates (u, v) for interpolation
            # We divide by (count - 1) to ensure 0 and 1 values at the edges
            u = x_tile / (count - 1)
            v = y_tile / (count - 1)

            # Apply the simplified bilinear interpolation formulas
            r_component = 1 - v
            g_component = v * (1 - u)
            b_component = u

            # Clamp components to be within 0.0 and 1.0 (due to potential floating point inaccuracies)
            current_color = (
                max(0.0, min(1.0, r_component)),
                max(0.0, min(1.0, g_component)),
                max(0.0, min(1.0, b_component))
            )

            # Create the individual tile
            tile_image = create_single_tile(tile_size, current_color, x_tile, y_tile, font_path=font_path)

            # Paste the tile onto the main image
            paste_x = x_tile * tile_size
            paste_y = y_tile * tile_size
            main_img.paste(tile_image, (paste_x, paste_y))

    return main_img




import argparse
parser = argparse.ArgumentParser(description="Process two optional named parameters.")
parser.add_argument('--ts', type=int, default=128, help='Tile Size')
parser.add_argument('--gs', type=int, default=128, help='Grid Size')

# Parse the arguments
args = parser.parse_args()


# --- Configuration ---
tile_sidelength = args.ts  # Size of each individual tile in pixels
grid_count = args.gs      # Number of tiles per side (e.g., 15 means 15x15 grid)

# Path to a font file (adjust this for your system)
# On Windows, you can typically use 'C:/Windows/Fonts/arial.ttf' or similar
# You might need to find a suitable font on your system.
# For testing, you can use None to let PIL use its default font.
# If a specific font path is provided and doesn't exist, it will fall back to default.
windows_font_path = "C:/Windows/Fonts/arial.ttf" # Example path for Windows
# If Arial is not found, try Times New Roman:
# windows_font_path = "C:/Windows/Fonts/times.ttf"

font_to_use = None
if os.name == 'nt': # Check if OS is Windows
    if os.path.exists(windows_font_path):
        font_to_use = windows_font_path
        print(f"Using font: {windows_font_path}")
    else:
        print(f"Warning: Windows font not found at '{windows_font_path}'. Using default PIL font.")
else: # Assume Linux/macOS for other OS types
    # Common Linux/macOS font paths (adjust as needed)
    linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    mac_font_path = "/Library/Fonts/Arial.ttf"
    if os.path.exists(linux_font_path):
        font_to_use = linux_font_path
        print(f"Using font: {linux_font_path}")
    elif os.path.exists(mac_font_path):
        font_to_use = mac_font_path
        print(f"Using font: {mac_font_path}")
    else:
        print("Warning: No common Linux/macOS font found. Using default PIL font.")


# --- Generate and save the image ---
print(f"Generating a {grid_count}x{grid_count} grid of tiles, each {tile_sidelength}x{tile_sidelength} pixels.")
print(f"Total image size will be {grid_count * tile_sidelength}x{grid_count * tile_sidelength} pixels.")

try:
    final_image = generate_interpolated_grid_image(tile_sidelength, grid_count, font_path=font_to_use)
    output_filename = "../../media/tiled_grid_mip_0.exr"
    np_img = np.array(final_image).astype(np.float32) / 255.0  # Normalize for EXR
    spec = oiio.ImageSpec(final_image.width, final_image.height, 3, oiio.TypeDesc("float"))
    out = oiio.ImageOutput.create(output_filename)
    out.open(output_filename, spec)
    out.write_image(np_img.reshape(-1))  # Flatten for OIIO’s expected input
    out.close()

    print(f"Successfully created '{output_filename}'")

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")