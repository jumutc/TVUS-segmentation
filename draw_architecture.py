"""
UNet++ Architecture Information Flow Diagram

Simplified visualization showing:
- Replaceable backbone encoder (e.g., InceptionResNetV2, EfficientNet, ResNet)
- Decoder with dense skip connections
- Final segmentation mask output

Usage:
    pip install plotly numpy
    python draw_architecture.py
"""
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np

# ============================================================================
# CONFIGURATION: Replaceable Backbone Encoder
# ============================================================================
BACKBONE = "InceptionResNetV2"  # Options: InceptionResNetV2, EfficientNet-B7, ResNet50, etc.

# ============================================================================
# ARCHITECTURE CONSTANTS
# ============================================================================
NUM_LEVELS = 4  # Simplified: 4 levels

# ============================================================================
# DATA STRUCTURES
# ============================================================================
nodes = []  # (x, y, label, color, node_type, key, size)
links = []  # (src_key, dst_key, link_type)

# Node types
NODE_ENCODER = "encoder"
NODE_DECODER = "decoder"
NODE_OUTPUT = "output"

# Link types
LINK_ENCODER_CHAIN = "encoder_chain"  # Vertical connections within encoder
LINK_ENCODER_DECODER = "encoder_decoder"
LINK_DENSE_SKIP = "dense_skip"
LINK_DECODER_OUTPUT = "decoder_output"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_node_info(key):
    """Get node information by key."""
    for node in nodes:
        if node[5] == key:  # key is at index 5
            return node
    return None


def get_node_pos(key):
    """Get node position by key."""
    node = get_node_info(key)
    if node:
        return (node[0], node[1])
    return None


def add_node(x, y, label, color, node_type, key, size):
    """Add a node to the diagram."""
    nodes.append((x, y, label, color, node_type, key, size))


def add_link(src_key, dst_key, link_type):
    """Add a link between two nodes."""
    links.append((src_key, dst_key, link_type))


# ============================================================================
# BUILD ARCHITECTURE
# ============================================================================
def build_encoder():
    """Build simplified encoder column."""
    encoder_color = qualitative.Plotly[0]  # Blue
    node_size = 75  # Enlarged for encoder
    
    for level in range(NUM_LEVELS):
        if level == 0:
            # Split long backbone name
            if len(BACKBONE) > 15:
                backbone_parts = BACKBONE.split('ResNet')
                if len(backbone_parts) > 1:
                    label = f"{backbone_parts[0]}<br>ResNet<br>Encoder<br>Block {level}"
                else:
                    label = f"{BACKBONE[:12]}<br>Encoder<br>Block {level}"
            else:
                label = f"{BACKBONE}<br>Encoder<br>Block {level}"
        else:
            label = f"Encoder<br>Features<br>Block {level}"
        
        x_pos = 1.0
        y_pos = (NUM_LEVELS - 1 - level) * 1.8 + 1.0
        key = f"E{level}"
        add_node(x_pos, y_pos, label, encoder_color, NODE_ENCODER, key, node_size)


def build_decoder():
    """Build simplified decoder."""
    decoder_color = qualitative.Plotly[1]  # Orange
    start_x = 3.5
    node_size = 60  # Enlarged for decoder nodes
    
    for row in range(NUM_LEVELS):
        for col in range(NUM_LEVELS - row):
            x_pos = start_x + col * 1.5
            y_pos = (NUM_LEVELS - 1 - row) * 1.8 + 1.0
            
            label = f"Decoder<br>X<sup>{row}</sup><sub>{col}</sub>"
            key = f"X{row}_{col}"
            add_node(x_pos, y_pos, label, decoder_color, NODE_DECODER, key, node_size)


def build_output():
    """Build segmentation mask output."""
    output_color = qualitative.Plotly[2]  # Green
    node_size = 140  # Reduced by 1.5x (was 210, now 210/1.5 = 140)
    
    # Position closer to decoder blocks
    x_pos = 3.5 + (NUM_LEVELS - 1) * 1.5 + 0.8  # Reduced from 1.8 to 0.8
    y_pos = 1.0
    
    label = "Segmentation<br>Mask<br>Output"
    key = "output"
    add_node(x_pos, y_pos, label, output_color, NODE_OUTPUT, key, node_size)


def build_connections():
    """Build all connections."""
    # Encoder chain: E0 → E1 → E2 → E3 (vertical connections)
    for level in range(NUM_LEVELS - 1):
        add_link(f"E{level}", f"E{level+1}", LINK_ENCODER_CHAIN)
    
    # Encoder to decoder (row 0) - skip connections
    for level in range(NUM_LEVELS):
        add_link(f"E{level}", f"X0_{level}", LINK_ENCODER_DECODER)
    
    # Vertical connections within decoder: X^{r-1}_c → X^r_c
    # This ensures all decoder nodes receive inputs from above (no deadends)
    for row in range(1, NUM_LEVELS):
        for col in range(NUM_LEVELS - row):
            add_link(f"X{row-1}_{col}", f"X{row}_{col}", LINK_DENSE_SKIP)
    
    # Dense skip connections within decoder
    # Connect X^r_{c+1} to X^{r+1}_c (diagonal dense connections)
    for row in range(NUM_LEVELS - 1):
        for col in range(NUM_LEVELS - row - 1):
            add_link(f"X{row}_{col+1}", f"X{row+1}_{col}", LINK_DENSE_SKIP)
    
    # Decoder to output
    add_link(f"X{NUM_LEVELS-1}_0", "output", LINK_DECODER_OUTPUT)


# ============================================================================
# VISUALIZATION
# ============================================================================
def get_link_color(link_type):
    """Get color for link types."""
    return {
        LINK_ENCODER_CHAIN: 'rgba(70, 130, 180, 0.7)',  # Steel blue for encoder chain
        LINK_ENCODER_DECODER: 'rgba(100, 149, 237, 0.6)',
        LINK_DENSE_SKIP: 'rgba(255, 140, 0, 0.5)',
        LINK_DECODER_OUTPUT: 'rgba(50, 205, 50, 0.7)',
    }.get(link_type, 'gray')


def calculate_arrow_endpoints(src_key, dst_key, link_type):
    """Calculate arrow start and end points at node edges, ensuring arrows don't enter shape bounds."""
    src_node = get_node_info(src_key)
    dst_node = get_node_info(dst_key)
    
    if not src_node or not dst_node:
        return None, None, None
    
    src_x, src_y = src_node[0], src_node[1]
    dst_x, dst_y = dst_node[0], dst_node[1]
    src_size = src_node[6]  # size is at index 6 (diameter in pixels)
    dst_size = dst_node[6]
    
    # Convert pixel size to approximate data coordinate radius
    # For a canvas of ~1100px width and ~8-9 data units, roughly 110px per data unit
    # Node sizes are diameters, so divide by 2 for radius, then convert to data coords
    # Add padding proportional to node size to ensure arrows are clearly outside the shape bounds
    # Larger nodes need significantly more padding
    base_padding = 0.12
    # Scale padding more aggressively for larger nodes (output node is 140px)
    src_padding = base_padding * (1 + src_size / 100.0)  # More padding for larger nodes
    dst_padding = base_padding * (1 + dst_size / 100.0)
    src_radius = (src_size / 2.0) / 110.0 + src_padding
    dst_radius = (dst_size / 2.0) / 110.0 + dst_padding
    
    # Calculate direction vector
    dx = dst_x - src_x
    dy = dst_y - src_y
    dist = (dx**2 + dy**2)**0.5
    
    if dist == 0:
        return None, None, None
    
    # Normalize direction
    dx_norm = dx / dist
    dy_norm = dy / dist
    
    # Calculate edge points - ensure they're outside the node bounds
    start_x = src_x + dx_norm * src_radius
    start_y = src_y + dy_norm * src_radius
    end_x = dst_x - dx_norm * dst_radius
    end_y = dst_y - dy_norm * dst_radius
    
    # For encoder-to-decoder connections, use curved paths to avoid crossing
    if link_type == LINK_ENCODER_DECODER:
        # Calculate a curved path that goes around decoder nodes
        # Use a control point offset to create a curve
        mid_x = (start_x + end_x) / 2.0
        mid_y = (start_y + end_y) / 2.0
        
        # Offset the control point to create an arc
        # For connections going right, offset upward/downward based on level
        level = int(src_key[1]) if src_key.startswith('E') else 0
        offset = 0.3 * (1 if level % 2 == 0 else -1)  # Alternate offset direction
        
        # Perpendicular direction for offset
        perp_x = -dy_norm * offset
        perp_y = dx_norm * offset
        
        control_x = mid_x + perp_x
        control_y = mid_y + perp_y
        
        return (start_x, start_y), (end_x, end_y), (control_x, control_y)
    else:
        # Straight arrows for other connections
        return (start_x, start_y), (end_x, end_y), None


def create_diagram():
    """Create and display the simplified architecture diagram."""
    fig = go.Figure()
    
    # Draw arrows using annotations
    for src_key, dst_key, link_type in links:
        start_point, end_point, control_point = calculate_arrow_endpoints(src_key, dst_key, link_type)
        if start_point and end_point:
            if control_point and link_type == LINK_ENCODER_DECODER:
                # Draw curved arrow using a path
                # Create a quadratic Bezier curve
                t = np.linspace(0, 1, 50)
                curve_x = (1-t)**2 * start_point[0] + 2*(1-t)*t * control_point[0] + t**2 * end_point[0]
                curve_y = (1-t)**2 * start_point[1] + 2*(1-t)*t * control_point[1] + t**2 * end_point[1]
                
                # Draw the curve
                fig.add_trace(go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode='lines',
                    line=dict(color=get_link_color(link_type), width=2.5),
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                # Add arrowhead at the end
                # Calculate direction at end point (use points before the actual end to get tangent)
                if len(curve_x) >= 2:
                    dx_end = end_point[0] - curve_x[-2]
                    dy_end = end_point[1] - curve_y[-2]
                    dist_end = (dx_end**2 + dy_end**2)**0.5
                    if dist_end > 0:
                        dx_norm = dx_end / dist_end
                        dy_norm = dy_end / dist_end
                        # Position arrow shaft start slightly before the end point
                        arrow_length = 0.12
                        arrow_x = end_point[0] - dx_norm * arrow_length
                        arrow_y = end_point[1] - dy_norm * arrow_length
                        
                        fig.add_annotation(
                            x=end_point[0],
                            y=end_point[1],
                            ax=arrow_x,
                            ay=arrow_y,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowwidth=2.5,
                            arrowcolor=get_link_color(link_type),
                            showarrow=True
                        )
            else:
                # Straight arrow
                fig.add_annotation(
                    x=end_point[0],
                    y=end_point[1],
                    ax=start_point[0],
                    ay=start_point[1],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=get_link_color(link_type),
                    showarrow=True
                )
    
    # Draw nodes grouped by type
    for node_type in [NODE_ENCODER, NODE_DECODER, NODE_OUTPUT]:
        type_nodes = [(x, y, label, color, size) for x, y, label, color, nt, k, size in nodes if nt == node_type]
        if not type_nodes:
            continue
        
        x_vals, y_vals, labels, colors, sizes = zip(*type_nodes)
        
        # Use the stored node size
        node_size = sizes[0]  # All nodes of same type have same size
        
        node_name = {
            NODE_ENCODER: f"Encoder ({BACKBONE})",
            NODE_DECODER: "Decoder",
            NODE_OUTPUT: "Output"
        }[node_type]
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=list(colors)[0],
                line=dict(width=2, color='white'),
                sizemode='diameter'
            ),
            text=list(labels),
            textposition="middle center",
            textfont=dict(
                size=10 if node_type == NODE_DECODER else (14 if node_type == NODE_OUTPUT else 11),
                color='white',
                family="Arial"
            ),
            name=node_name,
            showlegend=True,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Update layout
    max_x = max(x for x, _, _, _, _, _, _ in nodes) + 1.5
    max_y = max(y for _, y, _, _, _, _, _ in nodes) + 1.0
    
    fig.update_layout(
        title=dict(
            text=f"UNet++ Architecture<br><sub>Backbone: {BACKBONE}</sub>",
            x=0.5,
            font=dict(size=14, family="Arial")
        ),
        xaxis=dict(range=[-0.5, max_x], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-0.5, max_y], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=70, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=10),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1
        ),
        width=1100,
        height=750
    )
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Build and display the simplified UNet++ architecture diagram."""
    print(f"Building simplified UNet++ diagram with backbone: {BACKBONE}")
    
    # Build architecture
    build_encoder()
    build_decoder()
    build_output()
    build_connections()
    
    # Create and show diagram
    fig = create_diagram()
    fig.show()
    
    print(f"✓ Diagram created successfully")


if __name__ == "__main__":
    main()
