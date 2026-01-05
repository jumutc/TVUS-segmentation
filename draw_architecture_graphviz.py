"""
UNet++ Architecture Diagram using Graphviz

Graphviz is excellent for architecture/network diagrams with automatic layout.

Installation:
    pip install graphviz
    # System: sudo apt-get install graphviz (Linux) or brew install graphviz (Mac)

Usage:
    python draw_architecture_graphviz.py
"""
try:
    from graphviz import Digraph
except ImportError:
    print("Error: graphviz not installed. Install with: pip install graphviz")
    print("Also install system Graphviz: sudo apt-get install graphviz")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
BACKBONE = "InceptionResNetV2"
NUM_LEVELS = 4

# ============================================================================
# CREATE DIAGRAM
# ============================================================================
def create_unetpp_diagram():
    """Create UNet++ architecture diagram using Graphviz."""
    
    # Create directed graph
    dot = Digraph(comment='UNet++ Architecture')
    dot.attr(rankdir='LR')  # Left to right
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    dot.attr('graph', bgcolor='white', fontname='Arial', fontsize='12')
    
    # Color scheme
    encoder_color = '#4A90E2'
    decoder_color = '#F5A623'
    output_color = '#7ED321'
    
    # Create encoder nodes
    encoder_nodes = []
    for level in range(NUM_LEVELS):
        if level == 0:
            label = f"{BACKBONE}\nEncoder Block {level}"
        else:
            label = f"Encoder Features\nBlock {level}"
        node_id = f"E{level}"
        dot.node(node_id, label, fillcolor=encoder_color, fontcolor='white')
        encoder_nodes.append(node_id)
    
    # Create decoder nodes
    decoder_nodes = {}
    for row in range(NUM_LEVELS):
        for col in range(NUM_LEVELS - row):
            node_id = f"X{row}_{col}"
            label = f"Decoder\nX^{row}_{col}"
            dot.node(node_id, label, fillcolor=decoder_color, fontcolor='white')
            decoder_nodes[(row, col)] = node_id
    
    # Create output node
    dot.node('output', 'Segmentation\nMask\nOutput', 
             fillcolor=output_color, fontcolor='white', style='rounded,filled,bold')
    
    # Encoder chain connections
    for level in range(NUM_LEVELS - 1):
        dot.edge(f"E{level}", f"E{level+1}", color='#6C7A89', penwidth='2.5')
    
    # Encoder to decoder connections (row 0)
    for level in range(NUM_LEVELS):
        dot.edge(f"E{level}", f"X0_{level}", color=encoder_color, penwidth='2.5')
    
    # Vertical decoder connections
    for row in range(1, NUM_LEVELS):
        for col in range(NUM_LEVELS - row):
            dot.edge(f"X{row-1}_{col}", f"X{row}_{col}", 
                    color=decoder_color, penwidth='2.0')
    
    # Dense skip connections
    for row in range(NUM_LEVELS - 1):
        for col in range(NUM_LEVELS - row - 1):
            dot.edge(f"X{row}_{col+1}", f"X{row+1}_{col}", 
                    color=decoder_color, penwidth='2.0')
    
    # Decoder to output
    dot.edge(f"X{NUM_LEVELS-1}_0", 'output', 
            color=output_color, penwidth='3.0')
    
    # Add title
    dot.attr(label=f'UNet++ Architecture\nBackbone: {BACKBONE}')
    dot.attr(labelloc='t')
    dot.attr(labeljust='c')
    
    return dot


def main():
    """Generate and display the diagram."""
    print(f"Creating UNet++ diagram with Graphviz (backbone: {BACKBONE})")
    
    dot = create_unetpp_diagram()
    
    # Render diagram
    # Options: 'png', 'svg', 'pdf', 'pdf'
    output_file = dot.render('unetpp_architecture', format='png', cleanup=True)
    print(f"✓ Diagram saved as: {output_file}")
    
    # Also create SVG version (scalable)
    svg_file = dot.render('unetpp_architecture', format='svg', cleanup=False)
    print(f"✓ SVG version saved as: {svg_file}")
    
    print("\nNote: Graphviz uses automatic layout algorithms.")
    print("For custom positioning, you may need to use 'neato' or 'fdp' engines.")


if __name__ == "__main__":
    main()
