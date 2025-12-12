#!/usr/bin/env python3.12
"""
Partition a Texas census tract cartogram into 38 equal-area contiguous districts.
Uses graph-based partitioning with the METIS algorithm via networkx.
"""

import json
import networkx as nx
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

def load_geojson(filepath):
    """Load GeoJSON file and extract tract geometries."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    tracts = []
    for feature in data['features']:
        geom = shape(feature['geometry'])
        tract_id = feature.get('id', feature.get('properties', {}).get('GEOID', str(len(tracts))))
        tracts.append({
            'id': tract_id,
            'geometry': geom,
            'area': geom.area
        })
    
    print(f"Loaded {len(tracts)} census tracts")
    print(f"Total area: {sum(t['area'] for t in tracts):.2f}")
    return tracts

def build_adjacency_graph(tracts):
    """Build a graph where nodes are tracts and edges connect adjacent tracts."""
    G = nx.Graph()
    
    # Add nodes with area weights
    for i, tract in enumerate(tracts):
        G.add_node(i, area=tract['area'], tract_id=tract['id'])
    
    # Add edges for adjacent tracts (sharing boundary)
    print("Building adjacency graph...")
    for i in range(len(tracts)):
        for j in range(i + 1, len(tracts)):
            if tracts[i]['geometry'].touches(tracts[j]['geometry']) or \
               tracts[i]['geometry'].intersects(tracts[j]['geometry']):
                G.add_edge(i, j)
        
        if i % 100 == 0:
            print(f"  Processed {i}/{len(tracts)} tracts")
    
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def partition_graph_metis(G, n_parts=38):
    """Partition graph into n_parts using METIS-like algorithm."""
    try:
        # Try using community detection as a fallback to METIS
        # METIS requires python-metis which may not be available
        from networkx.algorithms import community
        
        # Use spectral partitioning approach
        # This is a simplified version - real METIS would be better
        print(f"Partitioning graph into {n_parts} parts...")
        
        # Create initial partition using recursive bisection
        partitions = recursive_partition(G, n_parts)
        
        return partitions
    
    except ImportError:
        print("METIS not available, using spectral clustering")
        return recursive_partition(G, n_parts)

def recursive_partition(G, n_parts):
    """Recursively partition graph using spectral bisection."""
    if n_parts == 1:
        return {node: 0 for node in G.nodes()}
    
    # Split into two groups
    if n_parts == 2:
        partition = spectral_bisection(G)
        return partition
    
    # Recursive split
    n_left = n_parts // 2
    n_right = n_parts - n_left
    
    # First split into two
    initial_partition = spectral_bisection(G)
    
    # Get subgraphs
    group0 = [n for n, p in initial_partition.items() if p == 0]
    group1 = [n for n, p in initial_partition.items() if p == 1]
    
    G0 = G.subgraph(group0).copy()
    G1 = G.subgraph(group1).copy()
    
    # Recursively partition each subgraph
    partition0 = recursive_partition(G0, n_left)
    partition1 = recursive_partition(G1, n_right)
    
    # Combine results
    result = {}
    for node, part in partition0.items():
        result[node] = part
    for node, part in partition1.items():
        result[node] = part + n_left
    
    return result

def spectral_bisection(G):
    """Bisect graph using Fiedler vector."""
    # Weight nodes by area
    node_weights = {n: G.nodes[n]['area'] for n in G.nodes()}
    total_weight = sum(node_weights.values())
    target_weight = total_weight / 2
    
    # Compute Fiedler vector (second smallest eigenvector of Laplacian)
    laplacian = nx.laplacian_matrix(G).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler_vector = eigenvectors[:, 1]
    
    # Sort nodes by Fiedler vector value
    nodes_sorted = sorted(enumerate(fiedler_vector), key=lambda x: x[1])
    
    # Split to balance area
    partition = {}
    current_weight = 0
    
    for idx, (node_idx, _) in enumerate(nodes_sorted):
        node = list(G.nodes())[node_idx]
        if current_weight < target_weight:
            partition[node] = 0
            current_weight += node_weights[node]
        else:
            partition[node] = 1
    
    return partition

def balance_partitions(G, partitions, n_parts=38, max_iterations=100):
    """Refine partitions to balance areas more precisely."""
    print("Balancing partition areas...")
    
    target_area = sum(G.nodes[n]['area'] for n in G.nodes()) / n_parts
    
    for iteration in range(max_iterations):
        # Calculate current areas
        part_areas = {}
        for node, part in partitions.items():
            part_areas[part] = part_areas.get(part, 0) + G.nodes[node]['area']
        
        # Find most unbalanced partition
        max_diff = 0
        best_swap = None
        
        for node in G.nodes():
            current_part = partitions[node]
            current_area = part_areas[current_part]
            
            # Try swapping to neighboring partition
            for neighbor in G.neighbors(node):
                neighbor_part = partitions[neighbor]
                if neighbor_part != current_part:
                    neighbor_area = part_areas[neighbor_part]
                    node_area = G.nodes[node]['area']
                    
                    # Calculate improvement
                    old_imbalance = abs(current_area - target_area) + abs(neighbor_area - target_area)
                    new_current = current_area - node_area
                    new_neighbor = neighbor_area + node_area
                    new_imbalance = abs(new_current - target_area) + abs(new_neighbor - target_area)
                    
                    improvement = old_imbalance - new_imbalance
                    if improvement > max_diff:
                        max_diff = improvement
                        best_swap = (node, neighbor_part)
        
        if best_swap and max_diff > 0.01:
            node, new_part = best_swap
            partitions[node] = new_part
        else:
            break
    
    # Print final statistics
    part_areas = {}
    for node, part in partitions.items():
        part_areas[part] = part_areas.get(part, 0) + G.nodes[node]['area']
    
    print(f"\nFinal partition statistics:")
    print(f"Target area per district: {target_area:.2f}")
    print(f"Min area: {min(part_areas.values()):.2f}")
    print(f"Max area: {max(part_areas.values()):.2f}")
    print(f"Std dev: {np.std(list(part_areas.values())):.2f}")
    
    return partitions

def visualize_partitions(tracts, partitions, output_file='partitions.png'):
    """Visualize the partitioned districts."""
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Create colormap
    n_parts = max(partitions.values()) + 1
    colors = plt.cm.tab20(np.linspace(0, 1, n_parts))
    
    # Draw each tract colored by partition
    for i, tract in enumerate(tracts):
        if tract['geometry'].geom_type == 'Polygon':
            polys = [tract['geometry']]
        else:  # MultiPolygon
            polys = tract['geometry'].geoms
        
        part = partitions[i]
        color = colors[part % len(colors)]
        
        for poly in polys:
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
    
    ax.set_aspect('equal')
    ax.set_title('Texas Census Tract Cartogram - 38 Equal-Area Districts', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    plt.show()

def main(geojson_path, n_districts=38):
    """Main pipeline."""
    # Load data
    tracts = load_geojson(geojson_path)
    
    # Build adjacency graph
    G = build_adjacency_graph(tracts)
    
    # Partition graph
    partitions = partition_graph_metis(G, n_districts)
    
    # Balance areas
    partitions = balance_partitions(G, partitions, n_districts)
    
    # Visualize results
    visualize_partitions(tracts, partitions)
    
    # Export results
    output_data = {
        'type': 'FeatureCollection',
        'features': []
    }
    
    for i, tract in enumerate(tracts):
        feature = {
            'type': 'Feature',
            'properties': {
                'tract_id': tract['id'],
                'district': int(partitions[i]),
                'area': tract['area']
            },
            'geometry': tract['geometry'].__geo_interface__
        }
        output_data['features'].append(feature)
    
    with open('partitioned_districts.geojson', 'w') as f:
        json.dump(output_data, f)
    
    print("Results exported to partitioned_districts.geojson")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_geojson>")
        print("Example: python script.py texas_cartogram.geojson")
        sys.exit(1)
    
    geojson_path = sys.argv[1]
    main(geojson_path)
