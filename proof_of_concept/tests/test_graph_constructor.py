import unittest
import os
import sys
import torch
import networkx as nx
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.graph_constructor import GraphConstructor


class TestGraphConstructor(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for graph constructor tests"""
        # Sample genes
        self.genes = [f"gene_{i}" for i in range(5)]
        
        # Sample ligand-receptor pairs
        self.lr_pairs = [
            ("gene_0", "gene_1"),
            ("gene_2", "gene_3")
        ]
        
        # Sample cell type assignments
        self.cell_ids = ["cell_0", "cell_1", "cell_2"]
        self.cell_types = {"cell_0": "type_A", "cell_1": "type_A", "cell_2": "type_B"}
        
        # Sample prior GRNs
        self.prior_grns = {
            "type_A": nx.DiGraph(),
            "type_B": nx.DiGraph()
        }
        
        # Add edges to prior GRNs
        for gene_source in self.genes:
            self.prior_grns["type_A"].add_node(gene_source)
            self.prior_grns["type_B"].add_node(gene_source)
            
            for gene_target in self.genes:
                if gene_source != gene_target:
                    if np.random.random() < 0.3:  # 30% chance of edge
                        self.prior_grns["type_A"].add_edge(gene_source, gene_target)
                    if np.random.random() < 0.3:  # 30% chance of edge
                        self.prior_grns["type_B"].add_edge(gene_source, gene_target)
        
        # Sample gene expression history
        self.gene_expression_history = {}
        for cell_id in self.cell_ids:
            self.gene_expression_history[cell_id] = {}
            for gene_idx in range(len(self.genes)):
                self.gene_expression_history[cell_id][gene_idx] = {}
                for t in range(5):  # 5 time points
                    self.gene_expression_history[cell_id][gene_idx][t] = np.random.normal(0, 1)
        
        # Sample cell positions
        self.cell_positions = {}
        for cell_id in self.cell_ids:
            self.cell_positions[cell_id] = {}
            for t in range(5):  # 5 time points
                self.cell_positions[cell_id][t] = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
        
        # Create graph constructor
        self.graph_constructor = GraphConstructor(
            genes=self.genes,
            ligand_receptor_pairs=self.lr_pairs,
            cell_type_assignments=self.cell_types,
            prior_grns=self.prior_grns
        )
    
    def test_initialization(self):
        """Test that the GraphConstructor initializes correctly"""
        # Check receptor and ligand genes are identified
        self.assertEqual(self.graph_constructor.receptor_genes, {"gene_1", "gene_3"})
        self.assertEqual(self.graph_constructor.ligand_genes, {"gene_0", "gene_2"})
        
        # Check gene indices
        self.assertEqual(self.graph_constructor.gene_indices["gene_0"], 0)
        self.assertEqual(self.graph_constructor.gene_indices["gene_4"], 4)
    
    def test_construct_base_graph(self):
        """Test base graph construction"""
        # Construct a base graph for cell_0
        base_graph = self.graph_constructor.construct_base_graph("cell_0")
        
        # Check all genes are included
        for gene in self.genes:
            self.assertIn(gene, base_graph.nodes())
        
        # Check receptor nodes are created and connected to all genes
        for receptor_gene in self.graph_constructor.receptor_genes:
            receptor_node = f"r_{receptor_gene}"
            self.assertIn(receptor_node, base_graph.nodes())
            
            # Receptor should be connected to all genes
            for gene in self.genes:
                self.assertTrue(base_graph.has_edge(receptor_node, gene))
        
        # Check ligand nodes are created and connected to their genes
        for ligand_gene in self.graph_constructor.ligand_genes:
            ligand_node = f"l_{ligand_gene}"
            self.assertIn(ligand_node, base_graph.nodes())
            
            # Ligand should be connected to its gene
            self.assertTrue(base_graph.has_edge(ligand_gene, ligand_node))
    
    def test_update_graph_with_neighbors(self):
        """Test graph updating with neighbor information"""
        # First construct a base graph
        base_graph = self.graph_constructor.construct_base_graph("cell_0")
        
        # Update the graph with neighbor information
        time_point = 2
        updated_graph = self.graph_constructor.update_graph_with_neighbors(
            graph=base_graph,
            cell_id="cell_0",
            cell_positions=self.cell_positions,
            time_point=time_point,
            gene_expression_history=self.gene_expression_history,
            distance_threshold=50.0  # Large threshold to ensure neighbors
        )
        
        # Check that input ligand nodes from neighbors are added
        for neighbor_id in ["cell_1", "cell_2"]:
            for ligand_gene in self.graph_constructor.ligand_genes:
                input_ligand_node = f"l_{neighbor_id}_{ligand_gene}"
                self.assertIn(input_ligand_node, updated_graph.nodes())
                
                # Check that input ligand is connected to appropriate receptor
                for _, receptor_gene in self.lr_pairs:
                    if ligand_gene == self.lr_pairs[0][0] and receptor_gene == self.lr_pairs[0][1]:
                        receptor_node = f"r_{receptor_gene}"
                        self.assertTrue(updated_graph.has_edge(input_ligand_node, receptor_node))
                    elif ligand_gene == self.lr_pairs[1][0] and receptor_gene == self.lr_pairs[1][1]:
                        receptor_node = f"r_{receptor_gene}"
                        self.assertTrue(updated_graph.has_edge(input_ligand_node, receptor_node))
    
    def test_assign_node_features(self):
        """Test feature assignment to nodes"""
        # First construct a base graph and update with neighbors
        base_graph = self.graph_constructor.construct_base_graph("cell_0")
        time_point = 3
        updated_graph = self.graph_constructor.update_graph_with_neighbors(
            graph=base_graph,
            cell_id="cell_0",
            cell_positions=self.cell_positions,
            time_point=time_point,
            gene_expression_history=self.gene_expression_history,
            distance_threshold=50.0  # Large threshold to ensure neighbors
        )
        
        # Assign features with all time lags = 1
        delta_gl = delta_lr = delta_rg = delta_gg = 1
        pyg_graph = self.graph_constructor.assign_node_features(
            graph=updated_graph,
            cell_id="cell_0",
            time_point=time_point,
            gene_expression_history=self.gene_expression_history,
            delta_gl=delta_gl,
            delta_lr=delta_lr,
            delta_rg=delta_rg,
            delta_gg=delta_gg
        )
        
        # Check that PyTorch Geometric Data object is returned
        self.assertIsInstance(pyg_graph.x, torch.Tensor)
        
        # Check feature dimensions
        # Number of nodes should match number of features
        self.assertEqual(len(updated_graph.nodes()), pyg_graph.x.shape[0])
        
        # Each feature should be 1-dimensional (scalar)
        self.assertEqual(pyg_graph.x.shape[1], 1)
        
        # Check that gene_node_indices are stored
        self.assertTrue(hasattr(pyg_graph, 'gene_node_indices'))
        self.assertEqual(len(pyg_graph.gene_node_indices), len(self.genes))


if __name__ == '__main__':
    unittest.main() 