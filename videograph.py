"""
This module defines the VideoGraph class, which is used to represent the video graph.
"""

import json
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class VideoGraph:
    """
    This class defines the VideoGraph class, which is used to represent the video graph.
    """
    def __init__(self, max_img_embeddings=10, max_audio_embeddings=10, img_matching_threshold=0.3, audio_matching_threshold=0.7):
        """Initialize a video graph with nodes for faces, voices and text events.
        
        Args:
            max_img_embeddings: Maximum number of image embeddings per face node
            max_audio_embeddings: Maximum number of audio embeddings per voice node
        """
        self.nodes = {}  # node_id -> node object
        self.edges = {}  # (node_id1, node_id2) -> edge weight
        self.max_img_embeddings = max_img_embeddings
        self.max_audio_embeddings = max_audio_embeddings
        self.next_node_id = 0
        self.img_matching_threshold = img_matching_threshold
        self.audio_matching_threshold = audio_matching_threshold

        # Maintain ordered text nodes
        self.text_nodes = []  # List of text node IDs in insertion order
        self.text_nodes_contents = np.array([])  # Numpy array of processed text contents for searching

    def _process_text(self, text):
        """Process text for searching by removing special characters and converting to lowercase"""
        # Remove special characters and convert to lowercase
        processed = ''.join(c.lower() for c in text if c.isalnum() or c.isspace())
        return processed

    def where_text(self, search_text):
        """Search for text in text_nodes_contents and return indices where found
        
        Args:
            search_text: Text to search for
            
        Returns:
            List of indices where text was found
        """
        processed_search = self._process_text(search_text)
        return np.where(np.char.find(self.text_nodes_contents, processed_search) != -1)[0]

    class Node:
        def __init__(self, node_id, node_type):
            self.id = node_id
            self.type = node_type  # 'img', 'voice', 'episodic' or 'semantic'
            self.embeddings = []
            self.metadata = {}

    def add_img_node(self, img_embedding):
        """Add a new face node with initial image embedding(s).
        
        Args:
            img_embedding: Single embedding or list of embeddings
        """
        node = self.Node(self.next_node_id, 'img')
        if isinstance(img_embedding, list):
            node.embeddings.extend(img_embedding[:self.max_img_embeddings])
        else:
            node.embeddings.append(img_embedding)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node.id

    def add_voice_node(self, audio_embedding):
        """Add a new voice node with initial audio embedding(s).
        
        Args:
            audio_embedding: Single embedding or list of embeddings
        """
        node = self.Node(self.next_node_id, 'voice')
        if isinstance(audio_embedding, list):
            node.embeddings.extend(audio_embedding[:self.max_audio_embeddings])
        else:
            node.embeddings.append(audio_embedding)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node.id

    def add_text_node(self, text, text_type='episodic'):
        """Add a new text node with episodic or semantic content.
        
        Args:
            text: Text content
            text_type: Type of text node ('episodic' or 'semantic')
        """
        if text_type not in ['episodic', 'semantic']:
            raise ValueError("text_type must be either 'episodic' or 'semantic'")

        node = self.Node(self.next_node_id, text_type)
        node.metadata['text'] = text['content']
        node.embeddings.append(text['embedding'])
        self.nodes[self.next_node_id] = node
        self.text_nodes.append(self.next_node_id)  # Add to ordered list

        # Process and add text content
        processed_text = self._process_text(text)
        self.text_nodes_contents = np.append(self.text_nodes_contents, processed_text)

        self.next_node_id += 1
        return node.id

    def add_embedding(self, node_id, embeddings):
        """Add embeddings to an existing node. If total embeddings exceed max limit,
        randomly select embeddings to keep.
        
        Args:
            node_id: ID of target node
            embeddings: Single embedding or list of embeddings
            
        Returns:
            Boolean indicating success
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        if node.type == 'img':
            max_emb = self.max_img_embeddings
        elif node.type == 'voice':
            max_emb = self.max_audio_embeddings
        else:
            return False

        # Combine existing and new embeddings
        all_embeddings = node.embeddings + embeddings

        # If exceeding max limit, randomly select embeddings
        if len(all_embeddings) > max_emb:
            node.embeddings = random.sample(all_embeddings, max_emb)
        else:
            node.embeddings = all_embeddings

        return True

    def add_edge(self, node_id1, node_id2, weight=1.0):
        """Add or update bidirectional weighted edges between two nodes.
        Text-to-text connections are not allowed between same type text nodes."""
        if (node_id1 in self.nodes and node_id2 in self.nodes and
            not (self.nodes[node_id1].type == self.nodes[node_id2].type and 
                 self.nodes[node_id1].type in ['episodic', 'semantic'])):
            # Add both directions with same weight
            self.edges[(node_id1, node_id2)] = weight
            self.edges[(node_id2, node_id1)] = weight
            return True
        return False

    def update_edge_weight(self, node_id1, node_id2, delta_weight):
        """Update weight of existing bidirectional edge."""
        if (node_id1, node_id2) in self.edges:
            # Update both directions
            self.edges[(node_id1, node_id2)] += delta_weight
            self.edges[(node_id2, node_id1)] += delta_weight
            # if the weight is less than or equal to 0, remove the edge
            if self.edges[(node_id1, node_id2)] <= 0:
                del self.edges[(node_id1, node_id2)]
                del self.edges[(node_id2, node_id1)]
                print(f"Edge removed between {node_id1} and {node_id2}")
                
            return True
        return False

    def reinforce_node(self, node_id, delta_weight=1):
        """Reinforce all edges connected to the given node.
        
        Args:
            node_id: ID of the node to reinforce
            delta_weight: Amount to increase edge weights by (default: 1)
            
        Returns:
            int: Number of edges reinforced
        """
        if node_id not in self.nodes:
            return 0
            
        reinforced_count = 0
        for (n1, n2) in list(self.edges.keys()):  # Create a list to avoid modification during iteration
            if n1 == node_id or n2 == node_id:
                self.update_edge_weight(n1, n2, delta_weight)
                reinforced_count += 1
                
        return reinforced_count

    def weaken_node(self, node_id, delta_weight=1):
        """Weaken all edges connected to the given node.
        
        Args:
            node_id: ID of the node to weaken
            delta_weight: Amount to decrease edge weights by (default: 1)
            
        Returns:
            int: Number of edges weakened
        """
        if node_id not in self.nodes:
            return 0
            
        weakened_count = 0
        for (n1, n2) in list(self.edges.keys()):  # Create a list to avoid modification during iteration
            if n1 == node_id or n2 == node_id:
                self.update_edge_weight(n1, n2, -delta_weight)  # Use negative delta_weight to decrease
                weakened_count += 1
                
        return weakened_count

    def get_connected_nodes(self, node_id):
        """Get all nodes connected to given node."""
        connected = set()  # Use set to avoid duplicates due to bidirectional edges
        for (n1, n2), _ in self.edges.items():
            if n1 == node_id:
                connected.add(n2)
            elif n2 == node_id:
                connected.add(n1)
        return list(connected)

    def get_text_nodes_in_order(self):
        """Get text nodes in their insertion order."""
        return self.text_nodes.copy()

    def _cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(emb1, emb2) / (norm1 * norm2)

    def _average_similarity(self, embeddings1, embeddings2):
        """Calculate average cosine similarity between two lists of embeddings."""
        if not embeddings1 or not embeddings2:
            return 0

        similarities = []
        for emb1 in embeddings1:
            for emb2 in embeddings2:
                similarities.append(self._cosine_similarity(emb1, emb2))
        return np.mean(similarities)

    def search_img_nodes(self, query_embeddings):
        """Search for face nodes using image embeddings.
        
        Args:
            query_embeddings: Single embedding or list of embeddings
            threshold: Minimum similarity score threshold
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        if not isinstance(query_embeddings, list):
            query_embeddings = [query_embeddings]

        threshold = self.img_matching_threshold

        results = []
        for node_id, node in self.nodes.items():
            if node.type == 'img':
                similarity = self._average_similarity(query_embeddings, node.embeddings)
                if similarity >= threshold:
                    results.append((node_id, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_voice_nodes(self, query_embeddings):
        """Search for voice nodes using audio embeddings.
        
        Args:
            query_embeddings: Single embedding or list of embeddings
            threshold: Minimum similarity score threshold
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        if not isinstance(query_embeddings, list):
            query_embeddings = [query_embeddings]

        threshold = self.audio_matching_threshold

        results = []
        for node_id, node in self.nodes.items():
            if node.type == 'voice':
                similarity = self._average_similarity(query_embeddings, node.embeddings)
                if similarity >= threshold:
                    results.append((node_id, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def visualize_video_graph(self):
        """Visualize the video graph using networkx."""
        G = nx.Graph()
        for node_id, node in self.nodes.items():
            G.add_node(node_id, type=node.type)
            if node.type == 'img':
                G.nodes[node_id]['embeddings'] = node.embeddings
            elif node.type == 'voice':
                G.nodes[node_id]['embeddings'] = node.embeddings

        for (node_id1, node_id2), weight in self.edges.items():
            G.add_edge(node_id1, node_id2, weight=weight)

        nx.draw(G, with_labels=True)
        plt.show()

    def get_video_graph(self):
        """Get the video graph as a networkx graph."""
        return nx.Graph(self.edges)

    def get_video_graph_json(self):
        """Get the video graph as a json object."""
        return json.dumps(self.edges)
